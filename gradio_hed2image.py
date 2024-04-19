from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image 

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_hed = HEDdetector()

# constable, lionel, lee, va, watts, boudin, cox

# bonheur = 1, boudin = 2, constable = 3, courbet = 4, jones = 5, manet = 4

artist = "all"

model = create_model('./models/cldm_v15_prompt.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/' + artist + '/checkpoints/last.ckpt', location='cuda'))

# model.load_state_dict(load_state_dict('models/control_sd15_hed.pth', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) )

        input_image = HWC3(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) )
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "art": torch.tensor([3.0])}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], "art": torch.tensor([3.0])}

        # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        
        # samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
        #                                              shape, cond, verbose=False, eta=eta,
        #                                              unconditional_guidance_scale=scale,
        #                                              unconditional_conditioning=un_cond, x_T=encoded_image)

        #TODO: add the function for encoding image to the cldm
        

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        token = model.prompt_tokens[3]
        np.save(f"outputs/tokens/style_token", token.cpu())

        cv2.imwrite('outputs/new2.jpg', results[0])
        # cv2.imwrite('outputs/map.jpg', detected_map)  
        print("done")

    return [detected_map] + results


# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Control Stable Diffusion with HED Maps")
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(source='upload', type="numpy")
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button(label="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                 guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                 detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
#                 eta = gr.Number(label="eta (DDIM)", value=0.0)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
#         with gr.Column():
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#     ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


# block.launch(server_name='localhost', share=True)


def main():

    # sky
    bonheur = "paintings/bonheur/51.jpg"
    boudin = "paintings/boudin/65.jpg"
    constable = "paintings/constable/106.jpg"
    manet1 = "paintings/manet/75.jpg"

    # courbet = "paintings/courbet/69.jpg"
    # jones = "paintings/jones/31.jpg"
    # manet = "paintings/manet/58.jpg"

    # whole
    # bonheur = "paintings/bonheur/4.jpg"
    # boudin = "paintings/boudin/12.jpg"
    # constable = "paintings/constable/13.jpg"
    # manet1 = "paintings/manet/12.jpg"

    input_image = cv2.imread(constable)
    prompt = "a 19th-century Realist oil painting of clouds and the sky by the artist constable"  
    a_prompt = ""
    n_prompt = ""
    num_samples = 1
    image_resolution = 512 
    detect_resolution = 512
    ddim_steps = 20 
    guess_mode = False
    strength = 1.0 
    scale = 9.0 
    seed = 0  
    eta = 0.0

    process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

if __name__ == "__main__":
    main()