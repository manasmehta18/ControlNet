import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torch import nn


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    
    print(f'Loaded state_dict from [{ckpt_path}]')

    # state_dict['cond_stage_model.transformer.text_model.prompt_token'] = nn.Parameter(torch.randn(8, 768))
    state_dict['prompt_token'] = nn.Parameter(torch.randn(8, 768))

    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
