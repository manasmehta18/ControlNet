import os
import numpy as np
import PIL
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms


class DataBase(Dataset):
    def __init__(self,
                 json_path,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 **kwargs
                 ):
        self.json_path = json_path
        self.data_root = data_root
        with open(self.json_path, "r") as f:
            cg_index = json.load(f)

        self.image_captions = []

        for index in cg_index['annotations']:
            if os.path.isdir(self.data_root):
                _, _, files = next(os.walk(self.data_root))
                file_count = len(files)
                for img in os.listdir(self.data_root):
                    if img == str(index['image_id']) + ".jpg" or img == str(index['image_id']) + ".tif" or img == str(index['image_id']) + ".jpeg" or img == str(index['image_id']) + ".JPEG":
                        self.image_captions.append((os.path.join(self.data_root,img),index['caption']))

        print(self.image_captions)
                   
        # self.image_captions = [(os.path.join(self.data_root,str(index['id']),img),index['caption']) for index in cg_index['annotations'] for img in os.listdir(os.path.join(self.data_root,str(index['id'])))]

        self._length = len(self.image_captions)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path, cap = self.image_captions[i]
        # example = dict((k, self.labels[k][i]) for k in self.labels)
        example = {}
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # image_gs = image.convert('L')
        # img = np.array(image_gs).astype(np.uint8)
        # stacked = np.stack((img,)*3, axis=-1)
        # image = Image.fromarray(stacked)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example['caption'] = cap

        return example

# constable, va, lionel
# boudin, lee
# cox, lionel, watts
# gogh, images, landscape, sketch

class DataTrain(DataBase):
    def __init__(self, **kwargs):
        super().__init__(json_path="/mnt/island/usr/mvm7168/diffusion/cse597-style-token/annotations/va/va__4.json", data_root="/mnt/island/usr/mvm7168/diffusion/cse597-style-token/va", size=kwargs['config']['size'], **kwargs)


class DataValidation(DataBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_path="/mnt/island/usr/mvm7168/diffusion/cse597-style-token/annotations/va/va__4.json", data_root="/mnt/island/usr/mvm7168/diffusion/cse597-style-token/va",
                         flip_p=flip_p, size=kwargs['config']['size'], **kwargs)


