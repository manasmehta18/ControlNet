import json
import cv2
import numpy as np

from torch.utils.data import Dataset

from annotator.util import resize_image, HWC3

# constable, lionel, lee, va, watts, boudin, cox

artist = "lionel"

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('annotations/' + artist + '.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # target = resize_image(HWC3(target), 512)
        # source = resize_image(HWC3(source), 512)

        target = cv2.resize(target, (512, 512))
        source = cv2.resize(source, (512, 512))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        source = np.stack((source,)*3, axis=-1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        target = np.stack((target,)*3, axis=-1)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

