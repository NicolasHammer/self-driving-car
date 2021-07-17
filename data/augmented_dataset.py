import random

import numpy as np
from torch.utils.data import Dataset

from data.manipulate_image import random_augment


class Augmented_Dataset(Dataset):
    def __init__(self, data, is_training, WIDTH, HEIGHT):
        self.data = data
        self.is_training = is_training
        self.width = WIDTH
        self.height = HEIGHT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):
        random_index = random.randint(0, len(self.data) - 1)
        image, choice = self.data[random_index]

        if self.is_training:
            im, ch = random_augment(image, choice)
        else:
            im = image
            ch = choice

        im = im.reshape(1, self.width, self.height)

        return im.astype(np.float32), np.array(ch, np.float32)

import torch
x = torch.Tensor()
x.dtype