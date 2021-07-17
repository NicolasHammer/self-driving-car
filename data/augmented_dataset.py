import random

import torch
from torch.utils.data import Dataset

from data.manipulate_image import random_augment


class Augmented_Dataset(Dataset):
    def __init__(self, data, is_training, device, WIDTH, HEIGHT):
        self.data = torch.Tensor(data).to(device)
        self.is_training = is_training
        self.width = WIDTH
        self.height = HEIGHT

    def __len__(self):
        return len(self.data)

    def __get_item__(self):
        random_index = random.randint(0, len(self.data) - 1)
        image, choice = self.data[random_index]

        if self.is_training:
            im, ch = random_augment(image, choice)
        else:
            im = image
            ch = choice

        im = im.reshape(self.width, self.height, 1)

        return im, ch
