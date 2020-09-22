import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path

data_path = Path('data')

def white_pixel_distribution(path=None):
    img = cv2.imread(str(path).replace('images', 'masks').replace(r'.jpg', r'_a.jpg'), cv2.IMREAD_GRAYSCALE)
    #rimg = cv2.resize(img, (576, 576))
    n_white_pix = np.sum(img >= 240)
    return n_white_pix

class AngyodysplasiaDataset(Dataset):
    def __init__(self, img_paths: list, to_augment=False, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        if self.limit is None:
            return len(self.img_paths)
        else:
            return self.limit

    def __getitem__(self, idx):
        if self.limit is None:
            img_file_name = self.img_paths[idx]
        else:
            img_file_name = np.random.choice(self.img_paths)

        img = load_image(img_file_name)

        if self.mode == 'train':
            mask = load_mask(img_file_name)

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name)


class MultiAngyodysplasiaDataset(Dataset):
    def __init__(self, img_paths: list, to_augment=False, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        if self.limit is None:
            return len(self.img_paths)
        else:
            return self.limit

    def __getitem__(self, idx):
        if self.limit is None:
            img_file_name = self.img_paths[idx]
        else:
            img_file_name = np.random.choice(self.img_paths)

        img = load_image(img_file_name)

        if self.mode == 'train':
            mask = load_mask(img_file_name)
            c = [1]
            #[angi]
            if white_pixel_distribution(img_file_name) > 0:
                c = torch.Tensor([1])
            else:
                c = torch.Tensor([0])

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float(), c
        else:
            label = 'angiodysplasia'
            if white_pixel_distribution(img_file_name) > 0:
                label = 'angiodysplasia'
            else:
                label = 'normal'

            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name), label

def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path).replace('images', 'masks').replace(r'.jpg', r'_a.jpg'), 0)
    return (mask > 0).astype(np.uint8)
