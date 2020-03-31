import glob
from PIL import Image
import numpy as np
import torch
import random
import h5py
import torch.utils.data as data

class Dataset_from_read(data.Dataset):

    def __init__(self, src_path, sigma, transform=None):
        self.path = glob.glob(src_path + '*.png')
        self.path.sort()
        self.sigma = sigma
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.path[index])

        if self.transform:
            clean = self.transform(clean)

        noise = torch.randn(clean.size()).mul_(self.sigma/255.0)
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean

    def __len__(self):
        return len(self.path)

class Dataset_from_h5(data.Dataset):

    def __init__(self, src_path, sigma, gray=False, transform=None):
        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

        self.sigma = sigma
        self.gray = gray
        self.transform = transform

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        clean = Image.fromarray(np.uint8(data*255))
        h5f.close()

        if self.gray:
            clean = clean.convert('L')

        if self.transform:
            clean = self.transform(clean)

        #noise = torch.randn(clean.size()).mul_(self.sigma/255.0)
        noise = torch.normal(torch.zeros(clean.size()), self.sigma/255.0)
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean

    def __len__(self):
        return len(self.keys)

class Dataset_h5_real(data.Dataset):

    def __init__(self, src_path, patch_size=128,  gray=False, train=True):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        if train:
            random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size
        self.train = train
        self.gray = gray

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        h5f.close()

        if self.train:
            (H, W, C) = data.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            p = 0.5
            if random.random() > p: #RandomRot90
                patch = patch.transpose(1, 0, 2)
            if random.random() > p: #RandomHorizontalFlip
                patch = patch[:, ::-1, :]
            if random.random() > p: #RandomVerticalFlip
                patch = patch[::-1, :, :]
        else:
            patch = data

        patch = np.clip(patch.astype(np.float32)/255.0, 0.0, 1.0)
        if self.gray:
            noisy = np.expand_dims(patch[:, :, 0], -1)
            clean = np.expand_dims(patch[:, :, 1], -1)
        else:
            noisy = patch[:, :, 0:3]
            clean = patch[:, :, 3:6]

        noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))).float()
        clean = torch.from_numpy(np.ascontiguousarray(np.transpose(clean, (2, 0, 1)))).float()

        return noisy, clean

    def __len__(self):
        return len(self.keys)
