import os
import random, time
import torch
import torchvision.transforms as transforms
import numpy as np


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def RandomRot(img, angle=90, p=0.5):
    if random.random() > p:
        return transforms.functional.rotate(img, angle)
    return img

def step_lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.1):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cycle_lr_adjust(optimizer, epoch, base_lr=1e-5, max_lr=1e-4, step_size=10, gamma=1):
    cycle = np.floor(1 + epoch/(2  * step_size))
    x = np.abs(epoch/step_size - 2 * cycle + 1)
    scale =  gamma ** (epoch // (2 * step_size))
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
