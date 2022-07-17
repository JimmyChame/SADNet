import numpy as np
import torch
import cv2
# ================================================================= #
# Reproduce the matlab psnr in python
# modified from skimage.metrics.peak_signal_noise_ratio
# ================================================================= #

def compare_psnr(input, gt, data_range=None):
    """
    Parameters
    ----------
    input: ndarray, dtype in [np.uint8, np.float16, np.float32, np.float64]
    gt: ndarray, dtype in [np.uint8, np.float16, np.float32, np.float64]
    data_range: the dynamic range of input data
    
    Returns
    -------
    psnr: float
    """
    if input.shape != gt.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if data_range is None:
        if input.dtype != gt.dtype:
            raise TypeError("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        if input.dtype == np.uint8:
            data_range = 255.0
        else:
            data_range = 1.0

    if np.max(input) > data_range:
        raise ValueError(
                "image_true has intensity values outside the range expected")

    input, gt = input.astype(np.float32), gt.astype(np.float32)
    error = np.mean((input - gt) ** 2, dtype=np.float64)

    return 10 * np.log10((data_range ** 2) / error)
# ================================================================= #
# Reproduce the matlab ssim in pytorch
# modified from https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
# ================================================================= #

def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)

def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())

def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)

def _ssim(img1, img2, conv, data_range):
    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = conv(img1)
    mu2 = conv(img2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv(img1 ** 2) - mu1_sq
    sigma2_sq = conv(img2 ** 2) - mu2_sq
    sigma12 = conv(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                        (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())

def compare_ssim(input, gt, data_range=None, device='cpu'):
    if input.shape != gt.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if data_range is None:
        if input.dtype != gt.dtype:
            raise TypeError("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        if input.dtype == np.uint8:
            data_range = 255.0
        else:
            data_range = 1.0

    if np.max(input) > data_range:
        raise ValueError(
                "image_true has intensity values outside the range expected")
    
    with torch.no_grad():
        input = torch.tensor(input).to(device).float()
        gt = torch.tensor(gt).to(device).float()

        if len(gt.shape) == 2:
            conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
            conv2d.weight.requires_grad = False
            conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
            conv = conv2d.to(device)
        elif len(gt.shape) == 3:
            conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
            conv3d.weight.requires_grad = False
            conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
            conv = conv3d.to(device)
        else:
            raise not NotImplementedError('only support 2d / 3d images.')
        return _ssim(input, gt, conv, data_range)
#=====================================================================#
# Reproduce the matlab ssim in python
# ================================================================= #
from scipy.ndimage import convolve as conv

def compare_ssim_py(input, gt, data_range=None):
    if input.shape != gt.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if data_range is None:
        if input.dtype != gt.dtype:
            raise TypeError("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        if input.dtype == np.uint8:
            data_range = 255.0
        else:
            data_range = 1.0

    if np.max(input) > data_range:
        raise ValueError(
                "image_true has intensity values outside the range expected")
    if len(gt.shape) == 2:
        weight = generate_2d_gaussian_kernel()
    elif len(gt.shape) == 3:
        weight = generate_3d_gaussian_kernel()
    else:
        raise not NotImplementedError('only support 2d / 3d images.')

    img1, img2 = input.copy(), gt.copy()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = conv(img1, weight, mode='nearest')
    mu2 = conv(img2, weight, mode='nearest')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv(img1 ** 2, weight, mode='nearest') - mu1_sq
    sigma2_sq = conv(img2 ** 2, weight, mode='nearest') - mu2_sq
    sigma12 = conv(img1 * img2, weight, mode='nearest') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                        (sigma1_sq + sigma2_sq + C2))


    return np.mean(ssim_map).astype(np.float32)


