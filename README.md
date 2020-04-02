# SADNet
By Meng Chang, Qi Li, Huajun Feng, Zhihai Xu

This is the official Pytorch implementation of "Spatial-Adaptive Network for Single Image Denoising" [[Paper]](https://arxiv.org/abs/2001.10291)

## Prerequisites
* Python 3.6
* Pytorch 1.1
* CUDA 9.0

## Get Started
### Installation
The Deformable ConvNets V2 (DCNv2) module in our code adopts  [chengdazhi's implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch).

Please make sure your machine has a GPU, which is required for the DCNv2 module.

### Train
1. Download the training dataset and use `gen_dataset_*.py` to package them in the h5py format.
2. Place the h5py file in `/dataset/train/` or set the 'src_path' in `option.py` to your own path.
3. You can set any training parameters in `option.py`. After that, train the model:
```
cd $SADNet_ROOT
python train.py
```

### Test
1. Download the trained models from [Google Drive](https://drive.google.com/open?id=1ZTTPuVwTY_OwN-OHNy_S21JoGaQxnuup) or [Baidu Drive](https://pan.baidu.com/s/1RWiM3BQqPSTda4NYyi9bWQ) (4nbq) and place them in `/ckpt/`.
2. Place the testing dataset in `/dataset/test/` or set the testing path in `option.py` to your own path.
3. Set the parameters in `option.py` (eg. 'epoch_test', 'gray' and etc.)
3. test the trained models:
```
cd $SADNet_ROOT
python test.py
```

## Citation
If you find the code helpful in your research or work, please cite the following papers.
```
@article{chang2020spatial,
  title={Spatial-Adaptive Network for Single Image Denoising},
  author={Chang, Meng and Li, Qi and Feng, Huajun and Xu, Zhihai},
  journal={arXiv preprint arXiv:2001.10291},
  year={2020}
}
```

## Acknowledgments
The DCNv2 module in our code adopts from [chengdazhi's implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch).
