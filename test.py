import os, time, glob
import numpy as np
from imageio import imread, imwrite
from skimage.measure import compare_psnr, compare_ssim
import torch
import torchvision.transforms as transforms

from utils import *
from option import args
from model.__init__ import make_model

import argparse

parser = argparse.ArgumentParser(description='SADNET')
# File paths
# parser.add_argument('--gt_src_path', type=str, default="./dataset/test/",
#                     help='testing clear image path, if not, set None')
# parser.add_argument('--noise_src_path', type=str, default="./dataset/test/color_sig50/",
#                     help='testing noisy image path')
parser.add_argument('--gt_src_path', type=str, default="/hdd4T_2/",
                    help='testing clear image path, if not, set None')
parser.add_argument('--noise_src_path', type=str, default="/hdd4T_1/cm/codes/DataSet/NoiseData/sig50/",
                    help='testing noisy image path')
parser.add_argument('--test_items', default=["CBSD68", "Kodak24"],
                    help='testing dataset')
parser.add_argument('--result_png_path', type=str, default="./result/SADNET_color_sig50/",
                    help='result directory')
parser.add_argument('--ckpt_dir', type=str, default="./ckpt/SADNET_color_sig50/",
                    help='model directory')

# Hardware specifications
parser.add_argument('--gpu', type=str, default="0",
                    help='GPUs')
args = parser.parse_args()

def evaluate_net():

    noise_src_folder_list = []
    dst_png_path_list = []
    if args.gt_src_path:
        gt_src_folder_list = []


    for item in args.test_items:
        noise_tmp = sorted(glob.glob(args.noise_src_path + item + '/'))
        noise_src_folder_list.extend(noise_tmp)
        dst_png_path_list.append(args.result_png_path + item + '/'])
        if args.gt_src_path:
            gt_src_folder_list.extend(sorted(glob.glob(args.gt_src_path + item + '/')))

    if args.gt_src_path:
        psnr = np.zeros(len(gt_src_folder_list))
        ssim = np.zeros(len(gt_src_folder_list))
    test_time = np.zeros(len(noise_src_folder_list))

    if torch.cuda.is_available():
        model = torch.load(ckpt_dir + 'model_*.pth')
        model = model.cuda()
    else:
        #model = torch.load(ckpt_dir + 'model_*.pth', map_location='cpu')
        print('There are not available cuda devices !')

    model.eval()

    #=================#
    for i in range(len(gt_src_folder_list)):
        in_files = glob.glob(noise_src_folder_list[i] + '*')
        in_files.sort()
        if args.gt_src_path:
            gt_files = glob.glob(gt_src_folder_list[i] + '*')
            gt_files.sort()
        create_dir(dst_png_path_list[i])

        for ind in range(len(in_files)):
            if args.gt_src_path:
                clean = imread(gt_files[ind]).astype(np.float32) / 255
                clean = clean[0:(clean.shape[0]//8)*8, 0:(clean.shape[1]//8)*8]

            img_test = imread(in_files[ind]).astype(np.float32) / 255
            img_test = img_test[0:(img_test.shape[0]//8)*8, 0:(img_test.shape[1]//8)*8]

            img_test = transforms.functional.to_tensor(img_test)
            img_test = img_test.unsqueeze_(0).float()
            if torch.cuda.is_available():
                img_test = img_test.cuda()

            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                out_image = model(img_test)
            torch.cuda.synchronize()
            if ind > 0:
                test_time[i] += (time.time() - start_time)
            print("took: %4.4fs" % (time.time() - start_time))
            print("process folder:%s" % src_folder_list[i])
            print("[*] save images")

            rgb = out_image.cpu().detach().numpy().transpose((0,2,3,1))
            if img_test.ndim == 3:
                rgb = np.clip(rgb[0], 0, 1)
            elif img_test.ndim == 2:
                rgb = np.clip(rgb[0, :, :, 0], 0, 1)

            rgb = np.uint8(rgb*255)
            # save image
            #imwrite(dst_png_path_list[i] + img_name + ".png", rgb)

            if args.gt_src_path:
                clean = np.uint8(clean*255)

                psnr[i] += compare_psnr(clean, rgb)

                if clean.ndim == 2:
                    ssim[i] += compare_ssim(clean, rgb))
                elif clean.ndim == 3:
                    ssim[i] += compare_ssim(clean, rgb, multichannel=True)

        test_time[i] = test_time[i] / (len(in_files)-1)
        if args.gt_src_path:
            psnr[i] = psnr[i] / len(in_files)
            ssim[i] = ssim[i] / len(in_files)
        #===========


    #print psnr,ssim
    for i in range(len(gt_src_folder_list)):
        print('src_folder: %s: ' %(gt_src_folder_list[i]))
        if args.gt_src_path:
            print('psnr: %f, ssim: %f, average time: %f' % (psnr[i], ssim[i], test_time[i]))
        else:
            print('average time: %f' % (test_time[i]))

    return 0

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    evaluate_net()
