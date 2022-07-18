import os, time, glob
import numpy as np
from imageio import imread, imwrite
#from skimage.measure import compare_psnr, compare_ssim
from metrics import compare_psnr, compare_ssim
import torch
import torchvision.transforms as transforms

from utils import *
from option import args
from model.__init__ import make_model


def evaluate_net():

    noise_src_folder_list = []
    dst_png_path_list = []
    if args.gt_src_path:
        gt_src_folder_list = []


    for item in args.test_items:
        noise_tmp = sorted(glob.glob(args.noise_src_path + item + '/'))
        noise_src_folder_list.extend(noise_tmp)
        dst_png_path_list.append(args.result_png_path + item + '/')
        if args.gt_src_path:
            gt_src_folder_list.extend(sorted(glob.glob(args.gt_src_path + item + '/')))

    if args.gt_src_path:
        psnr = np.zeros(len(gt_src_folder_list))
        ssim = np.zeros(len(gt_src_folder_list))
    test_time = np.zeros(len(noise_src_folder_list))

    # Build model
    if args.gray:
        input_channel, output_channel = 1, 1
    else:
        input_channel, output_channel = 3, 3

    model = make_model(input_channel, output_channel, args)

    if torch.cuda.is_available():
        model_dict = torch.load(args.ckpt_dir_test+'model_%04d_dict.pth' % args.epoch_test)
        model.load_state_dict(model_dict)
        model = model.cuda()
    else:
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

            img_name = os.path.split(in_files[ind])[-1].split('.')[0]
            noisy = imread(in_files[ind]).astype(np.float32) / 255
            noisy = noisy[0:(noisy.shape[0]//8)*8, 0:(noisy.shape[1]//8)*8]

            img_test = transforms.functional.to_tensor(noisy)
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
            print("process folder:%s" % noise_src_folder_list[i])
            print("[*] save images")

            rgb = out_image.cpu().detach().numpy().transpose((0,2,3,1))
            if noisy.ndim == 3:
                rgb = np.clip(rgb[0], 0, 1)
            elif noisy.ndim == 2:
                rgb = np.clip(rgb[0, :, :, 0], 0, 1)

            # save image
            imwrite(dst_png_path_list[i] + img_name + ".png", np.uint8(rgb*255))

            if args.gt_src_path:

                psnr[i] += compare_psnr(clean, rgb)
                if torch.cuda.is_available():
                    ssim[i] += compare_ssim(clean, rgb, device='cuda')
                else:
                    ssim[i] += compare_ssim(clean, rgb)
                '''
                if clean.ndim == 2:
                    ssim[i] += compare_ssim(clean, rgb)
                elif clean.ndim == 3:
                    ssim[i] += compare_ssim(clean, rgb, multichannel=True)
                '''

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
