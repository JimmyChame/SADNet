import h5py
from PIL import Image
import os
import numpy as np
import glob

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_patch(img, img_size=(512, 512), patch_size=(150, 150), stride=150, random_crop=False):
    count = 0
    patch_list = []
    if random_crop == True:
        crop_num = 100
        pos = [(np.random.randint(patch_size, img_size[0] - patch_size), np.random.randint(patch_size, img_size[1] - patch_size))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list

def gen_dataset_h5(gt_path, in_path, test_items, dst_path):

    gt_folder_list = []
    in_folder_list = []

    for item in test_items:
        tmp = sorted(glob.glob(gt_path + item))
        gt_folder_list.extend(tmp)
        tmp = sorted(glob.glob(in_path + item))
        in_folder_list.extend(tmp)

    h5py_name = dst_path + "valid.h5"
    h5f = h5py.File(h5py_name, 'w')

    count = 0
    for i in range(len(gt_folder_list)):
        noisy_imgs = glob.glob(in_folder_list[i] + '*.png')
        noisy_imgs.sort()
        gt_imgs = glob.glob(gt_folder_list[i] + '*.png')
        gt_imgs.sort()
        print('processing...' + str(count))
        for ind in range(len(noisy_imgs)):
            gt = np.array(Image.open(gt_imgs[ind]))
            noisy = np.array(Image.open(noisy_imgs[ind]))
            img = np.concatenate([noisy, gt], 2)
            img = img[0:(img.shape[0]//8)*8, 0:(img.shape[1]//8)*8]
            [h, w, c] = img.shape

            data = img.copy()
            h5f.create_dataset(str(count), shape=(h,w,c), data=data)
            count += 1
    h5f.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    gt_path = "./test/"
    in_path = "./test/color_sig50/"
    dst_path = "./test/color_sig50/"

    test_items = ["CBSD68/"]

    create_dir(dst_path)
    print("start...")
    gen_dataset_h5(gt_path, in_path, test_items, dst_path)
    print('end')
