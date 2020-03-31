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


def gen_dataset(src_files, dst_path):
    create_dir(dst_path)
    h5py_name = dst_path + "train.h5"
    h5f = h5py.File(h5py_name, 'w')

    count = 0
    for src_path in src_path_list:
        file_path = glob.glob(src_path+'*')
        for file_name in file_path:
            if 'SIDD' in file_name:
                gt_imgs = glob.glob(file_name+'/*GT*.PNG')
                gt_imgs.sort()
                noisy_imgs = glob.glob(file_name+'/*NOISY*.PNG')
                noisy_imgs.sort()
                print('SIDD processing...' + str(count))
                for i in range(len(noisy_imgs)):
                    gt = np.array(Image.open(gt_imgs[i]))
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f.create_dataset(str(count), shape=(300,300,6), data=data)
                        count += 1


            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name+'/*Reference.bmp')
                full_imgs = glob.glob(file_name+'/*full.bmp')
                noisy_imgs = glob.glob(file_name+'/*Noisy.bmp')
                noisy_imgs.sort()

                ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
                full = np.array(Image.open(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                print('RENOIR processing...' + str(count))
                for i in range(len(noisy_imgs)):
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f.create_dataset(str(count), shape=(300,300,6), data=data)
                        count += 1

    h5f.close()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    src_path_list = ["./train/SIDD/SIDD_Medium_Srgb/Data/",
                    "./train/RENOIR/Mi3_Aligned/",
                    "./train/RENOIR/T3i_Aligned/",
                    "./train/RENOIR/S90_Aligned/",
                    ]
    dst_path = "./train/SIDD_RENOIR_h5/"

    create_dir(dst_path)
    print("start...")
    gen_dataset(src_path_list, dst_path)
    print('end')
