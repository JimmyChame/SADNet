import argparse

parser = argparse.ArgumentParser(description='SADNET')
# File paths
parser.add_argument('--src_path', type=str, default="./dataset/train/DIV2K_h5/train.h5",
                    help='training dataset path')
parser.add_argument('--val_path', type=str, default="./dataset/test/color_sig50/valid.h5",
                    help='validating dataset path, if not, set None')
parser.add_argument('--ckpt_dir', type=str, default="./ckpt/SADNET_color_sig50/",
                    help='model directory')
parser.add_argument('--log_dir', type=str, default="./log/SADNET_color_sig50/",
                    help='log directory')
# Hardware specifications
parser.add_argument('--gpu', type=str, default="1",
                    help='GPUs')

# Training parameters
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--patch_size', type=int, default=128,
                    help='training patch size')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='the number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--milestone', type=int, default=60,
                    help='the epochs for weight decay')
parser.add_argument('--val_epoch', type=int, default=1,
                    help='do validation per every N epochs')
parser.add_argument('--save_val_img', type=bool, default=True,
                    help='save the last validated image for comparison')
parser.add_argument('--val_patch_size', type=int, default=512,
                    help='patch size in validation dataset')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for every milestone')
parser.add_argument('--finetune', type=bool, default=False,
                    help='if finetune model, set True')
parser.add_argument('--init_epoch', type=int, default=0,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--t_loss', type=str, default='L2',
                    help='training loss')

# model
parser.add_argument('--real', type=bool, default=False,
                    help='real world noisy image or synthetic noisy image')
parser.add_argument('--sigma', type=int, default=50,
                    help='Gaussian noise std. added to the synthetic images')
parser.add_argument('--gray', type=bool, default=False,
                    help='if gray image, set True; if color image, set False')
parser.add_argument('--NetName', default='SADNET',
                    help='model name')
parser.add_argument('--n_channel', type=int, default=32,
                    help='number of convolutional channels')
parser.add_argument('--offset_channel', type=int, default=32,
                    help='number of offset channels')

# test
# File paths
parser.add_argument('--gt_src_path', type=str, default="./dataset/test/",
                    help='testing clear image path, if not, set None')
parser.add_argument('--noise_src_path', type=str, default="./dataset/test/color_sig50/",
                    help='testing noisy image path')
parser.add_argument('--test_items', default=["CBSD68", "Kodak24"],
                    help='testing dataset')
parser.add_argument('--result_png_path', type=str, default="./result/SADNET_color_sig50/",
                    help='result directory')
parser.add_argument('--ckpt_dir_test', type=str, default="./ckpt/SADNET_color_sig50/",
                    help='model directory')
parser.add_argument('--epoch_test', type=int, default=200,
                    help='the epoch for testing')

args = parser.parse_args()
