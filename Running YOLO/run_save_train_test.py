import argparse
import os
import subprocess
from glob import glob
import shutil 

# python run_save_train_test.py --img_list /hdd/dataplus2021/fcw/wnd_train_img_gen_ratio2.txt --lbl_list /hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio2.txt --out_dir /hdd/dataplus2021/share/exper_v0ratio2/
parser = argparse.ArgumentParser()
parser.add_argument('--img_list', type=str, default='none', help='directory of train imgs')                                             # if yes, input absolute path
parser.add_argument('--lbl_list', type=str, default='none', help='directory of train labels')                                           # if yes, input absolute path
parser.add_argument('--epochs', type=str, default='300', help='num epochs')
parser.add_argument('--out_dir', type=str, default='/home/fcw/batch_size_experiments/', help='directory to output imgs')
parser.add_argument('--img_list_val', type=str, default='none', help='directory of val imgs')                                             # if yes, input absolute path
parser.add_argument('--lbl_list_val', type=str, default='none', help='directory of val labels') 
parser.add_argument('--img_list_supplement', type=str, default='none', help='directory of supplement imgs')                                             # if yes, input absolute path
parser.add_argument('--lbl_list_supplement', type=str, default='none', help='directory of supplement labels') 
parser.add_argument('--version', type=str, default='v0', help='version num')
parser.add_argument('--device', type=str, default='1', help='gpu id')
parser.add_argument('--supplement_batch_size', type=str, default='1', help='supplement batch size')
parser.add_argument('--experiment', type=str, default='1', help='returns name of experiment')


opt = parser.parse_args()

def make_data_file(out_root, img_list, lbl_list, version, img_list_val, lbl_list_val, img_list_supplement, lbl_list_supplement, baseline_boolean):
    if not os.path.exists(out_root):                                                                    # make root dir
        os.makedirs(out_root)

    try:
        f = open(out_root + 'train_data_' + version + '.data', 'r+')
        f.truncate(0)
    except:
        pass

    with open(out_root + 'train_data_' + version + '.data', 'w') as f:                                     # create master label text file
        f.write('train=' + img_list + '\n')
        f.write('train_label=' + lbl_list + '\n')
        f.write('classes=1\n')
        if not baseline_boolean:
            f.write('supplement=' + img_list_supplement + '\n')
            f.write('supplement_label=' + lbl_list_supplement + '\n')
        f.write('valid=' + img_list_val + '\n')
        f.write('valid_label=' + lbl_list_val + '\n')
        # SWITCH
        # f.write('names=/hdd/dataplus2021/whtest/repro_bass_300/yolov3/data/wnd.names\n')
        f.write('names=./data/wnd.names\n')
        
        f.write('backup=backup/\n')
        f.write('eval=wnd')


def run_train(out_root, epochs, device, supplement_batch_size, baseline_boolean):
    if baseline_boolean:
        subprocess.run(['python', 'train.py',                                                    # train gp_gan
                        '--cfg', './cfg/yolov3-spp.cfg',
                        '--data', out_root + 'train_data_' + version + '.data',
                        '--img-size', '608',
                        '--epochs', epochs,
                        '--batch-size', '8',
                        '--device', device])
    else:
        subprocess.run(['python', 'train_mixed_batch.py',                                                    # train gp_gan
                        '--cfg', './cfg/yolov3-spp.cfg',
                        '--data', out_root + 'train_data_' + version + '.data',
                        '--img-size', '608',
                        '--epochs', epochs,
                        '--batch-size', '8',
                        '--supplement-batch-size', supplement_batch_size,
                        '--device', device])
        


def run_test(out_root, device):
    subprocess.run(['python', 'test.py',                                                    # test gp_gan
                    '--cfg', './cfg/yolov3-spp.cfg',
                    '--data', out_root + 'train_data_' + version + '.data',
                    '--img-size', '608',
                    '--weights', out_root + 'weights/last.pt', # DONE
                    '--device', device])


def copy_outputs(out_root, version):
    if not os.path.exists(out_root + version + '_outputs/'):                                                                    # make root dir
        os.makedirs(out_root + version + '_outputs/')

    file_names = ['PR_curve.png', 'precision.txt', 'recall.txt', 'results.png', 'results.txt', 'test_batch0_gt.jpg', 'test_batch0_pred.jpg', 'train_batch0.jpg', 'test_results.txt', 'ious.txt']
    file_names.extend((out_root + 'weights/best.pt', out_root + 'weights/last.pt'))
    for file in file_names:
        # SWITCH
        # shutil.copy2('/hdd/dataplus2021/whtest/repro_bass_300/yolov3/' + file, out_root + version + '_outputs/')

        #COPY FILE I CREATE
        shutil.copy2('./' + file, out_root + version + '_outputs/')


img_list = opt.img_list
lbl_list = opt.lbl_list
epochs = opt.epochs
out_root = opt.out_dir
img_list_val = opt.img_list_val
lbl_list_val = opt.lbl_list_val
version = opt.version
device = opt.device
img_list_supplement = opt.img_list_supplement
lbl_list_supplement = opt.lbl_list_supplement
supplement_batch_size = opt.supplement_batch_size
baseline_boolean = opt.experiment == "Baseline"

def main(img_list, lbl_list, out_root, epochs, version, device, img_list_supplement, lbl_list_supplement, supplement_batch_size, baseline_boolean):
    make_data_file(out_root, img_list, lbl_list, version, img_list_val, lbl_list_val, img_list_supplement, lbl_list_supplement, baseline_boolean)
    print("Made .data file\n")
    # print((out_root, img_list, lbl_list, version, img_list_val, lbl_list_val))
    # raise Exception

    #Change back
    run_train(out_root, epochs, device, supplement_batch_size, baseline_boolean)
    print("Finished Training\n")

    run_test(out_root, device)
    print("Finished Testing\n")

    copy_outputs(out_root, version)
    print("Copied outputs\n")


main(img_list, lbl_list, out_root, epochs, version, device, img_list_supplement, lbl_list_supplement, supplement_batch_size, baseline_boolean)
