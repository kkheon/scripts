
import os
import glob
import re
import numpy as np
import seaborn as sns
from matplotlib import pyplot

from yuv_diff import yuv_diff
from yuv_diff import yuv_diff_temporal
from utils import plot_psnr_diff
from utils import plot_ssim_diff

import math


if __name__ == '__main__':
    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff_temporal"

    #w = 3840
    #h = 2160
    w = 1920
    h = 1072

    #block_size = 64
    #block_size = 128
    block_size = 256
    scale = 1

    start_frame = 0
    frame_size = 5


    ## path setting
    #list_label = sorted(glob.glob(os.path.join(label_path, "*.yuv")))
    #list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))


    # hm result
    #list_input = ['/home/kkheon/HM-16.9_CNN/bin/data_vsr/test/label_hm_arcnn/QP32/rec_BasketballDrive.yuv']

    # down-sampling data
    list_input = ['/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32/result_mf_vcnn_down_3/mf_vcnn_down_scene_53.yuv']

    # check of out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save each image's PSNR result as file.
    for idx, each_image in enumerate(list_input):
        each_input = list_input[idx]

        list_df_psnr, list_df_ssim, input_y = yuv_diff_temporal(each_input, start_frame, w, h, frame_size, block_size, scale)

        # save block-image
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):

                # block index
                y_in_block = int(y / block_size)
                x_in_block = int(x / block_size)

                psnrs = [None]  # first frame have no difference.
                ssims = [None]  # first frame have no difference.

                # get psnr, ssim value
                for each_frame_index in range(0, frame_size - 1):
                    # psnr
                    each_df_psnr = list_df_psnr[each_frame_index]
                    psnr = each_df_psnr.iloc[y_in_block][x_in_block]
                    psnrs.append(psnr)

                    # ssim
                    each_df_ssim = list_df_ssim[each_frame_index]
                    ssim = each_df_ssim.iloc[y_in_block][x_in_block]
                    ssims.append(ssim)

                result_imgs = []
                xlabels = []
                # pick block from input
                for each_frame_index in range(0, frame_size):
                    result_imgs.append(input_y[each_frame_index, y:y+block_size, x:x+block_size])
                    xlabels.append(start_frame + each_frame_index)

                # separate dir depending on PSNR
                psnr_diff = np.mean(psnrs[1:])
                psnr_diff_floor = math.floor(psnr_diff)
                str_psnr_diff_floor = str(psnr_diff_floor)
                output_path_psnr = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                if not os.path.exists(output_path_psnr):
                    os.makedirs(output_path_psnr)

                #plot_psnr_diff(result_imgs, psnrs, idx, x, y, save_dir=output_path_psnr)
                plot_ssim_diff(result_imgs, psnrs, xlabels, idx, x, y, save_dir=output_path_psnr)



