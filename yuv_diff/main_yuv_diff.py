
import os
import glob
import re
import seaborn as sns
from matplotlib import pyplot

from yuv_diff import yuv_diff
from utils import plot_psnr_diff

import math


if __name__ == '__main__':
    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff"

    #w = 3840
    #h = 2160
    w = 1920
    h = 1072

    #block_size = 64
    #block_size = 128
    block_size = 256
    scale = 1

    frame_label = 2
    #frame_input = 0
    frame_input = 2


    ## path setting
    #list_label = sorted(glob.glob(os.path.join(label_path, "*.yuv")))
    #list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    # for test
    #list_label = ['/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/train/scene_0.yuv']
    #list_input_a = ['/home/kkheon/VSR-Tensorflow/data_vsr/train/result_mf_vcnn_up_4/QP32/img_mf_vcnn_up_rec_mf_vcnn_down_scene_0.yuv']

    # for diff
    #list_label = ['/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val/scene_53.yuv']
    #list_input_a = ['/home/kkheon/dataset/myanmar_v1/orig_hm/val/QP32/rec_scene_53.yuv']
    #list_input_b = ['/home/kkheon/HM-16.9_CNN/bin/data_vsr/val/label_hm_arcnn/QP32/rec_scene_53.yuv']

    list_label = ['/home/kkheon/test_images/ClassB_chopped/BasketballDrive_1920x1080_50_100fr_1920x1072.yuv']
    list_input_a = ['/home/kkheon/HM-16.9_CNN/bin/data_vsr/test/label_hm_arcnn/QP32/rec_BasketballDrive.yuv']
    list_input_b = ['/data/kkheon/dataset/hevc_ctc/ClassB_1072p_hm/QP32/rec_BasketballDrive.yuv']

    # check of  out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save each image's PSNR result as file.
    for idx, each_image in enumerate(list_label):

        each_label = list_label[idx]
        each_input_a = list_input_a[idx]

        df_psnr_a, df_sse, label_y, input_y_a = yuv_diff(each_label, frame_label, each_input_a, frame_input, w, h, block_size, scale)

        # parsing filename
        _, in_filename = each_image.rsplit('/', 1)
        in_filename, _ = in_filename.rsplit('.', 1)

        # df to file
        filename_psnr = os.path.join(output_path, 'df_psnr_a_' + in_filename)
        df_psnr_a.to_csv(filename_psnr + '.txt', sep=' ')

        # df to image
        pyplot.figure(figsize=(20, 10))
        sns_plot = sns.heatmap(df_psnr_a, annot=True, vmin=20, vmax=80)
        fig = sns_plot.get_figure()
        fig.savefig(filename_psnr + '.png')

        # image b
        each_input_b = list_input_b[idx]
        df_psnr_b, df_sse, label_y, input_y_b = yuv_diff(each_label, frame_label, each_input_b, frame_input, w, h, block_size, scale)

        # df to file
        filename_psnr = os.path.join(output_path, 'df_psnr_b_' + in_filename)
        df_psnr_b.to_csv(filename_psnr + '.txt', sep=' ')

        # df to image
        pyplot.figure(figsize=(20, 10))
        sns_plot = sns.heatmap(df_psnr_b, annot=True, vmin=20, vmax=80)
        fig = sns_plot.get_figure()
        fig.savefig(filename_psnr + '.png')

        #=== save diff as image ===#
        df_psnr_diff = df_psnr_b - df_psnr_a

        # df to file
        filename_psnr = os.path.join(output_path, 'df_psnr_diff_' + in_filename)
        df_psnr_diff.to_csv(filename_psnr + '.txt', sep=' ')

        # df to image
        pyplot.figure(figsize=(40, 10))
        sns_plot = sns.heatmap(df_psnr_diff, annot=True, vmin=-10, vmax=10, fmt=".2f", cmap="RdBu")
        fig = sns_plot.get_figure()
        fig.savefig(filename_psnr + '.png')

        # save block-image
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                # pick block from input
                sub_label_y = label_y[y:y+block_size, x:x+block_size]
                sub_input_y_a = input_y_a[y:y+block_size, x:x+block_size]
                sub_input_y_b = input_y_b[y:y+block_size, x:x+block_size]

                sub_input_y_diff = ((sub_input_y_b - sub_input_y_a + 128) / 2)

                # each PSNR result.
                y_in_block = int(y / block_size)
                x_in_block = int(x / block_size)
                psnr_a = df_psnr_a.iloc[y_in_block][x_in_block]
                psnr_b = df_psnr_b.iloc[y_in_block][x_in_block]
                psnr_diff = df_psnr_diff.iloc[y_in_block][x_in_block]

                # save results in a single plot
                result_imgs = [sub_label_y, sub_input_y_a, sub_input_y_b, sub_input_y_diff]
                psnrs = [None, psnr_a, psnr_b, psnr_diff]

                # separate dir depending on PSNR
                psnr_diff_floor = math.floor(psnr_diff)
                str_psnr_diff_floor = str(psnr_diff_floor)
                output_path_psnr = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                if not os.path.exists(output_path_psnr):
                    os.makedirs(output_path_psnr)

                plot_psnr_diff(result_imgs, psnrs, idx, x, y, save_dir=output_path_psnr)



