import os
import glob
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot

from yuv_diff import yuv_diff
from yuv_diff import yuv_diff_temporal
from yuv_diff import yuv_diff_n_frame

from utils import plot_psnr_diff
from utils import plot_ssim_diff
from utils import plot_framework_diff
from utils import parse_dec_bit_lcu
from utils import draw_scatter_plot_from_df

from read_yuv import read_yuv420

import math


if __name__ == '__main__':
    # options
    #do_plot_framework_diff = True
    do_plot_framework_diff = False

    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff_framework_multi"

    w = 1920
    h = 1080

    w_up = 3840
    h_up = 2160

    block_size = 64
    block_size_up = 128

    w_up_in_block = math.ceil(w_up / block_size)
    h_up_in_block = math.ceil(h_up / block_size)

    scale = 2

    start_frame = 0
    frame_size = 5

    path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32'

    depth_range = range(4, 36, 4)
    target_layer = [
#        'result_vdsr_4_bak_layer_4_lambda_0_depth/result_vdsr_4_layer_4_lambda_0_depth_',
        'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_',
    ]


    list_path = []
    for each_target_layer in target_layer:
        for each_depth_range in depth_range:
            list_path.append(each_target_layer + str(each_depth_range))

    #list_path = [
    #    'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_12',
    #    'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_16',
    #]

    # input 1 : label, High Resolution(HR)
    path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val'

    # input 5 : LR's HM result + up
    path_down_rec_up = path_target + '/result_mf_vcnn_up_4/QP32'
    prefix_down_rec_up = 'mf_vcnn_up_rec_mf_vcnn_down_'

    # check of out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataframe for saving raw data
    list_columns_raw = ['img_name', 'frame', 'x', 'y']
    df_raw = pd.DataFrame(columns=list_columns_raw)

    # input list
    #list_yuv_name = ['scene_53.yuv']
    #list_yuv_name = sorted(glob.glob(os.path.join(path_label, "*.yuv")))
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "*.yuv")))]
    list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_53.yuv")))]

    # diff label-LR-UP
    for each_path in list_path:

        if 'layer' in each_path:
            # save psnr as id : each_path
            _, id = each_path.split('/', 1)
            _, id = id.split('vdsr_4_', 1)
        else:
            id = each_path

        list_columns_raw_frm = list_columns_raw + [id]
        df_raw_frm = pd.DataFrame(columns=list_columns_raw_frm)

        # save each image's PSNR result as file.
        for idx, each_yuv in enumerate(list_yuv_name):

            # label
            each_label = os.path.join(path_label, each_yuv)

            ## diff label-HM
            #each_label_rec = os.path.join(path_label_rec, prefix_label_rec + each_yuv)
            #df_psnr_label_rec, df_sse_label_rec, label_y, input_y_label_rec = yuv_diff_n_frame(each_label, start_frame, each_label_rec, start_frame, w_up, h_up, block_size_up, scale, frame_size)


            each_input = os.path.join(path_target, each_path, 'QP32', prefix_down_rec_up + each_yuv)
            df_psnr, df_sse, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_input, start_frame, w_up, h_up, block_size_up, scale, frame_size)

            # save block-image
            w_in_block = int(w / block_size)
            for each_frame_index in range(0, frame_size):
                #df_raw_frm = pd.DataFrame(columns=list_columns_raw_frm)
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        # block index
                        y_up = int(y * 2)
                        x_up = int(x * 2)

                        # block index
                        y_in_block = int(y / block_size)
                        x_in_block = int(x / block_size)

                        # psnr : LR+HM+UP
                        each_df_psnr = df_psnr.loc[df_psnr['frame'] == each_frame_index]
                        each_block_psnr = each_df_psnr.iloc[y_in_block][x_in_block]

                        # to save the raw data
                        list_raw = [[each_yuv, each_frame_index, x, y, each_block_psnr]]
                        df_raw_frm = df_raw_frm.append(pd.DataFrame(list_raw, columns=list_columns_raw_frm))

                # after each frame
                ## append to total_df
                #df_raw = df_raw.append(df_raw_frm)


        # after each_path
        # merge to total_df
        df_raw = pd.merge(df_raw, df_raw_frm, on=list_columns_raw, how='outer')

    # calculate row-wise mean & std
    df_raw_psnr = df_raw[df_raw.columns.difference(['img_name', 'frame', 'x', 'y'])]
    df_raw['mean'] = df_raw_psnr.mean(axis=1)
    df_raw['std'] = df_raw_psnr.std(axis=1)
    df_raw['max'] = df_raw_psnr.max(axis=1)
    df_raw['min'] = df_raw_psnr.min(axis=1)
    df_raw['range'] = df_raw['max'] - df_raw['min']


    # to exclude some columns
    # df[df.columns.difference(['b'])]

    # save df_raw as csv
    filename_psnr = os.path.join(output_path, 'df_raw')
    df_raw.to_csv(filename_psnr + '.txt', sep=' ')

    # save based on range
    df_raw_range_01 = df_raw[df_raw['range'] > 0.1]
    filename_psnr = os.path.join(output_path, 'df_raw_range_01')
    df_raw_range_01.to_csv(filename_psnr + '.txt', sep=' ')

    df_raw_range_02 = df_raw[df_raw['range'] > 0.2]
    filename_psnr = os.path.join(output_path, 'df_raw_range_02')
    df_raw_range_02.to_csv(filename_psnr + '.txt', sep=' ')

    df_raw_range_03 = df_raw[df_raw['range'] > 0.3]
    filename_psnr = os.path.join(output_path, 'df_raw_range_03')
    df_raw_range_03.to_csv(filename_psnr + '.txt', sep=' ')

    df_raw_range_05 = df_raw[df_raw['range'] > 0.5]
    filename_psnr = os.path.join(output_path, 'df_raw_range_05')
    df_raw_range_05.to_csv(filename_psnr + '.txt', sep=' ')

    df_raw_range_10 = df_raw[df_raw['range'] > 1.0]
    filename_psnr = os.path.join(output_path, 'df_raw_range_10')
    df_raw_range_10.to_csv(filename_psnr + '.txt', sep=' ')

