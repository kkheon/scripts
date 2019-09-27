
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

def parse_avg_time(filename):
    try:
        with open(filename) as data:
            lines = data.readlines()
            for each_line in lines:
                if 'TotalAvg' in each_line:
                    list_numbers = re.findall('[.0-9]+', each_line)
                    return list_numbers[5]

            return None

    except IOError as err:
        print('File error'+str(err))

if __name__ == '__main__':

    # size info
    # settings
    w = 1920
    h = 1080
    #block_size = 64
    block_size = 32

    scale = 2

    w_up = int(w * scale)
    h_up = int(h * scale)
    block_size_up = int(block_size * scale)

    w_up_in_block = math.ceil(w_up / block_size)
    h_up_in_block = math.ceil(h_up / block_size)

    frame_size = 5
    n_block = int(w / block_size) * int(h / block_size)

    # target data column name
    # conv_3x3, conv_3x1, conv_1x3, conv_3x1_targeted, conv_1x3_targeted,

    # read psnr dataframe
    filename = '/home/kkheon/scripts/yuv_diff/result_diff_framework_multi_plot_diff/df_raw.txt'
    df_psnr = pd.read_csv(filename, sep=" ")

    # read complexity
    path_cpu_time = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32/single_video'
    list_sub_path = [
        'result_vdsr_4_layer_6_single_video',
        'result_vdsr_4_conv_3x1_layer_6_single_video',
    ]

    list_id_complexity = [
        'conv_3x3',
        'conv_3x1',
        #'conv_1x3',
        #'conv_3x1_targeted',
        #'conv_1x3_targeted',
    ]

    list_columns = list_id_complexity
    df_complexity = pd.DataFrame(columns=list_columns)

    list_complexity = []
    for each_sub_path in list_sub_path:
        each_sub_path_target = os.path.join(path_cpu_time, each_sub_path, 'avg*.txt')

        list_avg_file = sorted(glob.glob(each_sub_path_target))

        # pick last one
        avg_filename = list_avg_file[-1]

        # parse and get time.
        avg_time = parse_avg_time(avg_filename)

        # should be divided by size
        avg_time_block = avg_time / frame_size / n_block

        # append to list
        list_complexity.append(avg_time_block)

    # to df
    df_complexity = df_complexity.append(pd.DataFrame([list_complexity], columns=list_columns))

    # match
    # for each block in df_psnr,
    # draw graph 

    #

