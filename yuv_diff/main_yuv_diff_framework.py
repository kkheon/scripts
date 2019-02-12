
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
    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff_framework"

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


    # input list
    # input 1 : label, High Resolution(HR)
    list_label = ['/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val/scene_53.yuv']
    # input 2 : HR's HM result
    list_label_rec = ['/home/kkheon/dataset/myanmar_v1/orig_hm/val/QP32/rec_scene_53.yuv']
    # input 3 : LR
    list_down = ['/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32/result_mf_vcnn_down_3/mf_vcnn_down_scene_53.yuv']
    # input 4 : LR's HM result
    list_down_rec = ['/data/kkheon/data_vsr_bak/val/val_t1_mf1_vcnn_fixed_ipppp/result_QP32/result_mf_vcnn_down_3_hm/QP32/rec_mf_vcnn_down_scene_53.yuv']
    # input 5 : LR's HM result + up
    list_down_rec_up = ['/data/kkheon/data_vsr_bak/val/val_t1_mf1_vcnn_fixed_ipppp/result_QP32/result_mf_vcnn_up_4/QP32/mf_vcnn_up_rec_mf_vcnn_down_scene_53.yuv']

    ## path setting
    #list_label = sorted(glob.glob(os.path.join(label_path, "*.yuv")))
    #list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    # hm result
    #list_input = ['/home/kkheon/HM-16.9_CNN/bin/data_vsr/test/label_hm_arcnn/QP32/rec_BasketballDrive.yuv']

    # bit info
    #list_dec_bit_lcu = ['/home/kkheon/scripts/yuv_diff/dec/decoder_bit_lcu.txt']

    # check of out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save each image's PSNR result as file.
    for idx, each_image in enumerate(list_down_rec):

        # load LR
        each_down_yuv = list_down_rec[idx]
        array_input_y, array_input_cbcr = read_yuv420(each_down_yuv, w, h, frame_size, start_frame=start_frame)
        input_y_down = array_input_y.squeeze(axis=3)

        # load LR+HM
        each_down_yuv = list_down_rec[idx]
        array_input_y, array_input_cbcr = read_yuv420(each_down_yuv, w, h, frame_size, start_frame=start_frame)
        input_y_down_rec = array_input_y.squeeze(axis=3)

        # diff label-HM
        each_label = list_label[idx]
        each_label_rec = list_label_rec[idx]
        df_psnr_label_rec, df_sse_label_rec, label_y, input_y_label_rec = yuv_diff_n_frame(each_label, start_frame, each_label_rec, start_frame, w_up, h_up, block_size_up, scale, frame_size)

        # diff label-LR-UP
        each_down_rec_up = list_down_rec_up[idx]
        df_psnr_down_rec_up, df_sse_down_rec_up, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_down_rec_up, start_frame, w_up, h_up, block_size_up, scale, frame_size)

        # get HR bitrate info
        bit_filename = '/home/kkheon/scripts/yuv_diff/dec/dec_hr/decoder_bit_lcu.txt'
        df_bit_lcu_hr = parse_dec_bit_lcu(bit_filename)

        # TODO : need a transform to down-sampled sized bit
        # sum 2x2 value into 1

        # get LR bitrate info
        # assumption : bit info file is already generated.
        bit_filename = '/home/kkheon/scripts/yuv_diff/dec/dec_lr/decoder_bit_lcu.txt'
        df_bit_lcu_lr = parse_dec_bit_lcu(bit_filename)


        # dataframe for diff-scatter plot
        list_columns = ['img_idx', 'frame_idx', 'x', 'y', 'bit_diff', 'psnr_diff']
        #df_diff = pd.DataFrame(columns=['img_idx', 'frame_idx', 'x', 'y', 'bit_diff', 'psnr_diff'])

        # save block-image
        w_in_block = int(w / block_size)
        for each_frame_index in range(0, frame_size):
            df_diff_frm = pd.DataFrame(columns=list_columns)
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    # block index
                    y_up = int(y * 2)
                    x_up = int(x * 2)

                    # block index
                    y_in_block = int(y / block_size)
                    x_in_block = int(x / block_size)

                    # get psnr, ssim value
                    #for each_frame_index in range(0, frame_size):
                    psnrs = [None, None, None]  # label, lr, enc(lr)
                    ssims = [None, None, None]

                    # psnr : LR-HM-UP
                    each_df_psnr_down_rec_up = df_psnr_down_rec_up.loc[df_psnr_down_rec_up['frame'] == each_frame_index]
                    each_block_psnr_down_rec_up = each_df_psnr_down_rec_up.iloc[y_in_block][x_in_block]
                    psnrs.append(each_block_psnr_down_rec_up)

                    # psnr : label-HM
                    each_df_psnr_label_rec = df_psnr_label_rec.loc[df_psnr_label_rec['frame'] == each_frame_index]
                    each_block_psnr_label_rec = each_df_psnr_label_rec.iloc[y_in_block][x_in_block]
                    psnrs.append(each_block_psnr_label_rec)

                    bitrates = [None, None]

                    # bitrate : LR-HM
                    each_df_bitrate = df_bit_lcu_lr.loc[df_bit_lcu_lr['frame'] == each_frame_index]
                    block_index = y_in_block * w_in_block + x_in_block
                    bitrate = each_df_bitrate.iloc[block_index]
                    bitrates.append(bitrate['bit'])

                    # add again for LR+HM+UP
                    bitrates.append(bitrate['bit'])

                    # sum 2x2 block's bits
                    # bitrate : label+HM
                    # Step 1 : extract frame's bit info
                    each_df_bitrate = df_bit_lcu_hr.loc[df_bit_lcu_hr['frame'] == each_frame_index]

                    # Step 2 : extract bits from df & reshape values
                    each_df_bitrate_reshaped = pd.DataFrame(each_df_bitrate['bit'].values.reshape(h_up_in_block, w_up_in_block))

                    # Step 3 : 2x2 sum
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()

                    # Step 4 : add to list
                    bitrate = each_df_bitrate_reshaped.iloc[y_in_block][x_in_block]
                    bitrates.append(bitrate)

                    #block_index = y_in_block * w_in_block + x_in_block
                    #bitrate = each_df_bitrate.iloc[block_index]
                    #bitrates.append(bitrate['bit'])

                    # plot single frame
                    xlabels = ['ORG', 'LR', 'LR+HM', 'LR+HM+UP', 'ORG+HM']
                    result_imgs = []

                    # append imgs
                    result_imgs.append(label_y[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])
                    result_imgs.append(input_y_down[each_frame_index, y:y+block_size, x:x+block_size])
                    result_imgs.append(input_y_down_rec[each_frame_index, y:y+block_size, x:x+block_size])
                    result_imgs.append(input_y_down_rec_up[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])
                    result_imgs.append(input_y_label_rec[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])

                    # separate dir depending on PSNR
                    psnr_diff = psnrs[3] - psnrs[4]  # psnr(LR+HM+UP) - psnr(HM)

                    if math.isinf(psnr_diff):
                        psnr_diff = 99.0

                    #psnr_diff_floor = math.floor(psnr_diff)
                    #str_psnr_diff_floor = str(psnr_diff_floor)
                    #output_path_psnr = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                    #if not os.path.exists(output_path_psnr):
                    #    os.makedirs(output_path_psnr)

                    # bitrate rate
                    if bitrates[4] == 0:
                        if bitrates[3] == 0:
                            bitrate_diff_rate = 0.0
                        else:
                            bitrate_diff_rate = 1.0
                    else:
                        bitrate_diff_rate = (bitrates[4] - bitrates[3]) / bitrates[4]  # (bitrates(HM) - bitrates(LR+HM+UP)) / bitrates(HM)

                    bitrate_diff_rate_format = float("{0:.1f}".format(bitrate_diff_rate))

                    # categorization
                    if psnr_diff >= 0:
                        if bitrate_diff_rate < 0.1:
                            group_index = '0'
                        elif bitrate_diff_rate < 0.3:
                            group_index = '3'
                        else:
                            group_index = '6'
                    elif psnr_diff < -3:
                        if bitrate_diff_rate < 0.1:
                            group_index = '2'
                        elif bitrate_diff_rate < 0.3:
                            group_index = '5'
                        else:
                            group_index = '8'
                    else:
                        if bitrate_diff_rate < 0.1:
                            group_index = '1'
                        elif bitrate_diff_rate < 0.3:
                            group_index = '4'
                        else:
                            group_index = '7'

                    output_path_group = os.path.join(output_path, 'group_' + group_index)

                    if not os.path.exists(output_path_group):
                        os.makedirs(output_path_group)

                    plot_framework_diff(result_imgs, psnrs, bitrates, xlabels, idx, each_frame_index, x, y, save_dir=output_path_group)

                    # save to df_diff
                    list_diff = [[idx, each_frame_index, x, y, bitrate_diff_rate, psnr_diff]]
                    df_diff_frm = df_diff_frm.append(pd.DataFrame(list_diff, columns=list_columns))

            # after each frame

            # scatter plot
            scatter_plot_filename = output_path + '/scatter_plot_img_%d_frm_%d' % (idx, each_frame_index)
            draw_scatter_plot_from_df(scatter_plot_filename, df_diff_frm, 'bit_diff', 'psnr_diff')

