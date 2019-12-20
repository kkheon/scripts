
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

from psnr import mse_to_psnr

import math


if __name__ == '__main__':
    # options
    #do_plot_framework_diff = True
    do_plot_framework_diff = False

    # settings
    output_path = "./result_diff_framework_adaptive_ratio"

    w_up = 3840
    h_up = 2160

    block_size = 64
    block_size_up = 128

    w_up_in_block = math.ceil(w_up / block_size)
    h_up_in_block = math.ceil(h_up / block_size)

    scale = 2

    start_frame = 0
    frame_size = 5
    #frame_size = 1

    fps = 60

    # for QP 32, 35, 34, 35, 33
    list_lambda = [
        49.222131819429933,
        360.15632358170024,
        273.42773026923868,
        360.15632358170024,
        73.983999999999995,
    ]


    # input 1 : label, High Resolution(HR)
    path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val'

    # input 2 : HR's HM result
    path_label_rec = '/home/kkheon/dataset/myanmar_v1/orig_hm/val/QP32'
    prefix_label_rec = 'rec_'

    # input 7 : HR's decoder_bit_lcu
    path_label_dec = path_label_rec + '_dec_v2'
    prefix_label_dec = 'decoder_bit_lcu_str_'

    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32'
    list_path_target = [
        '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_t1_mf1_vcnn_fixed_ipppp/result_QP32',
        '/data/kkheon/data_vsr_bak/val/val_t1_mf1_down_2x1/result_QP32',
        '/data/kkheon/data_vsr_bak/val/val_t1_mf1_down_1x2/result_QP32',
    ]
    list_scale = [
        (2, 2),
        (2, 1),
        (1, 2),
    ]
    list_id = [
        'down_2x2',
        'down_2x1',
        'down_1x2',
    ]

    n_target = len(list_path_target)

    # check of out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataframe for saving raw data
    #list_columns_raw = ['img_name', 'frame_idx', 'x', 'y', 'bit_org', 'psnr_org', 'bit_down/up', 'psnr_down/up', 'bit_diff', 'psnr_diff']
    list_columns_raw = ['img_name', 'frame_idx', 'x', 'y', 'bit_org', 'psnr_org', 'sse_org', 'rd_cost_org', ]

    for each_id in list_id:
        list_columns_raw += ['bit_'+each_id, 'psnr_'+each_id, 'sse_'+each_id, 'rd_cost_'+each_id, ]
    for each_id in list_id:
        list_columns_raw += ['rd_cost_diff_'+each_id, ]

    # min id, r, d
    list_columns_raw += ['min_id', 'min_r', 'min_d', ]
    list_columns_raw += ['down_min_id', 'down_min_r', 'down_min_d', ]

    df_raw = pd.DataFrame(columns=list_columns_raw)

    list_columns_summary = ['yuv_name',]
    list_columns_summary += ['bit_org_sum', 'bitrate_org_sum', 'psnr_org_sum']
    for each_id in list_id:
        list_columns_summary += ['bit_'+each_id, 'bitrate_'+each_id, 'psnr_'+each_id]

    # min
    list_columns_summary += ['bit_min_sum', 'bitrate_min_sum', 'psnr_min_sum']

    # min of down
    list_columns_summary += ['bit_down_min_sum', 'bitrate_down_min_sum', 'psnr_down_min_sum']

    df_summary = pd.DataFrame(columns=list_columns_summary)

    df_min_id = pd.DataFrame()
    df_down_min_id = pd.DataFrame()

    # input list
    #list_yuv_name = [
    #    'scene_53.yuv',
    #    'scene_54.yuv',
    #]
    #list_yuv_name = sorted(glob.glob(os.path.join(path_label, "*.yuv")))
    list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "*.yuv")))]

    # save each image's PSNR result as file.
    for idx, each_yuv in enumerate(list_yuv_name):
        #
        each_yuv_name, _ = each_yuv.split('.', 1)
        df_raw_yuv = pd.DataFrame(columns=list_columns_raw)

        # diff label-HM
        each_label = os.path.join(path_label, each_yuv)
        each_label_rec = os.path.join(path_label_rec, prefix_label_rec + each_yuv)
        df_psnr_label_rec, df_sse_label_rec, label_y, input_y_label_rec = yuv_diff_n_frame(each_label, start_frame, each_label_rec, start_frame, w_up, h_up, block_size_up, scale, frame_size)

        # get HR bitrate info
        # prefix
        prefix_down_dec = 'decoder_bit_lcu_str_'
        #bit_filename = '/home/kkheon/scripts/yuv_diff/dec/dec_hr/decoder_bit_lcu.txt'
        bit_filename = os.path.join(path_label_dec, prefix_down_dec + each_yuv_name + '.txt')
        df_bit_lcu_hr = parse_dec_bit_lcu(bit_filename)

        list_df_psnr_down_rec_up = []
        list_df_sse_down_rec_up = []
        list_input_y_down_rec_up = []
        list_df_bit_lcu_lr =[]

        for each_path_target in list_path_target:
            # input 3 : LR
            path_down = each_path_target + '/result_mf_vcnn_down_3'
            prefix_down = 'mf_vcnn_down_'
            # input 4 : LR's HM result
            path_down_rec = path_down + '_hm/QP32'
            prefix_down_rec = 'rec_mf_vcnn_down_'
            # input 5 : LR's HM result + up
            path_down_rec_up = each_path_target + '/result_mf_vcnn_up_4/QP32'
            prefix_down_rec_up = 'mf_vcnn_up_rec_mf_vcnn_down_'
            # input 6 : LR's decoder_bit_lcu
            path_down_dec = path_down_rec + '_dec_v2'
            prefix_down_dec = 'decoder_bit_lcu_str_mf_vcnn_down_'

            # diff label-LR-UP
            each_down_rec_up = os.path.join(path_down_rec_up, prefix_down_rec_up + each_yuv)
            df_psnr_down_rec_up, df_sse_down_rec_up, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_down_rec_up, start_frame, w_up, h_up, block_size_up, scale, frame_size)

            # get LR bitrate info
            # assumption : bit info file is already generated.
            bit_filename = os.path.join(path_down_dec, prefix_down_dec + each_yuv_name + '.txt')
            df_bit_lcu_lr = parse_dec_bit_lcu(bit_filename)

            # append to list
            list_df_psnr_down_rec_up.append(df_psnr_down_rec_up)
            list_df_sse_down_rec_up.append(df_sse_down_rec_up)
            list_input_y_down_rec_up.append(input_y_down_rec_up)
            list_df_bit_lcu_lr.append(df_bit_lcu_lr)

        # save block-image
        w_in_block = int(w_up / block_size_up)
        for each_frame_index in range(0, frame_size):
            df_raw_frm = pd.DataFrame(columns=list_columns_raw)

            # lambda for each frame
            each_lambda = list_lambda[each_frame_index]

            # make a frame-level bitrate reshape
            list_df_bit_lcu_lr_frm = []
            for target_index, each_df_bit_lcu_lr in enumerate(list_df_bit_lcu_lr):

                # Step 1 : read scale
                target_scale_w, target_scale_h = list_scale[target_index]

                # Step 2 : read bitrate
                each_df_bitrate = each_df_bit_lcu_lr.loc[each_df_bit_lcu_lr['frame'] == each_frame_index]

                # Step 3 : calculate up-sampled block size
                target_block_size_h = block_size * target_scale_h
                target_block_size_w = block_size * target_scale_w

                h_target_in_block = math.ceil(h_up / target_block_size_h)
                w_target_in_block = math.ceil(w_up / target_block_size_w)

                each_df_bitrate_reshaped = pd.DataFrame(each_df_bitrate['bit'].values.reshape(h_target_in_block, w_target_in_block))

                if target_block_size_h != block_size_up:
                    # do horizontal sum    ( because of shape )
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()

                if target_block_size_w != block_size_up:
                    # do vertical sum    ( because of shape )
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()

                # append to list
                list_df_bit_lcu_lr_frm.append(each_df_bitrate_reshaped)


            for y_up in range(0, h_up, block_size_up):
                for x_up in range(0, w_up, block_size_up):
                    # block index

                    # block index
                    y_in_block = int(y_up / block_size_up)
                    x_in_block = int(x_up / block_size_up)

                    # get psnr, ssim value
                    #for each_frame_index in range(0, frame_size):
                    psnrs = [None, ]  # label, lr, enc(lr)
                    ssims = [None, ]
                    bitrates = [None, ]
                    list_sse = [None, ]
                    list_rd_cost = [None, ]

                    # plot single frame
                    #xlabels = ['ORG', 'LR', 'LR+HM', 'LR+HM+UP', 'ORG+HM']
                    xlabels = []
                    result_imgs = []

                    # append imgs
                    xlabels.append('ORG')
                    result_imgs.append(label_y[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])

                    #
                    xlabels.append('ORG+HM')
                    result_imgs.append(input_y_label_rec[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])

                    # psnr : label+HM
                    each_df_psnr_label_rec = df_psnr_label_rec.loc[df_psnr_label_rec['frame'] == each_frame_index]
                    each_block_psnr_label_rec = each_df_psnr_label_rec.iloc[y_in_block][x_in_block]
                    psnrs.append(each_block_psnr_label_rec)

                    # bitrate : label+HM
                    # sum 2x2 block's bits
                    # Step 1 : extract frame's bit info
                    each_df_bitrate = df_bit_lcu_hr.loc[df_bit_lcu_hr['frame'] == each_frame_index]

                    # Step 2 : extract bits from df & reshape values
                    each_df_bitrate_reshaped = pd.DataFrame(each_df_bitrate['bit'].values.reshape(h_up_in_block, w_up_in_block))

                    # Step 3 : 2x2 sum
                    #print('before reshape = %d' % each_df_bitrate_reshaped.values.sum())
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped.index / 2
                    each_df_bitrate_reshaped[-2] = each_df_bitrate_reshaped[-2].astype('int')
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.groupby(-2).sum()
                    each_df_bitrate_reshaped = each_df_bitrate_reshaped.transpose()
                    #print('after reshape = %d' % each_df_bitrate_reshaped.values.sum())

                    # Step 4 : add to list
                    bitrate = each_df_bitrate_reshaped.iloc[y_in_block][x_in_block]
                    bitrates.append(bitrate)

                    # sse
                    each_df_sse_down_rec_up = df_sse_label_rec.loc[df_sse_label_rec['frame'] == each_frame_index]
                    each_block_sse_down_rec_up = each_df_sse_down_rec_up.iloc[y_in_block][x_in_block]
                    list_sse.append(each_block_sse_down_rec_up)

                    # rd cost
                    each_rd_cost = each_block_sse_down_rec_up + each_lambda * bitrate
                    list_rd_cost.append(each_rd_cost)

                    for i in range(0, n_target):
                        # psnr : LR+HM+UP
                        each_df_psnr_down_rec_up = list_df_psnr_down_rec_up[i].loc[list_df_psnr_down_rec_up[i]['frame'] == each_frame_index]
                        each_block_psnr_down_rec_up = each_df_psnr_down_rec_up.iloc[y_in_block][x_in_block]
                        psnrs.append(each_block_psnr_down_rec_up)

                        xlabels.append(list_id[i])
                        result_imgs.append(list_input_y_down_rec_up[i][each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])

                        ## bitrate : LR+HM
                        #each_df_bitrate_old = list_df_bit_lcu_lr[i].loc[list_df_bit_lcu_lr[i]['frame'] == each_frame_index]
                        #block_index = y_in_block * w_in_block + x_in_block
                        #bitrate_old = each_df_bitrate_old.iloc[block_index]
                        ##bitrates.append(bitrate['bit'])

                        # bitrate
                        each_df_bitrate = list_df_bit_lcu_lr_frm[i]
                        bitrate = each_df_bitrate.iloc[y_in_block][x_in_block]
                        bitrates.append(bitrate)

                        # sse
                        each_df_sse_down_rec_up = list_df_sse_down_rec_up[i].loc[list_df_sse_down_rec_up[i]['frame'] == each_frame_index]
                        each_block_sse_down_rec_up = each_df_sse_down_rec_up.iloc[y_in_block][x_in_block]
                        list_sse.append(each_block_sse_down_rec_up)

                        # rd cost
                        each_rd_cost = each_block_sse_down_rec_up + each_lambda * bitrate
                        list_rd_cost.append(each_rd_cost)


                    #====== saving option ======#
                    reference_index = 1
                    main_target_index = 2
                    # separate dir depending on PSNR
                    #psnr_diff = psnrs[3] - psnrs[4]  # psnr(LR+HM+UP) - psnr(HM)
                    psnr_diff = psnrs[main_target_index] - psnrs[reference_index]  # psnr(LR+HM+UP) - psnr(HM)

                    if math.isinf(psnr_diff):
                        psnr_diff = 99.0

                    #psnr_diff_floor = math.floor(psnr_diff)
                    #str_psnr_diff_floor = str(psnr_diff_floor)
                    #output_path_psnr = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                    #if not os.path.exists(output_path_psnr):
                    #    os.makedirs(output_path_psnr)

                    # bitrate rate
                    if bitrates[reference_index] == 0:
                        if bitrates[main_target_index] == 0:
                            bitrate_diff_rate = 0.0
                        else:
                            bitrate_diff_rate = 1.0
                    else:
                        bitrate_diff_rate = (bitrates[reference_index] - bitrates[main_target_index]) / bitrates[reference_index]  # (bitrates(HM) - bitrates(LR+HM+UP)) / bitrates(HM)

                    bitrate_diff_rate_format = float("{0:.1f}".format(bitrate_diff_rate))

                    if do_plot_framework_diff == True:
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

                        plot_framework_diff(result_imgs, psnrs, bitrates, xlabels, idx, each_frame_index, x_up, y_up, save_dir=output_path_group)

                    # to save the raw data
                    list_raw = [each_yuv, each_frame_index, x_up, y_up, ]

                    list_raw += [bitrates[1], psnrs[1], list_sse[1], list_rd_cost[1]]
                    for id_index, each_id in enumerate(list_id):
                        list_raw += [bitrates[id_index+2], psnrs[id_index+2], list_sse[id_index+2], list_rd_cost[id_index+2]]

                    # rd_cost_diff
                    for id_index, each_id in enumerate(list_id):
                        list_raw += [list_rd_cost[id_index+2] - list_rd_cost[1]]

                    # find min rd
                    min_rd = np.min(list_rd_cost[1:])
                    argmin_rd = np.argmin(list_rd_cost[1:]) + 1
                    min_rd_r = bitrates[argmin_rd]
                    min_rd_d = list_sse[argmin_rd]

                    if argmin_rd == 1:
                        min_id = 'org'
                    else:
                        min_id = list_id[argmin_rd-2]

                    # add id and r, d
                    list_raw += [min_id, min_rd_r, min_rd_d]

                    # find min rd between down-sampling
                    min_rd = np.min(list_rd_cost[2:])
                    argmin_rd = np.argmin(list_rd_cost[2:]) + 2
                    min_rd_r = bitrates[argmin_rd]
                    min_rd_d = list_sse[argmin_rd]
                    min_id = list_id[argmin_rd-2]

                    # add id and r, d
                    list_raw += [min_id, min_rd_r, min_rd_d]

                    #list_raw += [ bitrate_diff_rate, psnr_diff ]
                    df_raw_frm = df_raw_frm.append(pd.DataFrame([list_raw], columns=list_columns_raw))

            # after each frame

            ## scatter plot => not working
            #scatter_plot_filename = output_path + '/scatter_plot_img_%d_frm_%d' % (idx, each_frame_index)
            #draw_scatter_plot_from_df(scatter_plot_filename, df_raw_frm, 'bit_diff', 'psnr_diff')

            # append to total_df    : memory is available?
            df_raw_yuv = df_raw_yuv.append(df_raw_frm)

        # after each_yuv
        df_raw = df_raw.append(df_raw_yuv)

        # from df_raw
        # 0. each mode's frame level average    => should I re-calculate this?
        # 1. pick mode whose rd_cost is minimum and its result.
        # 2. the ratio of minimum rd cost between down-sampling

        # calculate stats

        # summary
        # PSNR => avg, R => sum
        #list_columns_raw = ['img_name', 'frame_idx', 'x', 'y', 'bit_org', 'psnr_org', 'rd_cost_org', ]

        #for each_id in list_id:
        #    list_columns_raw += ['bit_'+each_id, 'psnr_'+each_id, 'rd_cost_'+each_id, ]
        #for each_id in list_id:
        #    list_columns_raw += ['rd_cost_diff_'+each_id, ]

        ## min id, r, d
        #list_columns_raw += ['min_id', 'min_r', 'min_d', ]
        #list_columns_raw += ['down_min_id', 'down_min_r', 'down_min_d', ]

        list_summary = [each_yuv_name]

        # to calcualte PSNR from SSE
        img_size = h_up * w_up * frame_size
        ref_value = 255.0 * 255.0

        # to calculate bitrate from bit
        # Double dScale   = dFps / 1000 / (Double)m_uiNumPic;
        # bits / dScale

        dscale = fps / 1000 / frame_size

        bit_sum = df_raw_yuv['bit_org'].sum()
        bitrate_sum = bit_sum * dscale
        sse_sum = df_raw_yuv['sse_org'].sum()
        psnr_sum = mse_to_psnr(ref_value, sse_sum / img_size)

        list_summary += [bit_sum, bitrate_sum, psnr_sum]

        for each_id in list_id:
            bit_sum = df_raw_yuv['bit_'+each_id].sum()
            bitrate_sum = bit_sum * dscale
            sse_sum = df_raw_yuv['sse_'+each_id].sum()
            psnr_sum = mse_to_psnr(ref_value, sse_sum / img_size)
            list_summary += [bit_sum, bitrate_sum, psnr_sum]

        # min
        bit_sum = df_raw_yuv['min_r'].sum()
        bitrate_sum = bit_sum * dscale
        sse_sum = df_raw_yuv['min_d'].sum()
        psnr_sum = mse_to_psnr(ref_value, sse_sum / img_size)
        list_summary += [bit_sum, bitrate_sum, psnr_sum]

        # min of down
        bit_sum = df_raw_yuv['down_min_r'].sum()
        bitrate_sum = bit_sum * dscale
        sse_sum = df_raw_yuv['down_min_d'].sum()
        psnr_sum = mse_to_psnr(ref_value, sse_sum / img_size)
        list_summary += [bit_sum, bitrate_sum, psnr_sum]

        # list_summary to df
        df_summary = df_summary.append(pd.DataFrame([list_summary], columns=list_columns_summary))

        # mode ratio using groupby
        # 1. count
        #stat_min_id = df_raw_yuv.groupby('min_id').count()
        #stat_min_id = df_raw_yuv.groupby('min_id').size().reset_index(name='counts')
        each_df_min_id = df_raw_yuv.groupby('min_id').size().to_frame(name=each_yuv_name)
        df_min_id = pd.concat([df_min_id, each_df_min_id[each_yuv_name]], axis=1)

        each_df_down_min_id = df_raw_yuv.groupby('down_min_id').size().to_frame(name=each_yuv_name)
        df_down_min_id = pd.concat([df_down_min_id, each_df_down_min_id[each_yuv_name]], axis=1)


    # after yuv-loop

    # save df_raw as csv
    filename_psnr = os.path.join(output_path, 'df_raw')
    df_raw.to_csv(filename_psnr + '.txt', sep=' ')


    # add avg row
    df_summary.loc['mean'] = df_summary.mean()
    filename_summary = os.path.join(output_path, 'df_raw_summary')
    df_summary.to_csv(filename_summary + '.txt', sep=' ')

    df_min_id['mean'] = df_min_id.mean(axis=1)
    df_min_id['ratio'] = df_min_id['mean'] / df_min_id['mean'].sum()
    filename_min_id = os.path.join(output_path, 'df_raw_min_id')
    df_min_id.to_csv(filename_min_id + '.txt', sep=' ')

    df_down_min_id['mean'] = df_down_min_id.mean(axis=1)
    df_down_min_id['ratio'] = df_down_min_id['mean'] / df_down_min_id['mean'].sum()
    filename_down_min_id = os.path.join(output_path, 'df_raw_down_min_id')
    df_down_min_id.to_csv(filename_down_min_id + '.txt', sep=' ')

