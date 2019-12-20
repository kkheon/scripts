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
from skimage.feature import hog

from read_yuv import read_yuv420

import math

import requests


if __name__ == '__main__':
    # options
    #do_plot_framework_diff = True
    do_plot_framework_diff = False

    do_save_raw = True
    do_save_raw_range = True
    do_save_heatmap = True
    do_save_df_direction = True

    #do_save_raw = False
    do_save_raw_range = False
    do_save_heatmap = False
    #do_save_df_direction = False

    # settings
    w = 1920
    h = 1080
    #block_size = 64
    block_size = 32

    scale = 2

    #w_up = 3840
    #h_up = 2160
    w_up = int(w * scale)
    h_up = int(h * scale)
    block_size_up = int(block_size * scale)

    w_up_in_block = math.ceil(w_up / block_size)
    h_up_in_block = math.ceil(h_up / block_size)

    start_frame = 0
    frame_size = 5

    ###=== target : val
    ### input 1 : label, High Resolution(HR)
    ##path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val'
    ##path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32'

    #=== target : val_SJTU
    path_label = '/data/kkheon/dataset/SJTU_4K_test/label'
    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_QP32'
    output_path = "./result_diff_framework_multi_plot_SJTU_cmp"


    ##=== depth-based analysis
    #depth_range = range(4, 36, 4)
    #target_layer = [
#   #     'result_vdsr_4_bak_layer_4_lambda_0_depth/result_vdsr_4_layer_4_lambda_0_depth_',
    #    #'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_',

    #    'result_vdsr_4_layer_6_depth_',
    #]

    #list_path = []
    #list_id = []
    #for each_target_layer in target_layer:
    #    for each_depth_range in depth_range:
    #        each_path = each_target_layer + str(each_depth_range) + '/QP32'
    #        list_path.append(each_path)

    #        # generate id
    #        # save psnr as id : each_path
    #        #_, id = each_path.split('/', 1)
    #        #_, id = id.split('vdsr_4_', 1)

    #        _, id = each_path.split('vdsr_4_', 1)
    #        list_id.append(id)

    ## append depth=64
    #list_path.append('result_vdsr_4_layer_6_lambda_0/QP32')
    #list_id.append('layer_6_depth_64')

    ## append n(layer)
    ## not now

    ## append VDSR
    #list_path.append('result_vdsr_4_layer_18_lambda_0/QP32')
    #list_id.append('layer_18')

    # append val
    #=== target : val
    # input 1 : label, High Resolution(HR)
    #path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val'
    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_3x1_layer_6/QP32'
    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_1x3_layer_6/QP32'

    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_3x1_layer_6_depth_8/QP32'
    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_1x3_layer_6_depth_16/QP32'
    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_1x3_layer_6_depth_16/QP32'

    #path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_vdsr_4_conv_1x3_layer_6/QP32'
    path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_vdsr_4_conv_3x1_layer_6/QP32'

    #output_path = "./result_diff_framework_multi_plot_SJTU"
    output_path = path_target + '/diff'
    #output_path = "./result_diff_framework_multi_plot_diff"

    list_epoch = [
        '009',
        '019',
        '029',
        '039',
        '049',
        '059',
        '069',
        '079',
        '089',
        '099',
        '109',
        '119',
    ]
    list_path = []
    list_id = []
    for each_epoch in list_epoch:
        list_path.append(each_epoch)
        list_id.append(each_epoch)

    ## cmp : val
    #output_path = "./result_diff_framework_multi_plot_diff"
    #list_path = [
    #    '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32/result_vdsr_4_layer_6_lambda_0/QP32',
    #    '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32/result_vdsr_4_layer_6_lambda_0_conv3x1/QP32',
    #    '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_QP32/result_vdsr_4_layer_6_lambda_0_conv1x3/QP32',
    #    '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_3x1_layer_6/QP32/059',
    #    '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val/result_vdsr_4_conv_1x3_layer_6/QP32/009',
    #]
    #list_id = [
    #    'conv_3x3',
    #    'conv_3x1',
    #    'conv_1x3',
    #    'conv_3x1_targeted',
    #    'conv_1x3_targeted',
    #]

    # cmp : val_SJTU
    list_path = [
        '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_QP32/result_vdsr_4_layer_6_lambda_0/QP32',
        '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_vdsr_4_conv_3x1_layer_6/QP32/089',
        '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_SJTU/result_vdsr_4_conv_1x3_layer_6/QP32/119',
    ]
    list_id = [
        'conv_3x3',
        'conv_3x1',
        'conv_1x3',
    ]

    ##=== just compare, then id is the problem.
    ##list_path = [
    ##    'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_12',
    ##    'result_vdsr_4_bak_layer_6_lambda_0_depth/result_vdsr_4_layer_6_lambda_0_depth_16',
    ##]
    #list_path = [
    #    'result_vdsr_4_layer_6_lambda_0',
    #    'result_vdsr_4_layer_6_lambda_0_conv3x1',
    #    'result_vdsr_4_layer_6_lambda_0_conv1x3',
    #]
    #list_id = [
    #    'conv_3x3',
    #    'conv_3x1',
    #    'conv_1x3'
    #]


    # input 5 : LR's HM result + up
    #path_down_rec_up = path_target + '/result_mf_vcnn_up_4/QP32'
    prefix_down_rec_up = 'mf_vcnn_up_rec_mf_vcnn_down_'

    # check of out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataframe for saving raw data
    #list_columns_raw = ['img_name', 'frame', 'x', 'y']
    list_columns_raw = ['img_name', 'frame', 'x', 'y', 'direction']
    list_columns_raw = list_columns_raw + list_id
    df_raw = pd.DataFrame(columns=list_columns_raw)

    # input list
    #list_yuv_name = ['scene_53.yuv']
    #list_yuv_name = sorted(glob.glob(os.path.join(path_label, "*.yuv")))

    list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "*.yuv")))]
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_53.yuv")))]
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_54.yuv")))]
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_56.yuv")))]
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_5?.yuv")))]
    #list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "scene_54_offset_2.yuv")))]


    # save each image's PSNR result as file.
    for idx, each_yuv in enumerate(list_yuv_name):
        print('[%d/%d] %s' % (idx, len(list_yuv_name), each_yuv))

        # save block-image
        w_in_block = int(w / block_size)
        for each_frame_index in range(0, frame_size):

            list_df_psnr = []
            list_rec = []

            # diff label-LR-UP
            for each_path_index, each_path in enumerate(list_path):
                if len(list_id) == len(list_path):
                    id = list_id[each_path_index]
                else:
                    id = each_path

                # label
                each_label = os.path.join(path_label, each_yuv)

                #each_input = os.path.join(path_target, each_path, 'QP32', prefix_down_rec_up + each_yuv)
                each_input = os.path.join(path_target, each_path, prefix_down_rec_up + each_yuv)
                df_psnr, df_sse, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_input, start_frame, w_up, h_up, block_size_up, scale, frame_size)

                # to list
                list_df_psnr.append(df_psnr)
                list_rec.append(input_y_down_rec_up)


            #list_columns_raw_frm = list_columns_raw + list_id
            list_columns_raw_frm = list_columns_raw
            df_raw_frm = pd.DataFrame(columns=list_columns_raw_frm)
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    # block index
                    y_up = int(y * 2)
                    x_up = int(x * 2)

                    # block index
                    y_in_block = int(y / block_size)
                    x_in_block = int(x / block_size)

                    # psnr : LR+HM+UP
                    #list_raw = [each_yuv, each_frame_index, x, y]
                    list_psnr = []
                    list_result_imgs = []
                    list_bitrates = []   # no need to read bitrate


                    # result
                    for each_df_index, each_df_psnr in enumerate(list_df_psnr):
                        #each_df_psnr = df_psnr.loc[df_psnr['frame'] == each_frame_index]
                        each_block_psnr = each_df_psnr.iloc[y_in_block][x_in_block]

                        # to save the raw data
                        list_psnr.append(each_block_psnr)
                        list_bitrates.append(None)

                        if do_plot_framework_diff == True:
                            # to list_plot
                            each_rec = list_rec[each_df_index]
                            list_result_imgs.append(each_rec[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up])

                    # calculate hog
                    sub_label_y = label_y[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up].squeeze()
                    sub_label_h, sub_label_w = sub_label_y.shape

                    fd = hog(sub_label_y, orientations=4, pixels_per_cell=(sub_label_h, sub_label_w),
                        cells_per_block=(1, 1), visualize=False)

                    # 0 : vertical, 2 : horizontal
                    direction = fd.argmax()

                    #list_raw = [each_yuv, each_frame_index, x, y] + list_psnr
                    list_raw = [each_yuv, each_frame_index, x, y, direction] + list_psnr
                    # after df_pnsr-loop
                    df_raw_frm = df_raw_frm.append(pd.DataFrame([list_raw], columns=list_columns_raw_frm))

                    # plot : block-level
                    if do_plot_framework_diff == True:

                        # to make save rule
                        # temporally : diff
                        # separate dir depending on PSNR
                        #psnr_diff = list_psnr[0] - list_psnr[1]

                        psnr_diff = max(list_psnr) - min(list_psnr)

                        ##if psnr_diff < 0:
                        ##    psnr_diff = psnr_diff * 10

                        #psnr_diff_floor = math.floor(psnr_diff)
                        #str_psnr_diff_floor = str(psnr_diff_floor)
                        #output_path_group = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                        output_path_group = os.path.join(output_path, 'group_direction_' + str(direction))

                        if not os.path.exists(output_path_group):
                            os.makedirs(output_path_group)

                        # label
                        list_psnr = [None] + list_psnr
                        list_bitrates = [None] + list_bitrates
                        if do_plot_framework_diff == True:
                            list_result_imgs = [label_y[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up]] + list_result_imgs

                        xlabels = ['ORG'] + list_id

                        plot_framework_diff(list_result_imgs, list_psnr, list_bitrates, xlabels, idx, each_frame_index, x, y, save_dir=output_path_group, img_name=each_yuv)


            # after each frame
            # append to total_df
            df_raw = df_raw.append(df_raw_frm)


        ## after each_path
        ## merge to total_df
        #df_raw = pd.merge(df_raw, df_raw_frm, on=list_columns_raw, how='outer')

    # calculate row-wise mean & std
    df_raw_psnr = df_raw[df_raw.columns.difference(['img_name', 'frame', 'x', 'y', 'direction'])]
    df_raw['mean'] = df_raw_psnr.mean(axis=1)
    df_raw['std'] = df_raw_psnr.std(axis=1)
    df_raw['max'] = df_raw_psnr.max(axis=1)
    df_raw['min'] = df_raw_psnr.min(axis=1)
    df_raw['range'] = df_raw['max'] - df_raw['min']

    #df_raw['range'] = df_raw_psnr[-1] - df_raw_psnr[0]


    # ols


    # to exclude some columns
    # df[df.columns.difference(['b'])]

    if do_save_raw == True:
        # save df_raw as csv
        filename_psnr = os.path.join(output_path, 'df_raw')
        df_raw.to_csv(filename_psnr + '.txt', sep=' ')

    if do_save_raw_range == True:
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


    ## temp for conv 3x3 vs conv 3x1
    #df_raw['diff'] = df_raw['conv_3x3'] - df_raw['conv_3x1']

    ## save based on diff
    #df_raw_diff_01 = df_raw[df_raw['diff'] > 0]
    #filename_psnr = os.path.join(output_path, 'df_raw_conv_3x3_is_better')
    #df_raw_diff_01.to_csv(filename_psnr + '.txt', sep=' ')

    #df_raw_diff = df_raw[df_raw['diff'] < 0]
    #filename_psnr = os.path.join(output_path, 'df_raw_conv_3x3_is_better')
    #df_raw_diff.to_csv(filename_psnr + '.txt', sep=' ')

    #list_diff_range = np.arange(0, -1, -0.1)
    #for each_diff_range in list_diff_range:
    #    df_raw_diff = df_raw[df_raw['diff'] < each_diff_range]
    #    filename_psnr = os.path.join(output_path, 'df_raw_conv_3x3_diff_' + "%.1f" % each_diff_range)
    #    df_raw_diff.to_csv(filename_psnr + '.txt', sep=' ')

    if do_save_heatmap == True:
        # save heat-map
        list_img_name = df_raw['img_name'].values
        list_img_name_unique = np.unique(list_img_name)

        for each_name in list_img_name_unique:
            each_name_df_raw = df_raw.loc[df_raw['img_name'] == each_name]

            for each_frame_index in range(0, frame_size):
                each_df_raw = each_name_df_raw.loc[each_name_df_raw['frame'] == each_frame_index]

                each_df_raw_range = each_df_raw.pivot(index='y', columns='x', values='range')

                # df to image
                #pyplot.figure(figsize=(20, 10))
                pyplot.figure(figsize=(40, 20))
                sns_plot = sns.heatmap(each_df_raw_range, annot=True, fmt='.2f')
                fig = sns_plot.get_figure()

                filename_psnr = os.path.join(output_path, 'sns_' + each_name + '_frm_' + str(each_frame_index))
                fig.savefig(filename_psnr + '.png')



    if do_save_df_direction == True:
        #=== consider 'direction'
        # extract direction=0
        df_raw_direction_0 = df_raw[df_raw['direction'] == 0]
        df_raw_direction_1 = df_raw[df_raw['direction'] == 1]
        df_raw_direction_2 = df_raw[df_raw['direction'] == 2]
        df_raw_direction_3 = df_raw[df_raw['direction'] == 3]

        # calculate avg psnr
        df_raw_direction_0.loc['mean'] = df_raw_direction_0.replace([np.inf, -np.inf], np.nan).mean()
        df_raw_direction_1.loc['mean'] = df_raw_direction_1.replace([np.inf, -np.inf], np.nan).mean()
        df_raw_direction_2.loc['mean'] = df_raw_direction_2.replace([np.inf, -np.inf], np.nan).mean()
        df_raw_direction_3.loc['mean'] = df_raw_direction_3.replace([np.inf, -np.inf], np.nan).mean()

        # to summary
        df_summary_direction = pd.DataFrame(columns=list_columns_raw)
        df_summary_direction.loc['direction_0'] = df_raw_direction_0.loc['mean']
        df_summary_direction.loc['direction_1'] = df_raw_direction_1.loc['mean']
        df_summary_direction.loc['direction_2'] = df_raw_direction_2.loc['mean']
        df_summary_direction.loc['direction_3'] = df_raw_direction_3.loc['mean']

        # save as txt
        filename_psnr = os.path.join(output_path, 'df_raw_direction_0')
        df_raw_direction_0.to_csv(filename_psnr + '.txt', sep=' ')
        filename_psnr = os.path.join(output_path, 'df_raw_direction_1')
        df_raw_direction_1.to_csv(filename_psnr + '.txt', sep=' ')
        filename_psnr = os.path.join(output_path, 'df_raw_direction_2')
        df_raw_direction_2.to_csv(filename_psnr + '.txt', sep=' ')
        filename_psnr = os.path.join(output_path, 'df_raw_direction_3')
        df_raw_direction_3.to_csv(filename_psnr + '.txt', sep=' ')

        filename_psnr = os.path.join(output_path, 'df_summary_direction_0')
        df_summary_direction.to_csv(filename_psnr + '.txt', sep=' ')


# send telegram message
bot_id = "598336934:AAGKtE6tL9D8Ky30v0Fx1ZKbOqB9u1KEb5o"
chat_id = "55913643"

host = os.uname()[1]
msg = "[%s] %s is done." % (host, __file__)
url = "https://api.telegram.org/bot" + bot_id + "/sendMessage?chat_id=" + chat_id + "&text=" + msg
r = requests.get(url)
