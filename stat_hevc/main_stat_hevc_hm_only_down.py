
import os
import glob
import pandas as pd
import re

import json
import pickle

from stat_hevc import stat_hevc
from stat_psnr_summary import stat_psnr_summary as stat_psnr
from stat_vmaf import stat_vmaf

from stat_plot import plot_step_vmaf
from stat_plot import plot_step_vmaf_target


if __name__ == '__main__':

    list_id = [
        '2160',
        'VDSR(1080)',
        'VDSR(720)',
        'VDSR(544)',
        'CNN_DOWN(1080)',
        'CNN_DOWN(720)',
    ]

    list_dir = [
        '/data/kkheon/dataset/ultra_video_group/label_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_720_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_544_hm_5frm',

        '/data/kkheon/data_vsr_bak/test_ultra/lanczos_2160_to_1080/result_down_2_hm',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_720/result_down_2_hm',

    ]

    list_dir_up = [
        '',
        '/data/kkheon/data_vsr_bak/test_ultra/lanczos_2160_to_1080/result_vdsr_1_conv_3x3',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_720/result_vdsr_1_conv_3x3',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_544/result_vdsr_1_conv_3x3',

        '/data/kkheon/data_vsr_bak/test_ultra/lanczos_2160_to_1080/result_up_3',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_720/result_up_3',
    ]

    list_qp = range(12, 47, 1)
    #list_qp = [22, 32]

    # json file dir
    list_dir_filter_vmaf = [
        '/home/kkheon/scripts/vmaf_test/label_hm_5frm_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_hm_5frm_lanczos_1080_to_2160_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_hm_5frm_lanczos_720_to_2160_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_544_hm_5frm_lanczos_544_to_2160_vmaf',

        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_hm_5frm_lanczos_1080_to_2160_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_hm_5frm_lanczos_720_to_2160_vmaf',
    ]
    list_dir_vdsr_vmaf = [
        '',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_544_vmaf',

        '/data/kkheon/data_vsr_bak/test_ultra/lanczos_2160_to_1080/result_up_3_vmaf',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_720/result_up_3_vmaf',
    ]

    #
    run_multi_frame = True

    if run_multi_frame:
        path_prefix = ""
        down_filename = "result_*.txt"
        # for hm only
        down_dir_name = '.'

        #up_dir_name = '.'
        up_dir_name = 'result_vdsr*'
        #up_filename = "psnr_*" + ".txt"
        up_filename = "vsr_pred_*" + ".txt"
    else:
        #path_prefix = 'result_' + 'QP' + str(each_qp)

        # TODO : check if running
        # val_myanmar
        # path : result_QP32/result_mf_vcnn_down_3_hm/QP32
        # filename : result_mf_vcnn_down_scene_53.txt
        down_dir_name = 'result_mf_vcnn_down*_hm/QP*'
        down_filename = 'result_mf_vcnn_down_*.txt'

        # val_myanmar
        # path : result_QP32/result_mf_vcnn_up_4
        # file : ... just average file. not for each file.
        up_dir_name = 'result_mf_vcnn_up_*/QP*'
        up_filename = "psnr_*" + ".txt"

    df_total = pd.DataFrame()

    for i, each_dir in enumerate(list_dir):
        df_down = pd.DataFrame()
        df_up = pd.DataFrame()
        df_filter_vmaf = pd.DataFrame()
        df_vmaf = pd.DataFrame()

        path = each_dir
        each_id = list_id[i]

        each_dir_up = list_dir_up[i]

        each_dir_filter_vmaf = list_dir_filter_vmaf[i]
        each_dir_vmaf = list_dir_vdsr_vmaf[i]

        for each_qp in list_qp:

            # ========== down-sampled bitrate ========== #
            target_path = os.path.join(path, down_dir_name, 'QP' + str(each_qp))
            list_sub_dir = sorted(glob.glob(target_path))

            # add bicubic down-sampled hevc result.
            #target_path = os.path.join(each_dir, 'result_hm', 'QP' + str(each_qp))
            target_path = os.path.join(each_dir, 'QP' + str(each_qp))
            list_sub_dir.append(target_path)

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, down_filename)))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

            #==== VMAF : filter
            target_path = os.path.join(each_dir_filter_vmaf, 'QP' + str(each_qp))
            list_sub_dir = sorted(glob.glob(target_path))
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "*.json")))
                for each_txt in list_txt:
                    each_stat = stat_vmaf(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_filter_vmaf = df_filter_vmaf.append(each_frame_table)

            #==== VMAF : VDSR
            target_path = os.path.join(each_dir_vmaf, 'QP' + str(each_qp))
            list_sub_dir = sorted(glob.glob(target_path))
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "*.json")))
                for each_txt in list_txt:
                    each_stat = stat_vmaf(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_vmaf = df_vmaf.append(each_frame_table)

        # ========== up-sampled PSNR ========== #
        #target_path = os.path.join(each_dir_up, up_dir_name, 'QP' + str(each_qp))
        #target_path = os.path.join(each_dir_up, up_dir_name)
        target_path = os.path.join(each_dir_up)
        list_sub_dir = sorted(glob.glob(target_path))

        if each_dir_up != '':
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, up_filename)))
                for each_txt in list_txt:
                    each_stat = stat_psnr(each_txt, 'name_qp_frm')

                    each_frame_table = each_stat.get_frame_table()
                    df_up = df_up.append(each_frame_table)

        #===== save path =====#
        if each_dir_up == '':   # for label, there is no dir_up, so I give them their own path.
            out_dir = each_dir
        else:
            out_dir = each_dir_up
        #out_dir = each_dir

        #===== save df_down =====#
        ## type change
        #df_down[['psnr_y_up', 'ssim_up']] = df_down[['psnr_y_up', 'ssim_up']].astype(float)
        df_down[['psnr_y']] = df_down[['psnr_y']].astype(float)

        #========== Merge ==========#
        #merge_basis =['loop', 'name', 'qp', 'frm']
        merge_on = ['name', 'qp', 'frm']

        # merge with VMAF-filter
        df_down = pd.merge(df_down, df_filter_vmaf, on=merge_on, how='outer', suffixes=['', '_filter_vmaf'])
        df_down.rename(columns={'VMAF':'VMAF_filter'}, inplace=True)

        # add 'id' to df_total
        df_down['id'] = each_id

        ## to_file
        filename_down = os.path.join(out_dir, 'df_down.txt')
        df_down.to_csv(filename_down, sep=' ')

        # frame-level average
        #df_frame_level = df_down.groupby(['loop', 'frm', 'qp']).mean()
        df_frame_level = df_down.groupby(['frm', 'qp']).mean()

        # to_file
        filename_down = os.path.join(out_dir, 'df_down_frame_avg.txt')
        df_frame_level.to_csv(filename_down, sep=' ')

        # video-level average
        df_video_level = df_down.groupby(['name', 'qp'], as_index=False).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_down_video_avg.txt')
        df_video_level.to_csv(filename_merged, sep=' ')


        # check empty of df_up
        if df_up.empty == True:
            # === making df_total
            # append to df_total
            df_video_level['id'] = each_id
            df_total = df_total.append(df_video_level)

            continue

        # merge
        #df_up['id'] = each_id
        #df_merged = pd.merge(df_down, df_up, on='id', how='outer')
        df_merged = pd.merge(df_down, df_up, on=merge_on, how='outer', suffixes=['', '_up'])
        df_merged = df_merged.sort_values(['loop', 'name', 'epoch', 'frm', 'qp'])

        # drop the row which has null value.
        #df_merged = df_merged.dropna(how='any', axis=0)

        # drop duplicate rows
        df_merged = df_merged.drop_duplicates()

        # merge with VMAF
        df_merged = pd.merge(df_merged, df_vmaf, on=merge_on, how='outer', suffixes=['', '_vmaf'])

        filename_merged = os.path.join(out_dir, 'df_raw.txt')
        #df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
        df_merged.to_csv(filename_merged, index=None, sep=' ')

        # frame-level average
        df_frame_level = df_merged.groupby(['loop', 'epoch', 'frm', 'qp']).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_frame_avg.txt')
        df_frame_level.to_csv(filename_merged, sep=' ')

        # video-level average
        #df_video_level = df_merged.groupby(['loop', 'epoch', 'name', 'qp'], as_index=False).mean()
        df_video_level = df_merged.groupby(['loop', 'name', 'qp', 'epoch'], as_index=False).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_video_avg.txt')
        df_video_level.to_csv(filename_merged, sep=' ')

        # add 'id' to df_total
        df_video_level['id'] = each_id

        # append to df_total
        df_total = df_total.append(df_video_level)

        # qp-level average
        df_qp_level = df_merged.groupby(['loop', 'epoch', 'qp']).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_qp_avg.txt')
        df_qp_level.to_csv(filename_merged, sep=' ')


    # plot bitrate ladder
    out_dir = './'
    out_filename = os.path.join(out_dir, 'df_total.txt')
    df_total.to_csv(out_filename, sep=' ')
    #plot_step_vmaf(out_dir, df_total)
    plot_step_vmaf_target(out_dir, df_total, 'id')


    # save as pickle df_total
    with open('df_total.pickle', 'wb') as f:
        pickle.dump(df_total, f, pickle.HIGHEST_PROTOCOL)

    #=== build bitrate ladder ===#

