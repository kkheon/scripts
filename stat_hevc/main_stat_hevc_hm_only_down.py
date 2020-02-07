
import os
import glob
import pandas as pd
import re

import json

from stat_hevc import stat_hevc
from stat_psnr_summary import stat_psnr_summary as stat_psnr
from stat_vmaf import stat_vmaf

from stat_plot import plot_step_vmaf


if __name__ == '__main__':

    list_dir = [
        #'/data/kkheon/dataset/SJTU_4K_test/label_hm',
        #'/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_1080_hm',
        #'/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_720_hm',
        #'/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_544_hm',

        #'/data/kkheon/dataset/ultra_video_group/label_hm',
        #'/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1440_hm',
        #'/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080_hm',
        #'/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_720_hm',
        #'/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_544_hm',
        #'/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_360_hm',

        '/data/kkheon/dataset/ultra_video_group/label_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_720_hm_5frm',
        '/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_544_hm_5frm',
    ]

    list_dir_up = [
        '',
        '/data/kkheon/data_vsr_bak/test_ultra/lanczos_2160_to_1080',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_720',
        '/mnt/octopus/data_vsr_bak/test_ultra/lanczos_2160_to_544',

        #'/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_1080',
        #'/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_720',
        #'/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_544',
        '',
        '',
        '',
        '',
        '',
    ]
    list_qp = [
        #'QP22',
        #'QP27',
        #'QP32',
        #'QP37',
        #'QP42',
        #'QP47',

        'QP12',
        'QP13',
        'QP14',
        'QP15',
        'QP16',
        'QP17',
        'QP18',
        'QP19',

        'QP20',
        'QP21',
        'QP22',
        'QP23',
        'QP24',
        'QP25',
        'QP26',
        'QP27',
        'QP28',
        'QP29',

        'QP30',
        'QP31',
        'QP32',
        'QP33',
        'QP34',
        'QP35',
        'QP36',
        'QP37',
        'QP38',
        'QP39',

        'QP40',
        'QP41',
        'QP42',
        'QP43',
        'QP44',
        'QP45',
        'QP46',
        'QP47',
    ]

    # json file dir
    list_dir_filter_vmaf = [
        #'/home/kkheon/vmaf_test/SJTU_4K_test_vmaf',
        #'/home/kkheon/vmaf_test/result_vmaf_lanczos/lanczos_1080_to_2160_vmaf',
        #'/home/kkheon/vmaf_test/result_vmaf_lanczos/lanczos_720_to_2160_vmaf',
        #'/home/kkheon/vmaf_test/result_vmaf_lanczos/lanczos_544_to_2160_vmaf',

        #'/home/kkheon/scripts/vmaf_test/label_hm_vmaf',
        #'/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1440_hm_lanczos_1440_to_2160_vmaf',
        #'/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_hm_lanczos_1080_to_2160_vmaf',
        #'/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_hm_lanczos_720_to_2160_vmaf',
        #'/home/kkheon/scripts/vmaf_test/lanczos_2160_to_544_hm_lanczos_544_to_2160_vmaf',
        #'/home/kkheon/scripts/vmaf_test/lanczos_2160_to_360_hm_lanczos_360_to_2160_vmaf',

        '/home/kkheon/scripts/vmaf_test/label_hm_5frm_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_hm_5frm_lanczos_1080_to_2160_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_hm_5frm_lanczos_720_to_2160_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_544_hm_5frm_lanczos_544_to_2160_vmaf',
    ]
    list_dir_vdsr_vmaf = [
        '',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_1080_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_720_vmaf',
        '/home/kkheon/scripts/vmaf_test/lanczos_2160_to_544_vmaf',

        '',
        '',
        '',
        '',
        '',
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
        #path_prefix = 'result_' + each_qp

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
        each_dir_up = list_dir_up[i]

        each_dir_filter_vmaf = list_dir_filter_vmaf[i]
        each_dir_vmaf = list_dir_vdsr_vmaf[i]

        for each_qp in list_qp:

            # ========== down-sampled bitrate ========== #
            target_path = os.path.join(path, down_dir_name, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))

            # add bicubic down-sampled hevc result.
            #target_path = os.path.join(each_dir, 'result_hm', each_qp)
            target_path = os.path.join(each_dir, each_qp)
            list_sub_dir.append(target_path)

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, down_filename)))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

            #==== VMAF : filter
            target_path = os.path.join(each_dir_filter_vmaf, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "*.json")))
                for each_txt in list_txt:
                    each_stat = stat_vmaf(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_filter_vmaf = df_filter_vmaf.append(each_frame_table)

            #==== VMAF : VDSR
            target_path = os.path.join(each_dir_vmaf, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "*.json")))
                for each_txt in list_txt:
                    each_stat = stat_vmaf(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_vmaf = df_vmaf.append(each_frame_table)

        # ========== up-sampled PSNR ========== #
        #target_path = os.path.join(each_dir_up, up_dir_name, each_qp)
        target_path = os.path.join(each_dir_up, up_dir_name)
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
            df_total = df_total.append(df_video_level)

            continue

        # merge
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
        df_video_level = df_merged.groupby(['loop', 'epoch', 'name', 'qp'], as_index=False).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_video_avg.txt')
        df_video_level.to_csv(filename_merged, sep=' ')

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
    plot_step_vmaf(out_dir, df_total)


    #=== build bitrate ladder ===#
    # Step 1 : decide target bitrate based on VMAF
    # Setp 2 : select best resolution and QP for each target.


