
import os
import glob
import pandas as pd
import re

from stat_hevc import stat_hevc
from stat_psnr_summary import stat_psnr_summary as stat_psnr


if __name__ == '__main__':

    list_dir = [
        '/data/kkheon/dataset/SJTU_4K_test/label_hm',
        '/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_1080_hm',
        '/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_720_hm',
        #'/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_544_hm',
    ]

    list_dir_up = [
        '',
        '/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_1080',
        '/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_720',
        #'/data/kkheon/data_vsr_bak/test_SJTU/lanczos_2160_to_544',
    ]
    list_qp = [
        'QP32'
       ,'QP37'
       ,'QP42'
       ,'QP47'
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

    for i, each_dir in enumerate(list_dir):
        df_down = pd.DataFrame()
        df_up = pd.DataFrame()

        path = each_dir
        each_dir_up = list_dir_up[i]

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
                    each_stat = stat_hevc(each_txt, 'name_qp_frm')

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

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
        out_dir = each_dir_up
        #out_dir = each_dir

        #===== save df_down =====#
        ## type change
        #df_down[['psnr_y_up', 'ssim_up']] = df_down[['psnr_y_up', 'ssim_up']].astype(float)
        df_down[['psnr_y']] = df_down[['psnr_y']].astype(float)

        ## to_file
        filename_down = os.path.join(out_dir, 'df_down.txt')
        df_down.to_csv(filename_down, sep=' ')

        # frame-level average
        #df_frame_level = df_down.groupby(['loop', 'frm', 'qp']).mean()
        df_frame_level = df_down.groupby(['frm', 'qp']).mean()

        # to_file
        filename_down = os.path.join(out_dir, 'df_down_frame_avg.txt')
        df_frame_level.to_csv(filename_down, sep=' ')

        # check empty of df_up
        if df_up.empty == True:
            continue

        # merge
        df_merged = pd.merge(df_down, df_up, on='id', how='outer')
        #print(df_merged)

        df_merged = df_merged[['loop', 'name_x', 'frm_x', 'qp_x', 'bitrate', 'psnr_y_up_bicubic', 'ssim_up_bicubic', 'psnr_y_up', 'ssim_up', 'epoch']]
        df_merged.columns = ['loop', 'name', 'frm', 'qp', 'bitrate', 'psnr_y_up_bicubic', 'ssim_up_bicubic', 'psnr_y_up', 'ssim_up', 'epoch']
        df_merged = df_merged.sort_values(['loop', 'name', 'epoch', 'frm', 'qp'])

        # drop the row which has null value.
        #df_merged = df_merged.dropna(how='any', axis=0)

        # drop duplicate rows
        df_merged = df_merged.drop_duplicates()

        #df_merged.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_merged.txt', header=None, index=None, sep=' ')

        filename_merged = os.path.join(out_dir, 'df_raw.txt')
        #df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
        df_merged.to_csv(filename_merged, index=None, sep=' ')

        # type change
        df_merged[['psnr_y_up', 'ssim_up']] = df_merged[['psnr_y_up', 'ssim_up']].astype(float)

        # frame-level average
        df_frame_level = df_merged.groupby(['loop', 'epoch', 'frm', 'qp']).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_frame_avg.txt')
        df_frame_level.to_csv(filename_merged, sep=' ')


        # video-level average
        df_video_level = df_merged.groupby(['loop', 'epoch', 'name', 'qp']).mean()

        # to_file
        filename_merged = os.path.join(out_dir, 'df_video_avg.txt')
        df_video_level.to_csv(filename_merged, sep=' ')




