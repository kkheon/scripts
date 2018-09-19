
import os
import glob
import pandas as pd
import re

from stat_hevc import stat_hevc
from stat_psnr import stat_psnr
from stat_pred_psnr import stat_pred_psnr


if __name__ == '__main__':

    list_dir = [
        #'/home/kkheon/VCNN-Tensorflow/data_vsr/val'
    ]

    ## temp-down
    #filename = '/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_down_hm/QP32/result_mf_vcnn_down_scene_53.txt'
    #stat_scene_53 = stat_hevc(filename)

    #poc_53 = stat_scene_53.get_frame_table()
    #print(poc_53)
    #summary_53 = stat_scene_53.get_summary_table()
    #print(summary_53)

    ## temp-up
    #filename = '/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_up_frm5_g3/QP32/psnr_rec_mf_vcnn_down_scene_53.txt'
    #stat_up = stat_psnr(filename)
    #df_up = stat_up.get_frame_table()
    #print(df_up)


    # temp total
    target_poc = 3

    list_dir = [
       # '/hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v1_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v2_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v3_qp32'

        #'/home/kkheon/VCNN-Tensorflow/data_vsr/val'

        #'/home/kkheon/MF-VCNN-Tensorflow/data_vsr/val/v2_qp32'

        '/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed'
        #'/hdd2T/kkheon/data_vsr_bak/val/v7_qp32'
    ]

    list_qp = [
        'QP32'
       #,'QP37'
       #,'QP42'
       #,'QP47'
    ]

    #
    run_multi_frame = True

    for each_qp in list_qp:
        if run_multi_frame:
            path_prefix = ""
            down_dir_name = 'result_mf_vcnn_down*_hm/QP*'
            down_filename = "result_mf_vcnn_down_*.txt"

            #down_dir_name = 'result_mf_vcnn_down*_hm'
            #down_filename = each_qp + "/result_mf_vcnn_down_*.txt"

            # multi-frame-run
            # path : /hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_up_frm5_g3/QP32
            # file : psnr_rec_mf_vcnn_down_scene_53.txt
            up_dir_name = 'result_mf_vcnn_up_*/QP*'
            up_filename = "psnr_*" + ".txt"
            #up_dir_name = 'result_mf_vcnn_up_*'
            #up_filename = "vsr_pred_psnr_" + each_qp + ".txt"
        else:
            path_prefix = 'result_' + each_qp

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

        for each_dir in list_dir:
            #path = os.path.join(each_dir, 'result_' + each_qp)
            #path = each_dir
            path = os.path.join(each_dir, path_prefix)

            # ========== down-sampled bitrate ========== #
            target_path = os.path.join(path, down_dir_name)
            list_sub_dir = sorted(glob.glob(target_path))

            df_down = pd.DataFrame()
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, down_filename)))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

            # ========== up-sampled PSNR ========== #
            target_path = os.path.join(path, up_dir_name)
            list_sub_dir = sorted(glob.glob(target_path))

            df_up = pd.DataFrame()
            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, up_filename)))
                for each_txt in list_txt:
                    each_stat = stat_psnr(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_up = df_up.append(each_frame_table)

            ### merge
            df_merged = pd.merge(df_down, df_up, on='id', how='outer')
            #print(df_merged)

            df_merged = df_merged[['name_x', 'frm_x', 'qp_x', 'bitrate', 'psnr_y_up', 'ssim_up']]
            df_merged.columns = ['name', 'frm', 'qp', 'bitrate', 'psnr_y_up', 'ssim_up']
            df_merged = df_merged.sort_values(['name', 'frm', 'qp'])

            #df_merged.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_merged.txt', header=None, index=None, sep=' ')

            filename_merged = os.path.join(each_dir, 'df_raw.txt')
            #df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
            df_merged.to_csv(filename_merged, index=None, sep=' ')

            # type change
            df_merged[['psnr_y_up', 'ssim_up']] = df_merged[['psnr_y_up', 'ssim_up']].astype(float)

            # frame-level average
            df_frame_level = df_merged.groupby(['frm', 'qp']).mean()

            # to_file
            filename_merged = os.path.join(each_dir, 'df_avg_frame.txt')
            df_frame_level.to_csv(filename_merged, sep=' ')


            # video-level average
            df_video_level = df_merged.groupby(['name', 'qp']).mean()

            # to_file
            filename_merged = os.path.join(each_dir, 'df_avg_video.txt')
            df_video_level.to_csv(filename_merged, sep=' ')




