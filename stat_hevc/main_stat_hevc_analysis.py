

import os
import glob
import pandas as pd
import re

from stat_hevc import stat_hevc
from stat_psnr import stat_psnr


if __name__ == '__main__':


    list_dir = [
        '/home/kkheon/VSR-Tensorflow-exp-8/data_vsr/val'
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
        down_dir_name = 'result_mf_vcnn_down*_hm'
        #down_filename = "result_mf_vcnn_down_*.txt"
        down_filename = "result_*.txt"

        #down_dir_name = 'result_mf_vcnn_down*_hm'
        #down_filename = each_qp + "/result_mf_vcnn_down_*.txt"

        # multi-frame-run
        # path : /hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_up_frm5_g3/QP32
        # file : psnr_rec_mf_vcnn_down_scene_53.txt
        up_dir_name = 'result_mf_vcnn_up_*'
        up_filename = "psnr_*" + ".txt"
        #up_dir_name = 'result_mf_vcnn_up_*'
        #up_filename = "vsr_pred_psnr_" + each_qp + ".txt"
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

    for each_dir in list_dir:
        df_stat = pd.DataFrame()

        for each_qp in list_qp:
            path = os.path.join(each_dir, 'result_' + each_qp)

            # ========== down-sampled bitrate ========== #
            target_path = os.path.join(path, down_dir_name, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, down_filename)))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_stat = df_stat.append(each_frame_table)


        filename = os.path.join(each_dir, 'df_raw.txt')
        df_stat.to_csv(filename, index=None, sep=' ')

        # type change
        df_stat[['psnr_y_up', 'ssim_up']] = df_stat[['psnr_y_up', 'ssim_up']].astype(float)

        # frame-level average
        df_frame_level = df_stat.groupby(['loop', 'frm', 'qp']).mean()

        # to_file
        filename = os.path.join(each_dir, 'df_frame_avg.txt')
        df_frame_level.to_csv(filename, sep=' ')


        # video-level average
        df_video_level = df_stat.groupby(['loop', 'name', 'qp']).mean()

        # to_file
        filename = os.path.join(each_dir, 'df_video_avg.txt')
        df_video_level.to_csv(filename, sep=' ')




