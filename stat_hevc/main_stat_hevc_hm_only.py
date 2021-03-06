
import os
import glob
import pandas as pd
import re

from stat_hevc import stat_hevc
from stat_psnr import stat_psnr


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

        #'/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed'
        #'/hdd2T/kkheon/data_vsr_bak/val/v7_qp32'
        #'/home/kkheon/VSR-Tensorflow-exp5/data_vsr/val'

        #'/home/kkheon/VSR-Tensorflow/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp3/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp2-1/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp-4-1/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp-4-2/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp-7-3/data_vsr/val'
        #'/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val'

        #'/data/kkheon/test_HM16.9_cost_adaptation/result_val_mf_vcnn_down_4_hm_adap_dist_only'
        #'/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/val/result_QP32/result_mf_vcnn_down_4_hm'

        #'/home/kkheon/HM-16.9_CNN/bin/data_vsr/test/label_hm_arcnn'
        #'/data/kkheon/dataset/hevc_ctc/ClassB_1072p_hm'

        '/data/kkheon/dataset/SJTU_4K_test/label_hm',
        '/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_1080_hm',
        '/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_720_hm',
        '/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_544_hm',
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
        #down_dir_name = 'result_mf_vcnn_down*_hm'

        #down_filename = "result_mf_vcnn_down_*.txt"
        down_filename = "result_*.txt"

        #down_dir_name = 'result_mf_vcnn_down*_hm'
        #down_filename = each_qp + "/result_mf_vcnn_down_*.txt"

        # for hm only
        down_dir_name = '.'

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
        df_down = pd.DataFrame()
        df_up = pd.DataFrame()
        for each_qp in list_qp:
            #path = os.path.join(each_dir, 'result_' + each_qp)
            path = each_dir
            #path = os.path.join(each_dir, path_prefix)

            # ========== down-sampled bitrate ========== #
            target_path = os.path.join(path, down_dir_name, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))

            # add bicubic down-sampled hevc result.
            target_path = os.path.join(each_dir, 'result_hm', each_qp)
            list_sub_dir.append(target_path)

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, down_filename)))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

            # ========== up-sampled PSNR ========== #
            target_path = os.path.join(path, up_dir_name, each_qp)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, up_filename)))
                for each_txt in list_txt:
                    each_stat = stat_psnr(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_up = df_up.append(each_frame_table)

        #===== save df_down =====#
        ## type change
        #df_down[['psnr_y_up', 'ssim_up']] = df_down[['psnr_y_up', 'ssim_up']].astype(float)
        df_down[['psnr_y']] = df_down[['psnr_y']].astype(float)

        ## to_file
        filename_down = os.path.join(each_dir, 'df_down.txt')
        df_down.to_csv(filename_down, sep=' ')

        # frame-level average
        #df_frame_level = df_down.groupby(['loop', 'frm', 'qp']).mean()
        df_frame_level = df_down.groupby(['frm', 'qp']).mean()

        # to_file
        filename_down = os.path.join(each_dir, 'df_down_frame_avg.txt')
        df_frame_level.to_csv(filename_down, sep=' ')

        ## merge
        #df_merged = pd.merge(df_down, df_up, on='id', how='outer')
        ##print(df_merged)

        #df_merged = df_merged[['loop', 'name_x', 'frm_x', 'qp_x', 'bitrate', 'psnr_y_up', 'ssim_up']]
        #df_merged.columns = ['loop', 'name', 'frm', 'qp', 'bitrate', 'psnr_y_up', 'ssim_up']
        #df_merged = df_merged.sort_values(['loop', 'name', 'frm', 'qp'])

        ## drop the row which has null value.
        #df_merged = df_merged.dropna(how='any', axis=0)

        ##df_merged.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_merged.txt', header=None, index=None, sep=' ')

        #filename_merged = os.path.join(each_dir, 'df_raw.txt')
        ##df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
        #df_merged.to_csv(filename_merged, index=None, sep=' ')

        ## type change
        #df_merged[['psnr_y_up', 'ssim_up']] = df_merged[['psnr_y_up', 'ssim_up']].astype(float)

        ## frame-level average
        #df_frame_level = df_merged.groupby(['loop', 'frm', 'qp']).mean()

        ## to_file
        #filename_merged = os.path.join(each_dir, 'df_frame_avg.txt')
        #df_frame_level.to_csv(filename_merged, sep=' ')


        ## video-level average
        #df_video_level = df_merged.groupby(['loop', 'name', 'qp']).mean()

        ## to_file
        #filename_merged = os.path.join(each_dir, 'df_video_avg.txt')
        #df_video_level.to_csv(filename_merged, sep=' ')




