
import os
import glob
import pandas as pd
from stat_hevc import stat_hevc
from stat_psnr import stat_psnr


if __name__ == '__main__':

    list_dir = [
        #'/home/kkheon/VCNN-Tensorflow/data_vsr/val'
    ]

    # temp-down
    filename = '/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_down_hm/QP32/result_mf_vcnn_down_scene_53.txt'
    stat_scene_53 = stat_hevc(filename)

    poc_53 = stat_scene_53.get_frame_table()
    print(poc_53)
    summary_53 = stat_scene_53.get_summary_table()
    print(summary_53)

    # temp-up
    filename = '/hdd2T/kkheon/result_mf_vcnn/v2_qp32_bugfixed/result_mf_vcnn_up_frm5_g3/QP32/psnr_rec_mf_vcnn_down_scene_53.txt'
    stat_up = stat_psnr(filename)
    df_up = stat_up.get_frame_table()
    print(df_up)


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
    ]

    list_qp = [
        'QP32'
       #,'QP37'
       #,'QP42'
       #,'QP47'
    ]

    for each_dir in list_dir:
        for each_qp in list_qp:
            #path = os.path.join(each_dir, 'result_' + each_qp)
            path = each_dir

            #target_dir = 'result_mf_vcnn_down_*_hm/'
            #target_dir = 'result_vcnn_down*_hm/QP*'
            target_dir = 'result_mf_vcnn_down*_hm/QP*'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            df_down = pd.DataFrame()
            for each_sub_dir in list_sub_dir:
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_mf_vcnn_down_*.txt")))
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_vcnn_down_*.txt")))
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "result_vcnn_down_*.txt")))
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "result_mf_vcnn_down_*.txt")))

                for each_txt in list_txt:
                    each_stat = stat_hevc(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_down = df_down.append(each_frame_table)

                    #target_poc_stat = each_stat[['poc' == target_poc]]


            #target_dir = 'result_mf_vcnn_up_*/'
            #target_dir = 'result_vcnn_up_*/QP*'
            target_dir = 'result_mf_vcnn_up_*/QP*'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            df_up = pd.DataFrame()
            for each_sub_dir in list_sub_dir:
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "vsr_pred_psnr_" + each_qp + ".txt")))
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "psnr_*" + ".txt")))

                for each_txt in list_txt:
                    each_stat = stat_psnr(each_txt)

                    each_frame_table = each_stat.get_frame_table()
                    df_up = df_up.append(each_frame_table)

            ### merge
            df_merged = pd.merge(df_down, df_up, on='id', how='outer')
            print(df_merged)



