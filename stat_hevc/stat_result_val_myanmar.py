



import os
import re
import glob
import numpy as np
import pandas as pd

def get_poc_2(filename):

    f = open(filename, "r")

    # filename : /hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32/result_QP32/result_mf_vcnn_down_1_hm/QP32/result_mf_vcnn_down_Campfire_Party.txt
    #qp = re.findall('result_QP[_A-Za-z0-9]+', filename)
    qp = re.findall('QP[_A-Za-z0-9]+', filename)

    _, txt_name = filename.rsplit('/', 1)
    _, txt_name = filename.rsplit('down_', 1)
    yuv_name, _ = txt_name.split('.', 1)

    data = f.readlines()

    for each_line in data:
        if 'POC    2' in each_line:
            list_numbers = re.findall('[.0-9]+', each_line)

            bits = int(list_numbers[4])
            psnr_y = float(list_numbers[5])
            psnr_u = float(list_numbers[6])
            psnr_v = float(list_numbers[7])

            f.close()

            frame_rate = 60 # temp
            bitrate = bits * frame_rate / 1000

            #id = qp[0] + '_' + yuv_name
            id = qp[1] + '_' + yuv_name
            return [filename, qp[0], yuv_name, bitrate, id]

    f.close()

    # there was no POC 2.
    raise ValueError("no POC 2 in " + filename)

    #return 0

def get_up_psnr(filename):

    f = open(filename, "r")

    data = f.readlines()

    list_data = []

    #qp = re.findall('result_QP[_A-Za-z0-9]+', filename)
    qp = re.findall('QP[_A-Za-z0-9]+', filename)

    _, txt_name = filename.rsplit('/', 1)
    _, txt_name = filename.rsplit('down_', 1)
    yuv_name, _ = txt_name.split('.', 1)

    for each_line in data:
        if 'frm:  2' in each_line:
            list_numbers = re.findall('[.0-9]+', each_line)
            psnr_up = float(list_numbers[2])

            #id = qp[0] + '_' + yuv_name
            id = qp[1] + '_' + yuv_name
            #id = version[0] + '_' + qp[0] + '_' + vcnn_up_iter[0] + '_' + yuv_name
            list_data.append([filename, qp[0], yuv_name, psnr_up, id])

    f.close()

    return list_data



if __name__ == '__main__':

    list_dir = [
       # '/hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v1_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v2_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v3_qp32'

        '/home/kkheon/VCNN-Tensorflow/data_vsr/val'
    ]

    list_qp = [
        'QP32'
       #,'QP37'
       #,'QP42'
       #,'QP47'
    ]

    list_table_bitrate = []
    list_table = []
    for each_dir in list_dir:
        for each_qp in list_qp:
            path = os.path.join(each_dir, 'result_' + each_qp)

            #target_dir = 'result_mf_vcnn_down_*_hm/'
            target_dir = 'result_vcnn_down*_hm/QP*'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_mf_vcnn_down_*.txt")))
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_vcnn_down_*.txt")))
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "result_vcnn_down_*.txt")))

                list_bitrate = []
                for each_txt in list_txt:
                    list_temp_bitrate = get_poc_2(each_txt)
                    list_bitrate.append(list_temp_bitrate[3])

                    # to table
                    list_table_bitrate.append(list_temp_bitrate)

                avg_bitrate = np.mean(list_bitrate)
                print("\t" + each_sub_dir + " : " + str(avg_bitrate))

            #target_dir = 'result_mf_vcnn_up_*/'
            target_dir = 'result_vcnn_up_*/QP*'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                #list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "vsr_pred_psnr_" + each_qp + ".txt")))
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "psnr_*" + ".txt")))

                list_bitrate = []
                for each_txt in list_txt:
                    psnr_data = get_up_psnr(each_txt)

                    list_table = list_table + psnr_data


    #
    df_psnr = pd.DataFrame(list_table)
    df_psnr = df_psnr.sort_values([2, 1])

    #print(df_psnr)
    df_psnr.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_psnr.txt', header=None, index=None, sep=' ', mode='a')

    #
    df_bitrate = pd.DataFrame(list_table_bitrate)
    df_bitrate = df_bitrate.sort_values([2, 1])

    print(df_bitrate)
    df_bitrate.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_bitrate.txt', header=None, index=None, sep=' ')


    # merge
    df_merged = pd.merge(df_psnr[[3, 4]], df_bitrate[[3, 4]], on=4, how='outer')
    #df_merged = df_merged.sort_values(['4_x', '1_x', '3_x', '2_x'])
    #df_merged = df_merged.set_index(6)
    df_merged = df_merged[[4, '3_y', '3_x']]
    df_merged.to_csv(r'/home/kkheon/VCNN-Tensorflow/data_vsr/val/df_merged.txt', header=None, index=None, sep=' ')
