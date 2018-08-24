


import os
import re
import glob
import numpy as np
import pandas as pd

def get_poc_2(filename):

    f = open(filename, "r")

    # filename : /hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32/result_QP32/result_mf_vcnn_down_1_hm/QP32/result_mf_vcnn_down_Campfire_Party.txt
    version = re.findall('v[0-9]+', filename)
    qp = re.findall('result_QP[_A-Za-z0-9]+', filename)
    vcnn_down_iter = re.findall('vcnn_down_[_A-Za-z0-9]+', filename)

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

            #frame_rate = 25 # temp
            frame_rate = 60 # temp
            bitrate = bits * frame_rate / 1000

            #return bitrate
            list_numbers = re.findall('[.0-9]+', vcnn_down_iter[0])
            iter = str(int(list_numbers[0])+1)
            id = version[0] + '_' + qp[0] + '_' + iter + '_' + yuv_name
            return [filename, version[0], qp[0], vcnn_down_iter[0], yuv_name, bitrate, id]

    f.close()

    # there was no POC 2.
    raise ValueError("no POC 2 in " + filename)

    #return 0

def get_poc_summary(filename):

    f = open(filename, "r")

    # filename : /hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32/result_QP32/result_mf_vcnn_down_1_hm/QP32/result_mf_vcnn_down_Campfire_Party.txt
    version = re.findall('v[0-9]+', filename)
    qp = re.findall('result_QP[_A-Za-z0-9]+', filename)
    vcnn_down_iter = re.findall('vcnn_down_[_A-Za-z0-9]+', filename)

    _, txt_name = filename.rsplit('/', 1)
    _, txt_name = filename.rsplit('down_', 1)
    yuv_name, _ = txt_name.split('.', 1)

    data = f.readlines()

    list_data = []
    data_item = ""
    bits_acc = 0
    for each_line in data:
        if 'POC' in each_line:
            list_numbers = re.findall('[.0-9]+', each_line)

            poc = int(list_numbers[0])

            bits = int(list_numbers[4])
            psnr_y = float(list_numbers[5])
            psnr_u = float(list_numbers[6])
            psnr_v = float(list_numbers[7])

            #frame_rate = 25 # temp
            frame_rate = 60 # temp
            bitrate = bits * frame_rate / 1000

            bits_acc += bits
            bitrate_acc = bits_acc * frame_rate / 1000

            #return bitrate
            list_numbers = re.findall('[.0-9]+', vcnn_down_iter[0])
            iter = str(int(list_numbers[0])+1)
            id = version[0] + '_' + qp[0] + '_' + iter + '_' + yuv_name
            #list_data.append([filename, version[0], qp[0], vcnn_down_iter[0], yuv_name, bitrate, id, poc, bits_acc, bitrate_acc])
            data_item = [filename, version[0], qp[0], vcnn_down_iter[0], yuv_name, bitrate_acc, id, poc, bits_acc]

    f.close()

    ## there was no POC 2.
    #raise ValueError("no POC 2 in " + filename)

    return data_item

def get_up_psnr(filename):

    f = open(filename, "r")

    data = f.readlines()

    list_data = []

    version = re.findall('v[0-9]+', filename)
    qp = re.findall('result_QP[_A-Za-z0-9]+', filename)
    vcnn_up_iter = re.findall('vcnn_up_[_A-Za-z0-9]+', filename)

    for each_line in data:
        if 'label' in each_line:
            list_numbers = re.findall('[.0-9]+', each_line)
            psnr_up = float(list_numbers[-1])

            list_yuv_name = re.findall('[_A-Za-z0-9]+.yuv', each_line)
            yuv_name, _ = list_yuv_name[0].split('.', 1)

            list_numbers = re.findall('[.0-9]+', vcnn_up_iter[0])
            iter = list_numbers[0]
            id = version[0] + '_' + qp[0] + '_' + iter + '_' + yuv_name
            #id = version[0] + '_' + qp[0] + '_' + vcnn_up_iter[0] + '_' + yuv_name
            list_data.append([filename, version[0], qp[0], vcnn_up_iter[0], yuv_name, psnr_up, id])

    f.close()

    return list_data



if __name__ == '__main__':

    #base_dir = '/hdd2T/kkheon/data_vsr_bak/val_SJTU'
    base_dir = '/hdd2T/kkheon/data_vsr_bak/val/v2_type1_qp32_bugfixed'
    list_dir = [
       # '/hdd2T/kkheon/data_vsr_bak/val_SJTU/v2_qp32'

       # '/hdd2T/kkheon/data_vsr_bak/val_SJTU/v0_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v1_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v2_qp32'
       #,'/hdd2T/kkheon/data_vsr_bak/val_SJTU/v3_qp32'

        '/hdd2T/kkheon/data_vsr_bak/val/v2_type1_qp32_bugfixed'
    ]

    list_qp = [
        'QP27'
       ,'QP32'
       ,'QP37'
       ,'QP42'
       ,'QP47'
    ]

    list_table_bitrate = []
    list_table_bitrate_summary = []
    list_table = []
    for each_dir in list_dir:
        for each_qp in list_qp:
            path = os.path.join(each_dir, 'result_' + each_qp)

            target_dir = 'result_mf_vcnn_down_*_hm/'
            #target_dir = 'result_mf_vcnn_down_2_hm/'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_mf_vcnn_down_*.txt")))

                list_bitrate = []
                for each_txt in list_txt:
                    list_temp_bitrate = get_poc_2(each_txt)
                    list_bitrate.append(list_temp_bitrate[5])

                    # to table
                    list_table_bitrate.append(list_temp_bitrate)

                avg_bitrate = np.mean(list_bitrate)
                print("\t" + each_sub_dir + " : " + str(avg_bitrate))

                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, each_qp, "result_mf_vcnn_down_*.txt")))

                list_bitrate = []
                for each_txt in list_txt:
                    list_temp_bitrate = get_poc_summary(each_txt)

                    # to table
                    list_table_bitrate_summary.append(list_temp_bitrate)


            target_dir = 'result_mf_vcnn_up_*/'
            #target_dir = 'result_mf_vcnn_up_3/'
            target_path = os.path.join(path, target_dir)
            list_sub_dir = sorted(glob.glob(target_path))

            for each_sub_dir in list_sub_dir:
                list_txt = sorted(glob.glob(os.path.join(each_sub_dir, "vsr_pred_psnr_" + each_qp + ".txt")))

                list_bitrate = []
                for each_txt in list_txt:
                    psnr_data = get_up_psnr(each_txt)

                    list_table = list_table + psnr_data


    #
    df_psnr = pd.DataFrame(list_table)
    df_psnr = df_psnr.sort_values([4, 1, 3, 2])

    #print(df_psnr)
    #df_psnr.to_csv(r'df_psnr.txt', header=None, index=None, sep=' ', mode='a')
    filename_psnr = os.path.join(base_dir, 'df_psnr.txt')
    df_psnr.to_csv(filename_psnr, header=None, index=None, sep=' ', mode='a')

    #
    df_bitrate = pd.DataFrame(list_table_bitrate)
    df_bitrate = df_bitrate.sort_values([4, 1, 3, 2])

    print(df_bitrate)
    #df_bitrate.to_csv(r'df_bitrate.txt', header=None, index=None, sep=' ')
    filename_bit = os.path.join(base_dir, 'df_bitrate.txt')
    df_bitrate.to_csv(filename_bit, header=None, index=None, sep=' ')


    # merge
    df_merged = pd.merge(df_psnr[[5, 6]], df_bitrate[[5, 6]], on=6, how='outer')
    df_merged = df_merged[[6, '5_y', '5_x']]
    #df_merged.to_csv(r'df_merged.txt', header=None, index=None, sep=' ')
    filename_merge = os.path.join(base_dir, 'df_merged.txt')
    df_merged.to_csv(filename_merge, header=None, index=None, sep=' ')

    # merge bitrate_summary + psnr
    df_bitrate_summary = pd.DataFrame(list_table_bitrate_summary)
    df_bitrate_summary = df_bitrate_summary.sort_values([4, 1, 3, 2])

    #df_merged_summary = pd.merge(df_psnr[[5, 6]], df_bitrate[[5, 6]], df_bitrate_summary[[5, 6]], on=6, how='outer')
    df_merged_summary = pd.merge(df_bitrate_summary[[5, 6]], df_psnr[[5, 6]], on=6)
    #df_merged_summary = df_merged[[6, '5_y', '5_x']]
    #df_merged.to_csv(r'df_merged.txt', header=None, index=None, sep=' ')
    filename_merge = os.path.join(base_dir, 'df_merged_summary.txt')
    df_merged_summary.to_csv(filename_merge, header=None, index=None, sep=' ')

    # merge df_merged_summary + df_merged
    #df_merged_summary = pd.merge(df_psnr[[5, 6]], df_bitrate[[5, 6]], df_bitrate_summary[[5, 6]], on=6, how='outer')
    df_merged_summary = pd.merge(df_merged, df_merged_summary, on=6)
    #df_merged_summary = df_merged[[6, '5_y', '5_x']]
    #df_merged.to_csv(r'df_merged.txt', header=None, index=None, sep=' ')
    filename_merge = os.path.join(base_dir, 'df_merged_summary_2.txt')
    df_merged_summary.to_csv(filename_merge, header=None, index=None, sep=' ')

