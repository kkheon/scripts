
import os
import glob
import pandas as pd
import re

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import json

from stat_sse import stat_sse
from stat_vmaf import stat_vmaf


if __name__ == '__main__':

    list_dir_vmaf_org = './SJTU_4K_test_vmaf'

    list_dir_vmaf = './dataset_replace_vmaf'
    # result_Traffic_Flow_frm00_0576x0256_qp27.json

    list_dir_sse = './dataset_replace'
    # Traffic_Flow_frm00_0576x0256_qp27_sse.txt

    list_qp = [
    #    'QP22',
    #    'QP27',
        'QP32',
    #    'QP37',
    #    'QP42',
    #    'QP47',
    ]

    #===== save path =====#
    out_dir = './result_vmaf_sse'

    df_vmaf_org = pd.DataFrame()
    df_vmaf = pd.DataFrame()
    df_sse = pd.DataFrame()

    # ==== VMAF_ORG
    for each_qp in list_qp:
        target_path = os.path.join(list_dir_vmaf_org, each_qp)
        list_txt = sorted(glob.glob(os.path.join(target_path, "*.json")))
        for each_txt in list_txt:
            each_stat = stat_vmaf(each_txt)

            each_frame_table = each_stat.get_frame_table()
            df_vmaf_org = df_vmaf_org.append(each_frame_table)

    # ========== VMAF
    list_txt = sorted(glob.glob(os.path.join(list_dir_vmaf, "*.json")))
    for each_txt in list_txt:
        each_stat = stat_vmaf(each_txt, replaced=True)

        each_frame_table = each_stat.get_frame_table()
        df_vmaf = df_vmaf.append(each_frame_table)

    merge_on = ['name', 'frm']
    df_merged = pd.merge(df_vmaf_org, df_vmaf, on=merge_on, how='outer', suffixes=['_org', ''])
    df_merged['VMAF_diff'] = df_merged['VMAF'] - df_merged['VMAF_org']

    # drop the row which has null value.
    df_merged = df_merged.dropna(how='any', axis=0)

    filename_merged = os.path.join(out_dir, 'df_vmaf_diff.txt')
    #df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
    df_merged.to_csv(filename_merged, index=None, sep=' ')

    # ========== SSE ========== #
    list_txt = sorted(glob.glob(os.path.join(list_dir_sse, "*_sse.txt")))
    for each_txt in list_txt:
        each_stat = stat_sse(each_txt)

        each_frame_table = each_stat.get_frame_table()
        df_sse = df_sse.append(each_frame_table)

    #========== Merge ==========#
    merge_on = ['name', 'qp', 'frm_replaced', 'x', 'y', 'frm']

    # merge with VMAF
    df_merged = pd.merge(df_sse, df_merged, on=merge_on, how='outer', suffixes=['', '_vmaf'])

    df_merged = df_merged.dropna(how='any', axis=0)

    filename_merged = os.path.join(out_dir, 'df_raw.txt')
    #df_merged.to_csv(filename_merged, header=None, index=None, sep=' ')
    df_merged.to_csv(filename_merged, index=None, sep=' ')

    df_reg = pd.DataFrame()

    row_size, col_size = df_merged.shape
    print(df_merged.shape)

    for i in range(0, row_size, 4):
        #print(df_merged.loc[i:i+3])
        each_df = df_merged.loc[i:i+3].copy()

        # calculate regression with diff qp
        X = each_df['sse'].values.reshape(-1, 1)
        Y = each_df['VMAF_diff'].values.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression

        r_squared = linear_regressor.score(X, Y)
        print(r_squared)

        # save r_squared to dataframe
        each_df['r_squared'] = r_squared
        each_df['coef'] = linear_regressor.coef_[0][0]
        each_df['intercept'] = linear_regressor.intercept_[0]

        df_reg = df_reg.append(each_df)

        # poly fit


    # save linear regression
    filename = os.path.join(out_dir, 'df_raw_linear_regression.txt')
    df_reg.to_csv(filename, index=None, sep=' ')

