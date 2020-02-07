
import os
import glob
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

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
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

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

        model_1 = LinearRegression()  # create object for the class
        model_1.fit(X, Y)  # perform linear regression
        Y_pred_1 = model_1.predict(X)

        #r2_score = model_1.score(X, Y)
        #print(r2_score)
        #each_df['r2_score'] = float(r2_score)
        model_1_r2 = r2_score(Y, Y_pred_1)

        # save r2_score to dataframe
        each_df['r2_score'] = model_1_r2
        each_df['coef'] = model_1.coef_[0][0]
        each_df['intercept'] = model_1.intercept_[0]

        # poly fit
        poly = PolynomialFeatures(degree = 2) 
        X_poly = poly.fit_transform(X) 
          
        model_2 = LinearRegression()
        model_2.fit(X_poly, Y)
        Y_pred_2 = model_2.predict(X_poly)
        model_2_r2 = r2_score(Y, Y_pred_2)

        # save r2_score to dataframe
        each_df['degree2_r2_score'] = model_2_r2 
        r2_diff = model_2_r2 - model_1_r2
        each_df['r2_diff'] = r2_diff 

        df_reg = df_reg.append(each_df)

        # if r2_score diff is large, plot
        if r2_diff > 0.2: 
          plt.figure(figsize=(10,8));

          # plot results
          #plt.scatter(X, Y, label='Training points', color='lightgray') 
          plt.scatter(X, Y, label='Training points', color='black') 
          #plt.plot(X, Y_pred_1, label='Linear (d=1), $R^2=%.2f$' % model_1_r2, color='blue', lw=2, linestyle=':')
          #plt.plot(X, Y_pred_2, label='Quadratic (d=2), $R^2=%.2f$' % model_2_r2, color='red', lw=2, linestyle='-')

          # fit features
          X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
          Y_fit_pred_1 = model_1.predict(X_fit)
          Y_fit_pred_2 = model_2.predict(poly.fit_transform(X_fit))

          plt.plot(X_fit, Y_fit_pred_1, label='Linear (d=1), $R^2=%.2f$' % model_1_r2, color='blue', lw=2, linestyle=':')
          plt.plot(X_fit, Y_fit_pred_2, label='Quadratic (d=2), $R^2=%.2f$' % model_2_r2, color='red', lw=2, linestyle='-')
          
          plt.xlabel('SSE diff')
          plt.ylabel('VMAF diff')
          #plt.legend(loc='upper right')
          plt.legend(loc='lower right')
          
          plt.tight_layout()

          out_dir_fig = os.path.join(out_dir, 'plot')
          if not os.path.exists(out_dir_fig):
              os.makedirs(out_dir_fig)

          out_filename = each_df['name'].values[0] + '_QP' + str(each_df['qp'].values[0]) + '_frm' + str(each_df['frm'].values[0]) + '_' + str(each_df['x'].values[0]) + 'x' + str(each_df['y'].values[0]) + '.png'
          out_file = os.path.join(out_dir_fig, out_filename)
          plt.savefig(out_file, dpi=300)


    # sort
    df_reg.sort_values(by=['r2_score'], inplace=True)

    # save linear regression
    filename = os.path.join(out_dir, 'df_raw_linear_regression.txt')
    df_reg.to_csv(filename, index=None, sep=' ')

