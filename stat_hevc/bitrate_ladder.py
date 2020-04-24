import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_bitrate_ladder(df, out_dir):

    df_summary = pd.DataFrame()

    # Step 1 : decide target bitrate based on VMAF
    list_target_VMAF = range(20, 90, 10)

    list_video_name = df.name.unique()

    # for each video
    for each_video_name in list_video_name:
        each_df_video = df[df['name'] == each_video_name]

        # Step 2 : select best QP for original resolution.
        each_df_resolution = each_df_video[each_df_video['resolution'] == 2160]
        list_target_bitrate, df_target = find_bitrate_for_VMAF(each_df_resolution, list_target_VMAF)

        filename_merged = os.path.join(out_dir, each_video_name + '_ladder_0_org.txt')
        df_target.to_csv(filename_merged, sep=' ')

        # Step 3 : select best resolution and QP for each target.
        df_total = find_best_for_bitrate_loop(each_df_video, list_target_bitrate, 'VMAF_filter')

        filename_merged = os.path.join(out_dir, each_video_name + '_ladder_1_filter.txt')
        df_total.to_csv(filename_merged, sep=' ')

        # Step 4 : select best resolution and QP including VDSR result.
        df_total_vdsr = find_best_for_bitrate_loop(each_df_video, list_target_bitrate, 'VMAF')

        filename_merged = os.path.join(out_dir, each_video_name + '_ladder_2_VDSR.txt')
        df_total_vdsr.to_csv(filename_merged, sep=' ')

        # Step 5 : comparison
        # plot
        w = 16
        h = 12
        d = 100
        plt.figure(figsize=(w, h), dpi=d)

        plt.step(df_target['bitrate'], df_target['VMAF_filter'], label='2160p', where='post')
        plt.step(df_total['bitrate'], df_total['VMAF_filter'], label='lanczos', where='post')
        plt.step(df_total_vdsr['bitrate'], df_total_vdsr['VMAF'], label='VDSR', where='post')

        plt.legend(title='type')
        # save as
        out_filename = os.path.join(out_dir, 'plot_' + each_video_name + '.png')
        plt.savefig(out_filename)
        plt.clf()

        # basis : VMAF
        list_bitrate_0, df_0 = find_bitrate_for_VMAF(each_df_resolution, list_target_VMAF)
        list_bitrate_1, df_1 = find_bitrate_for_VMAF(each_df_video, list_target_VMAF)
        list_bitrate_2, df_2 = find_bitrate_for_VMAF(each_df_video, list_target_VMAF, index='VMAF')

        ## reorder
        #df_0.set_index(df_0['resolution'], inplace=True)
        #df_1.set_index(df_1['resolution'], inplace=True)
        #df_2.set_index(df_2['resolution'], inplace=True)
        df_0 = reindex_df_column(df_0, 'name', 0)
        df_1 = reindex_df_column(df_1, 'name', 0)
        df_2 = reindex_df_column(df_2, 'name', 0)
        df_0 = reindex_df_column(df_0, 'resolution', 1)
        df_1 = reindex_df_column(df_1, 'resolution', 1)
        df_2 = reindex_df_column(df_2, 'resolution', 1)

        filename_merged = os.path.join(out_dir, each_video_name + '_VMAF_ladder_0_org.txt')
        df_0.to_csv(filename_merged, sep=' ')
        filename_merged = os.path.join(out_dir, each_video_name + '_VMAF_ladder_1_filter.txt')
        df_1.to_csv(filename_merged, sep=' ')
        filename_merged = os.path.join(out_dir, each_video_name + '_VMAF_ladder_2_VDSR.txt')
        df_2.to_csv(filename_merged, sep=' ')

        # plot
        plt.figure(figsize=(w, h), dpi=d)

        plt.step(df_0['bitrate'], df_0['VMAF_filter'], label='2160p', where='post')
        plt.step(df_1['bitrate'], df_1['VMAF_filter'], label='lanczos', where='post')
        plt.step(df_2['bitrate'], df_2['VMAF'], label='VDSR', where='post')

        plt.legend(title='type')
        # save as
        out_filename = os.path.join(out_dir, 'plot_VMAF_ladder_' + each_video_name + '.png')
        plt.savefig(out_filename)

        # diff between tables.
        df_1['diff(bitrate_org)'] = df_1['bitrate'] - df_0['bitrate']
        df_2['diff(bitrate_org)'] = df_2['bitrate'] - df_0['bitrate']
        df_2['diff(bitrate_filter)'] = df_2['bitrate'] - df_1['bitrate']
        df_2['ratio(diff(bitrate_filter))'] = df_2['diff(bitrate_filter)'] / df_1['bitrate'] * 100

        df_2['resolution(filter)'] = df_1['resolution']

        df_1['diff(VMAF_org)'] = df_1['VMAF_filter'] - df_0['VMAF_filter']
        df_2['diff(VMAF_org)'] = df_2['VMAF'] - df_0['VMAF_filter']
        df_2['diff(VMAF_filter)'] = df_2['VMAF'] - df_1['VMAF_filter']

        # PSNR
        df_2['diff(PSNR)'] = df_2['psnr_y_up'] - df_1['psnr_y']
        df_2['diff(PSNR_filter)'] = df_2['psnr_y_up'] - df_1['psnr_y_up_bicubic']
        df_2_summary_video = df_2[['name', 'resolution',
                                   'resolution(filter)',
                                   'bitrate',
                                   #'diff(bitrate_org)',
                                   'diff(bitrate_filter)',
                                   'ratio(diff(bitrate_filter))',
                                   'VMAF_filter', 'VMAF',
                                   #'diff(VMAF_org)',
                                   'diff(VMAF_filter)',
                                   'psnr_y_up',
                                   #'diff(PSNR)',
                                   'diff(PSNR_filter)']]
        df_summary = df_summary.append(df_2_summary_video)

    # after loop
    ## formatting
    #pd.options.display.float_format = '{:.2f}'.format

    # save as
    filename = os.path.join(out_dir, 'summary.txt')
    df_summary.to_csv(filename, sep=' ', float_format='%.2f')

def find_bitrate_for_VMAF(df, list_VMAF, index='VMAF_filter'):

    df_target = pd.DataFrame()
    list_target_bitrate = []

    # find bitrate for each VMAF
    for each_VMAF in list_VMAF:
        df_cond = df.loc[df[index] >= each_VMAF]
        df_cond.reset_index(inplace=True)

        # find which has maximum VMAF
        df_cond_best = df_cond.iloc[df_cond['bitrate'].idxmin()]

        # add tag of target VMAF
        df_cond_best['target_VMAF'] = each_VMAF

        df_target = df_target.append(df_cond_best)

        list_target_bitrate.append(df_cond['bitrate'].min())

    # set target_VMAF as index
    df_target.set_index('target_VMAF', inplace=True)

    return list_target_bitrate, df_target

def find_best_for_bitrate(df, list_target_bitrate, index='VMAF_filter'):

    df_best = pd.DataFrame()
    for each_target_bitrate in list_target_bitrate:
        # which satisfy target bitrate
        df_cond = df.loc[df['bitrate'] <= each_target_bitrate]
        df_cond.reset_index(inplace=True)

        # find which has maximum VMAF
        df_cond_best_idx = df_cond[index].idxmax()
        if np.isnan(df_cond_best_idx) == False:
            df_cond_best = df_cond.iloc[df_cond_best_idx]
            df_best = df_best.append(df_cond_best)

    return df_best

def find_best_for_bitrate_loop(df, list_target_bitrate, index='VMAF_filter'):
    df_total = pd.DataFrame()
    df_best = find_best_for_bitrate(df, list_target_bitrate, index)
    df_total = df_total.append(df_best)

    return df_total

def find_best_for_bitrate_loop_each_resolution(df, list_target_bitrate, index='VMAF_filter'):
    df_total = pd.DataFrame()
    list_resolution = df.resolution.unique()
    # for each resolution
    for each_resolution in list_resolution:
        each_df_resolution = df[df['resolution'] == each_resolution]

        df_best_each_resol = find_best_for_bitrate(each_df_resolution, list_target_bitrate, index)
        df_total = df_total.append(df_best_each_resol)

    return df_total



def reindex_df_column(df, column_name, column_position):

    list = df.columns.tolist()  # list the columns in the df
    list.insert(column_position, list.pop(list.index(column_name)))  # Assign new position (i.e. 8) for "F"
    df = df.reindex(columns=list)  # Now move 'F' to ist new position

    return df

