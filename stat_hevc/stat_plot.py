
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_step_vmaf(path, df):

    w = 16
    h = 12
    d = 100
    plt.figure(figsize=(w, h), dpi=d)

    list_video_name = df.name.unique()

    # for each video
    for each_video_name in list_video_name:
        each_df_video = df[df['name'] == each_video_name]

        #list_resolution = each_df_video.resolution.unique()
        list_resolution = each_df_video['resolution'].unique()
        # for each resolution
        for each_resolution in list_resolution:
            each_df_resolution = each_df_video[each_df_video['resolution'] == each_resolution]

            # X : bitrate
            x = each_df_resolution['bitrate']
            # Y : VMAF
            y = each_df_resolution['VMAF_filter']

            plt.step(x, y, label=str(each_resolution) + '(lanczos)', where='post')
            #plt.plot(x, y, 'o', alpha=0.5, label=None)

            if 'VMAF' in each_df_resolution.columns:
                if each_df_resolution['VMAF'].isnull().values.any().any() == False:
                    y = each_df_resolution['VMAF']
                    plt.step(x, y, label=str(each_resolution) + '(VDSR)', where='post')
                    #plt.plot(x, y, 'o', alpha=0.5, label=None)

        plt.legend(title='resolution')
        # save as
        out_filename = os.path.join(path, 'plot_' + each_video_name + '.png')
        plt.savefig(out_filename)

        # axis adjustment
        x1, x2, y1, y2 = plt.axis()
        #print(plt.axis())

        # y :
        each_df_resolution = each_df_video[each_df_video['resolution'] == 2160]
        y_lim_min = each_df_resolution['VMAF_filter'].min()

        for each_resolution in list_resolution:
            each_df_resolution = each_df_video[each_df_video['resolution'] == each_resolution]

            x_lim_min = 0
            x_lim_max = each_df_resolution['bits'].max()
            #plt.axis((x_lim_min, x_lim_max, y1, y2))
            plt.axis((x_lim_min, x_lim_max, y_lim_min, y2))
            # save as
            out_filename = os.path.join(path, 'plot_' + each_video_name + '_period_' + str(each_resolution) + '.png')
            plt.savefig(out_filename)


        plt.clf()

def plot_step_vmaf_target(path, df, target='resolution'):

    w = 16
    h = 12
    d = 100

    plt.figure(figsize=(w, h), dpi=d)

    list_video_name = df['name'].unique()

    # for each video
    for each_video_name in list_video_name:
        each_df_video = df[df['name'] == each_video_name]


        each_df_vdsr = each_df_video[each_df_video['id'].isin(['2160','VDSR(1080)','VDSR(720)', 'VDSR(544)'])]

        # wanna draw original 'VMAF_filter'
        list_resolution = each_df_vdsr['resolution'].unique()
        for each_resolution in list_resolution:
            each_df_resolution = each_df_vdsr[each_df_vdsr['resolution'] == each_resolution]

            # X : bitrate
            x = each_df_resolution['bitrate']
            # Y : VMAF
            y = each_df_resolution['VMAF_filter']

            plt.step(x, y, 'o--', label=str(each_resolution) + '(lanczos)', where='post')
            #plt.plot(x, y, 'o', alpha=0.5, label=None)

        # wanna draw 'VMAF' (VDSR)
        list_id = each_df_video[target].unique()
        # for each resolution
        for each_id in list_id:
            each_df = each_df_video[each_df_video[target] == each_id]

            # X : bitrate
            x = each_df['bitrate']

            if 'VMAF' in each_df.columns:
                if each_df['VMAF'].isnull().values.any().any() == False:
                    y = each_df['VMAF']
                    plt.step(x, y, 'x', label=each_id, where='post')
                    #plt.plot(x, y, 'o', alpha=0.5, label=None)

        plt.legend(title=target)
        # save as
        out_filename = os.path.join(path, 'plot_' + each_video_name + '.png')
        plt.savefig(out_filename)

        # axis adjustment
        x1, x2, y1, y2 = plt.axis()
        #print(plt.axis())

        # y :
        each_df = each_df_video[each_df_video['resolution'] == 2160]
        y_lim_min = each_df['VMAF_filter'].min()

        list_resolution = each_df_video['resolution'].unique()
        for each_resolution in list_resolution:
            each_df = each_df_video[each_df_video['resolution'] == each_resolution]

            x_lim_min = 0
            x_lim_max = each_df['bitrate'].max()
            #plt.axis((x_lim_min, x_lim_max, y1, y2))
            plt.axis((x_lim_min, x_lim_max, y_lim_min, y2))
            # save as
            out_filename = os.path.join(path, 'plot_' + each_video_name + '_period_' + str(each_resolution) + '.png')
            plt.savefig(out_filename)

        plt.clf()

