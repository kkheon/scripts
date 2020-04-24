
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

        list_resolution = each_df_video.resolution.unique()
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

