
import os
import glob
import re
import pandas as pd
import seaborn as sns

from matplotlib import pyplot
import statsmodels.api as sm  # import statsmodels


from yuv_diff import yuv_diff

def draw_scatter_plot(output_path, in_filename_frm, df_sse_merged ):
    # scatter plot
    filename_sse = os.path.join(output_path, 'df_sse_' + in_filename_frm)
    ax = df_sse_merged.plot(kind='scatter', x='sse_LR', y='sse_HR', color='Red')
    fig = ax.get_figure()
    fig.savefig(filename_sse + '.png')


def get_linear_model(output_path, in_filename, df_sse):
    #== linear regression ===#
    X = df_sse["sse_LR"]  ## X usually means our input variables (or independent variables)
    y = df_sse["sse_HR"]  ## Y usually means our output/dependent variable
    X = sm.add_constant(X)  ## let's add an intercept (beta_0) to our model

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()  ## sm.OLS(output, input)
    predictions = model.predict(X)

    # Print out the statistics
    filename_model = os.path.join(output_path, 'model_summary_' + in_filename + '.txt')
    with open(filename_model, 'w') as f:
        f.write(model.summary().as_text())

if __name__ == '__main__':
    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff"

    w = 3840
    h = 2160
    #block_size = 64
    block_size = 128
    scale = 1  # todo : scale what for ?

    n_frame = 5

    frame_label = 2
    #frame_input = 0
    frame_input = 2

    ## path setting
    #list_label = sorted(glob.glob(os.path.join(label_path, "*.yuv")))
    #list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    # paths
    path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/train'
    path_input = '/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/train/result_mf_vcnn_up_4'
    path_label_low = '/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/train/result_mf_vcnn_down_3'
    path_input_low = path_label_low + '_hm'

    ## for test
    #list_label = ['/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/train/scene_0.yuv']
    #list_input = ['/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/train/result_mf_vcnn_up_4/QP32/img_mf_vcnn_up_rec_mf_vcnn_down_scene_0.yuv']
    #list_label_low = ['/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/train/result_mf_vcnn_down_3/mf_vcnn_down_scene_0.yuv']
    #list_input_low = ['/home/kkheon/VSR-Tensorflow-exp-4-3/data_vsr/train/result_mf_vcnn_down_3_hm/QP32/rec_mf_vcnn_down_scene_0.yuv']

    # consider QP
    path_input = path_input + '/QP32'
    path_input_low = path_input_low + '/QP32'

    # search .yuv
    list_label = sorted(glob.glob(os.path.join(path_label, "*.yuv")))
    list_input = sorted(glob.glob(os.path.join(path_input, "*.yuv")))
    list_label_low = sorted(glob.glob(os.path.join(path_label_low, "*.yuv")))
    list_input_low = sorted(glob.glob(os.path.join(path_input_low, "*.yuv")))

    # check of  out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initial df_sse_total
    df_sse_total = pd.DataFrame()

    # save each image's PSNR result as file.
    for idx, each_image in enumerate(list_label):
        # parsing filename
        _, in_filename = each_image.rsplit('/', 1)
        in_filename, _ = in_filename.rsplit('.', 1)

        # high resolution
        each_label = list_label[idx]
        each_input = list_input[idx]

        # low resolution
        each_label_low = list_label_low[idx]
        each_input_low = list_input_low[idx]
        w_low = int(w/2)
        h_low = int(h/2)
        block_size_low = int(block_size / 2)

        # initial df_sse_image
        df_sse_image = pd.DataFrame()

        for each_frame in range(n_frame):
            in_filename_frm = in_filename + '_frm%02d' % (each_frame)

            frame_label = each_frame
            frame_input = each_frame

            # high resolution
            df_psnr, df_sse = yuv_diff(each_label, frame_label, each_input, frame_input, w, h, block_size, scale)

            # low resolution
            df_psnr_low, df_sse_low = yuv_diff(each_label_low, frame_label, each_input_low, frame_input, w_low, h_low, block_size_low, scale)

            ## df to file
            #filename_psnr = os.path.join(output_path, 'df_psnr_' + in_filename_frm)
            #df_psnr_a.to_csv(filename_psnr + '.txt', sep=' ')

            ## df to image
            #pyplot.figure(figsize=(20, 10))
            #sns_plot = sns.heatmap(df_psnr_a, annot=True, vmin=20, vmax=80)
            #fig = sns_plot.get_figure()
            #fig.savefig(filename_psnr + '.png')

            #=== matching LR with HR ===#
            # 2D data frame to 1D
            df_sse_1d = pd.DataFrame(df_sse.values.flatten())
            df_sse_low_1d = pd.DataFrame(df_sse_low.values.flatten())

            # combine two df
            df_sse_merged = pd.concat([df_sse_low_1d, df_sse_1d], axis=1)
            df_sse_merged.columns = ['sse_LR', 'sse_HR']

            # scatter plot
            draw_scatter_plot(output_path, in_filename_frm, df_sse_merged)

            # df_sse_merged to df_sse_video
            df_sse_image = df_sse_image.append(df_sse_merged)

        # df_sse_image to df_sse_total
        df_sse_total = df_sse_total.append(df_sse_image)

        # scatter plot
        draw_scatter_plot(output_path, in_filename, df_sse_image)

        #== linear regression ===#
        get_linear_model(output_path, in_filename, df_sse_image)


    # scatter plot
    draw_scatter_plot(output_path, "total", df_sse_total)

    #== linear regression ===#
    get_linear_model(output_path, "total", df_sse_total)
