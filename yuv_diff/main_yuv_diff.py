
import os
import glob
import re
import seaborn as sns
from matplotlib import pyplot

from yuv_diff import yuv_diff


if __name__ == '__main__':
    # settings
    label_path = "./data_vsr/val/label"
    input_path = "./data_vsr/val/label"
    output_path = "./result_diff"

    w = 3840
    h = 2160
    block_size = 64
    scale = 1

    frame_label = 2
    #frame_input = 0
    frame_input = 2


    ## path setting
    #list_label = sorted(glob.glob(os.path.join(label_path, "*.yuv")))
    #list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    # for test
    list_label = ['/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/train/scene_0.yuv']
    list_input = ['/home/kkheon/VSR-Tensorflow/data_vsr/train/result_mf_vcnn_up_4/QP32/img_mf_vcnn_up_rec_mf_vcnn_down_scene_0.yuv']

    # check of  out_dir existence
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save each image's PSNR result as file.
    for idx, each_image in enumerate(list_label):

        each_label = list_label[idx]
        each_input = list_input[idx]

        df_psnr, df_sse = yuv_diff(each_label, frame_label, each_input, frame_input, w, h, block_size, scale)

        # parsing filename
        _, in_filename = each_image.rsplit('/', 1)
        in_filename, _ = in_filename.rsplit('.', 1)

        # df to file
        filename_psnr = os.path.join(output_path, 'df_psnr_' + in_filename)
        df_psnr.to_csv(filename_psnr + '.txt', sep=' ')

        # df to image
        pyplot.figure(figsize=(20, 10))
        sns_plot = sns.heatmap(df_psnr, annot=True, vmin=20, vmax=80)
        fig = sns_plot.get_figure()
        fig.savefig(filename_psnr + '.png')


        # save diff as image

        # if possible, write psnr in the image.

