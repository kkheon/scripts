
import numpy as np
import pandas as pd
from read_yuv import read_yuv420
from psnr import psnr

from skimage.measure import compare_ssim as ssim

def yuv_diff(filename_label, frame_label, filename_input, frame_input, w, h, block_size, scale):

    print("========== FRM : label=%03d, input=%03d ==========\n" % (frame_label, frame_input))

    # read yuv : label
    array_label_y, array_label_cbcr = read_yuv420(filename_label, w, h, 1, start_frame=frame_label)
    label_y = array_label_y.squeeze()

    # read yuv : input
    array_input_y, array_input_cbcr = read_yuv420(filename_input, w, h, 1, start_frame=frame_input)
    input_y = array_input_y.squeeze()

    # calculate MSE in pixel domain
    # make a copy of pixel domain
    label_y_pel = label_y.copy()
    input_y_pel = input_y.copy()

    # normalization
    label_y = label_y / 255.
    input_y = input_y / 255.

    # count for each frame
    block_count = 0

    # loop based on block
    list_psnr_frame = []
    list_sse_frame = []
    for y in range(0, h, block_size):
        list_psnr_row = []
        list_sse_row = []

        for x in range(0, w, block_size):

            # pick block from input
            sub_label_y = label_y[y:y+block_size, x:x+block_size]
            sub_input_y = input_y[y:y+block_size, x:x+block_size]

            # PSNR calculation
            each_psnr = psnr(sub_input_y, sub_label_y, scale)
            #each_ssim = ssim(sub_input_y, sub_label_y, data_range=256)

            list_psnr_row.append(float("{0:.4f}".format(each_psnr)))

            # TODO : later
            # yuv diff
            # save diff as image
            # combine yuv

            # calculate SSE in pixel domain
            sub_label_y_pel = label_y_pel[y:y+block_size, x:x+block_size]
            sub_input_y_pel = input_y_pel[y:y+block_size, x:x+block_size]

            # SSE
            each_sse = np.sum((sub_label_y_pel - sub_input_y_pel) ** 2)
            list_sse_row.append(float("{0:.4f}".format(each_sse)))

            block_count += 1

        list_psnr_frame.append(list_psnr_row)
        list_sse_frame.append(list_sse_row)

    # outside of the loop

    # stat the block-level PSNR
    # list_psnr to df
    df_psnr = pd.DataFrame(list_psnr_frame)
    df_sse = pd.DataFrame(list_sse_frame)


    return df_psnr, df_sse
