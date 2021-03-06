
import numpy as np
import pandas as pd
from read_yuv import read_yuv420
from psnr import psnr

from skimage.measure import compare_ssim as ssim

def yuv_diff(filename_label, start_frame_label, filename_input, start_frame_input, w, h, block_size, scale):

    print("========== FRM : label=%03d, input=%03d ==========\n" % (start_frame_label, start_frame_input))

    # read yuv : label
    array_label_y, array_label_cbcr = read_yuv420(filename_label, w, h, 1, start_frame=start_frame_label)
    label_y = array_label_y.squeeze()

    # read yuv : input
    array_input_y, array_input_cbcr = read_yuv420(filename_input, w, h, 1, start_frame=start_frame_input)
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


    return df_psnr, df_sse, label_y_pel, input_y_pel

def yuv_diff_n_frame(filename_label, start_frame_label, filename_input, start_frame_input, w, h, block_size, scale, n_frame):

    print("========== FRM : label=%03d, input=%03d ==========\n" % (start_frame_label, start_frame_input))

    # read yuv : label
    array_label_y, array_label_cbcr = read_yuv420(filename_label, w, h, n_frame, start_frame=start_frame_label)
    label_y = array_label_y.squeeze(axis=3)

    # read yuv : input
    array_input_y, array_input_cbcr = read_yuv420(filename_input, w, h, n_frame, start_frame=start_frame_input)
    input_y = array_input_y.squeeze(axis=3)

    # calculate MSE in pixel domain
    # make a copy of pixel domain
    label_y_pel = label_y.copy()
    input_y_pel = input_y.copy()

    # normalization
    label_y = label_y / 255.
    input_y = input_y / 255.

    df_psnr = pd.DataFrame()
    df_sse = pd.DataFrame()

    for frame_index in range(0, n_frame):
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
                sub_label_y = label_y[frame_index, y:y+block_size, x:x+block_size]
                sub_input_y = input_y[frame_index, y:y+block_size, x:x+block_size]

                # PSNR calculation
                each_psnr = psnr(sub_input_y, sub_label_y, scale)
                #each_ssim = ssim(sub_input_y, sub_label_y, data_range=256)

                list_psnr_row.append(float("{0:.4f}".format(each_psnr)))

                # TODO : later
                # yuv diff
                # save diff as image
                # combine yuv

                # calculate SSE in pixel domain
                sub_label_y_pel = label_y_pel[frame_index, y:y+block_size, x:x+block_size]
                sub_input_y_pel = input_y_pel[frame_index, y:y+block_size, x:x+block_size]

                # SSE
                #each_sse = np.sum((sub_label_y_pel - sub_input_y_pel) ** 2)
                each_sse = np.sum((sub_label_y_pel - sub_input_y_pel) ** 2)

                #list_sse_row.append(float("{0:.4f}".format(each_sse)))
                list_sse_row.append(each_sse)

                block_count += 1

            list_psnr_frame.append(list_psnr_row)
            list_sse_frame.append(list_sse_row)

        # outside of the loop-block

        # stat the block-level PSNR
        # list_psnr to df
        df_psnr_frm = pd.DataFrame(list_psnr_frame)
        df_sse_frm = pd.DataFrame(list_sse_frame)

        # add frame column
        df_psnr_frm.insert(0, 'frame', start_frame_input+frame_index)
        df_sse_frm.insert(0, 'frame', start_frame_input+frame_index)

        # append to global table.
        df_psnr = df_psnr.append(df_psnr_frm)
        df_sse = df_sse.append(df_sse_frm)

        # CHECKME : # why don't you add each block info to main data frame? no need to transform list to df.

    return df_psnr, df_sse, label_y_pel, input_y_pel

def yuv_diff_single_frame(label_y, input_y, w, h, block_size, scale):

    # normalization
    label_y = label_y / 255.
    input_y = input_y / 255.

    df_psnr = pd.DataFrame()
    df_sse = pd.DataFrame()

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

            # SSE
            each_sse = np.sum((sub_label_y - sub_input_y) ** 2)
            list_sse_row.append(float("{0:.4f}".format(each_sse)))

            block_count += 1

        list_psnr_frame.append(list_psnr_row)
        list_sse_frame.append(list_sse_row)

    # outside of the loop-block
    # stat the block-level PSNR
    # list_psnr to df
    df_psnr_frm = pd.DataFrame(list_psnr_frame)
    df_sse_frm = pd.DataFrame(list_sse_frame)

    # append to global table.
    df_psnr = df_psnr.append(df_psnr_frm)
    df_sse = df_sse.append(df_sse_frm)

    return df_psnr, df_sse


def yuv_diff_temporal(filename_input, start_frame_input, w, h, frame_size, block_size, scale):

    # read yuv : input
    array_input_y, array_input_cbcr = read_yuv420(filename_input, w, h, frame_size, start_frame=start_frame_input)
    input_y = array_input_y.squeeze()

    ## make a copy of pixel domain
    #input_y_pel = input_y.copy()

    # normalization
    input_y = input_y / 255.

    # make a copy of pixel domain
    input_y_pel = input_y.copy()

    # count for each frame
    block_count = 0

    # loop based on block
    list_df_psnr = []
    list_df_ssim = []
    for frame_index in range(1, frame_size):
        list_psnr_frame = []
        list_ssim_frame = []
        for y in range(0, h, block_size):
            list_psnr_row = []
            list_ssim_row = []

            for x in range(0, w, block_size):

                # pick block from input
                sub_input_y_prev = input_y[frame_index-1, y:y+block_size, x:x+block_size]
                sub_input_y_curr = input_y[frame_index,   y:y+block_size, x:x+block_size]

                # PSNR calculation
                each_psnr = psnr(sub_input_y_prev, sub_input_y_curr, scale)
                each_ssim = ssim(sub_input_y_prev, sub_input_y_curr, data_range=1)

                list_psnr_row.append(float("{0:.4f}".format(each_psnr)))
                list_ssim_row.append(float("{0:.4f}".format(each_ssim)))

                block_count += 1

            list_psnr_frame.append(list_psnr_row)
            list_ssim_frame.append(list_ssim_row)

        # outside of the loop

        # stat the block-level PSNR
        # list_psnr to df
        df_psnr = pd.DataFrame(list_psnr_frame)
        df_ssim = pd.DataFrame(list_ssim_frame)

        # append to list
        list_df_psnr.append(df_psnr)
        list_df_ssim.append(df_ssim)

    return list_df_psnr, list_df_ssim, input_y_pel
