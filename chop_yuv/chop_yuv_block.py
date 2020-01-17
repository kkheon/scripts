import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
import scipy.io
import pickle
from skimage.measure import compare_ssim as ssim
from read_yuv import read_yuv420

def chop_yuv420_block(in_filename, in_w, in_h, out_filename, out_w, out_h, block_size_w, block_size_h, n_frame, start_frame=0):

    # read org
    y, cbcr = read_yuv420(in_filename, in_w, in_h, n_frame, start_frame)

    # get part of org
    y_target = y[:, 0:out_h, 0:out_w]

    h_cbcr_target = int(out_h / 2)
    w_cbcr_target = int(out_w / 2)
    cbcr_target = cbcr[:, 0:h_cbcr_target, 0:w_cbcr_target, :]

    # save as yuv
    for y in range(0, out_h, block_size_h):
        print( '\ty=%04d' % y)
        
        for x in range(0, out_w, block_size_w):
            out_filename_postfix = "_%04dx%04d" % (x, y)
            out_file = out_filename + out_filename_postfix + '.yuv'

            x_cbcr= int(x/2)
            y_cbcr= int(y/2)
            block_size_h_cbcr = int(block_size_h/2)
            block_size_w_cbcr = int(block_size_w/2)

            yuv_sequence = []
            for each_frame in range(0, n_frame, 1):
                y_1d = y_target[each_frame, y:y+block_size_h, x:x+block_size_w].ravel()
                cb_1d = cbcr_target[each_frame, y_cbcr:y_cbcr+block_size_h_cbcr, x_cbcr:x_cbcr+block_size_w_cbcr, 0].ravel()
                cr_1d = cbcr_target[each_frame, y_cbcr:y_cbcr+block_size_h_cbcr, x_cbcr:x_cbcr+block_size_w_cbcr, 1].ravel()

                merged_yuv = np.concatenate([y_1d.ravel(), cb_1d.ravel(), cr_1d.ravel()])

                # for all the video sequence
                yuv_sequence.append(merged_yuv)

            ### save as yuv
            arr_yuv = np.asarray(yuv_sequence)
            arr_yuv_1d = arr_yuv.ravel()
            arr_yuv_1d.tofile(out_file)


if __name__ == '__main__':

    #in_h = 1080
    #in_w = 1920

    #out_h = 1072
    #out_w = 1920

    #n_frame = 100

    in_h = 2160
    in_w = 3840
    out_h = 2160
    out_w = 3840

    #in_h = 1080
    #in_w = 1920
    #out_h = 1080
    #out_w = 1920

    block_size_w = 64
    block_size_h = 64

    out_h = int(int(in_h / block_size_h) * block_size_h)
    out_w = int(int(in_w / block_size_w) * block_size_w)
    #
    n_frame = 5
    #n_frame = 60
    start_frame = 0

    ## to get label frame#2
    #start_frame = 2
    #n_frame = 1

    #input_path = "/data/kkheon/dataset/SJTU_4K_test/label"
    #input_path = "/data/kkheon/dataset/SJTU_4K_test/label_hm/QP27"
    #input_path = "/data/kkheon/dataset/SJTU_4K_test/label_hm/QP32"
    #input_path = "/data/kkheon/dataset/SJTU_4K_test/label_hm/QP37"
    #input_path = "/data/kkheon/dataset/SJTU_4K_test/label_hm/QP42"
    input_path = "/data/kkheon/dataset/SJTU_4K_test/label_hm/QP47"

    #input_path = '/hdd2T/kkheon/test_images/SJTU_4K'
    #input_path = '/data/kkheon/dataset/NFLX_dataset_public/ref'
    #input_path = '/data/kkheon/dataset/NFLX_dataset_public/dis'
    list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    #output_path = './chopped'
    #output_path = '/hdd2T/kkheon/test_images/ClassB_chopped'
    #output_path = input_path + '_chopped'
    #output_path = input_path + '_chopped_frm' + str(start_frame)
    #output_path = input_path + '_chopped_from' + str(start_frame) + '_n' + str(n_frame)
    output_path = input_path + '_chopped_block_size_' + str(block_size_w) + 'x' + str(block_size_h)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for each_input in list_input:
        print('input : ', each_input)

        _, input_name = each_input.rsplit('/', 1)
        input_name, _ = input_name.split('.', 1)
        #output_name = input_name + '_' + str(out_w) + 'x' + str(out_h) + '.yuv'
        output_name = input_name
        each_output = os.path.join(output_path, output_name)

        chop_yuv420_block(each_input, in_w, in_h, each_output, out_w, out_h, block_size_w, block_size_h, n_frame, start_frame)


