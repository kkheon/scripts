import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
import scipy.io
import pickle
from skimage.measure import compare_ssim as ssim

def read_yuv420(filename, w, h, n_frame, start_frame=0):

    y_size = int(w*h)
    uv_size = int(w/2*h/2)
    y_sequence = []
    uv_sequence = []

    # file read
    f = open(filename, "rb")

    # start_frame offset
    if start_frame > 0:
        frame_size = y_size + (2*uv_size)
        f.seek(start_frame * frame_size)

    for i in range(0, n_frame):
        plane_y = np.fromfile(f, dtype=np.uint8, count=y_size).reshape(h, w)
        plane_u = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))
        plane_v = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))

        ## normalize
        #plane_y = plane_y / 255.
        #plane_u = plane_u / 255.
        #plane_v = plane_v / 255.

        # reshape
        plane_y = plane_y.reshape((plane_y.shape[0], plane_y.shape[1], 1))
        plane_u = plane_u.reshape((plane_u.shape[0], plane_u.shape[1], 1))
        plane_v = plane_v.reshape((plane_v.shape[0], plane_v.shape[1], 1))

        # make UV of shape [height, width, color_plane]
        uv = np.concatenate((plane_u, plane_v), axis=2)

        # append to list
        y_sequence.append(plane_y)
        uv_sequence.append(uv)

    f.close()

    # Make list to numpy array. With this transform
    y_array = np.asarray(y_sequence) # [n_frames, h, w, 3]
    uv_array = np.asarray(uv_sequence) # [n_frames, h, w, 3]

    return y_array, uv_array


def chop_yuv420(in_filename, in_w, in_h, out_filename, out_w, out_h, n_frame, start_frame=0):

    # read org
    y, cbcr = read_yuv420(in_filename, in_w, in_h, n_frame, start_frame)

    # get part of org
    y_target = y[:, 0:out_h, 0:out_w]

    h_cbcr_target = int(out_h / 2)
    w_cbcr_target = int(out_w / 2)
    cbcr_target = cbcr[:, 0:h_cbcr_target, 0:w_cbcr_target, :]

    # save as yuv
    yuv_sequence = []
    for each_frame in range(0, n_frame, 1):
        y_1d = y_target[each_frame, :, :].ravel()
        cb_1d = cbcr_target[each_frame, :, :, 0].ravel()
        cr_1d = cbcr_target[each_frame, :, :, 1].ravel()

        merged_yuv = np.concatenate([y_1d.ravel(), cb_1d.ravel(), cr_1d.ravel()])

        # for all the video sequence
        yuv_sequence.append(merged_yuv)

    ### save as yuv
    arr_yuv = np.asarray(yuv_sequence)
    arr_yuv_1d = arr_yuv.ravel()
    arr_yuv_1d.tofile(out_filename)


if __name__ == '__main__':

    #in_h = 1080
    #in_w = 1920

    #out_h = 1072
    #out_w = 1920

    #n_frame = 100

    #input_path = '/hdd2T/kkheon/test_images/ClassB'

    in_h = 2160
    in_w = 3840
    out_h = 2160
    out_w = 3840

    #in_h = 1080
    #in_w = 1920
    #out_h = 1080
    #out_w = 1920

    #
    n_frame = 5
    #n_frame = 60
    start_frame = 0

    ## to get label frame#2
    #start_frame = 2
    #n_frame = 1

    #input_path = '/hdd2T/kkheon/test_images/SJTU_4K'
    #input_path = '/data/kkheon/dataset/NFLX_dataset_public/ref'
    #input_path = '/data/kkheon/dataset/NFLX_dataset_public/dis'

    input_path = '/data/kkheon/dataset/ultra_video_group/org'

    #input_path = '/hdd2T/kkheon/result_vcnn/result_4K_x2_qp32_g3_vcnn_up_4/QP32'
    #input_path = '/hdd2T/kkheon/result_vcnn/result_4K_x2_qp32_g3_vcnn_up_4/QP37'
    #input_path = '/hdd2T/kkheon/result_vcnn/result_4K_x2_qp32_g3_vcnn_up_4/QP42'
    #input_path = '/hdd2T/kkheon/result_vcnn/result_4K_x2_qp32_g3_vcnn_up_4/QP47'

    #input_path = '/hdd2T/kkheon/result_vcnn/val/result_vcnn_up_4_qp32/QP32'
    #input_path = '/hdd2T/kkheon/result_vcnn/val/result_vcnn_up_4_qp32/QP37'

    #input_path = '/hdd2T/kkheon/gen_dataset/output_bugfixed/orig/scenes_yuv/val'

    #input_path = '/home/kkheon/VCNN-Tensorflow/data_vsr/val/result_QP32/result_vcnn_up_4/QP32'
    #input_path = '/home/kkheon/VCNN-Tensorflow/data_vsr/val/result_QP32/result_vcnn_up_4/QP37'
    #input_path = '/home/kkheon/VCNN-Tensorflow/data_vsr/val/result_QP32/result_vcnn_up_4/QP42'
    #input_path = '/home/kkheon/VCNN-Tensorflow/data_vsr/val/result_QP32/result_vcnn_up_4/QP47'

    #input_path = '/hdd2T/kkheon/data_vsr_bak/val_SJTU/result_org_hm/QP51'

    list_input = sorted(glob.glob(os.path.join(input_path, "*.yuv")))

    #output_path = './chopped'
    #output_path = '/hdd2T/kkheon/test_images/ClassB_chopped'
    #output_path = input_path + '_chopped'
    #output_path = input_path + '_chopped_frm' + str(start_frame)
    output_path = input_path + '_chopped_from_' + str(start_frame) + '_n_' + str(n_frame)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for each_input in list_input:
        print('input : ', each_input)

        _, input_name = each_input.rsplit('/', 1)
        input_name, _ = input_name.split('.', 1)
        #output_name = input_name + '_' + str(out_w) + 'x' + str(out_h) + '.yuv'
        output_name = input_name + '.yuv'

        output_name = re.sub('_[0-9]+x[0-9]+','',output_name) 
        each_output = os.path.join(output_path, output_name)

        chop_yuv420(each_input, in_w, in_h, each_output, out_w, out_h, n_frame, start_frame)


