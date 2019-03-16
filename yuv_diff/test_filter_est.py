import numpy as np
import pandas as pd
import os
import math
from read_yuv import read_yuv420
from psnr import psnr
from test_filter_utils import bicubic_interpolation
from test_filter_utils import bicubic
from test_filter_utils import padding
from test_filter_utils import pred_alpha
from utils import plot_diff
from scipy import misc

from test_filter_example import bicubic_interpolation2

from matlab_imresize import imresize

#from scipy import interpolate
## estimate cubic filter
#x_table = input_y_down_rec[each_frame_index, y: y +block_size, x: x +block_size]
#y_table = x_table
#z_table = input_y_down_rec_up[each_frame_index, y_up:y_u p +block_size_up, x_up:x_u p +block_size_up]
#Z = interpolate.griddata((x_table, y_table), z_table, (X, Y), method='cubic')



import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures



#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
#
#X_train = [[1, 2], [2, 4], [6, 7]]
#y_train = [1.2, 4.5, 6.7]
#X_test = [[1, 3], [2, 5]]
#
## create a Linear Regressor
#lin_regressor = LinearRegression()
#
## pass the order of your polynomial here
#poly = PolynomialFeatures(2)
#
## convert to be used further to linear regression
#X_transform = poly.fit_transform(X_train)
#
## fit this to Linear Regressor
#lin_regressor.fit(X_transform,y_train)
#
## get the predictions
#y_preds = lin_regressor.predict(X_test)


#def cubicInterpolate (p, x):
#    return p[1] + \
#           0.5 * x*(p[2] - p[0] +
#                    x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] +
#                       x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

# settings
label_path = "./data_vsr/val/label"
input_path = "./data_vsr/val/label"
output_path = "./result_filter_est"

if not os.path.exists(output_path):
    os.makedirs(output_path)

w = 1920
h = 1080

w_up = 3840
h_up = 2160

scale = 2

block_size = 64
block_size_up = block_size * scale

w_up_in_block = math.ceil(w_up / block_size)
h_up_in_block = math.ceil(h_up / block_size)

start_frame = 0
frame_size = 1

path_target = '/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32'
path_down = path_target + '/result_mf_vcnn_down_3'

# input 1 : label, High Resolution(HR)
path_label = '/home/kkheon/dataset/myanmar_v1_15frm/orig/scenes_yuv/val'

# input 4 : LR's HM result
path_down_rec = path_down + '_hm/QP32'
prefix_down_rec = 'rec_mf_vcnn_down_'
# input 5 : LR's HM result + up
path_down_rec_up = path_target + '/result_mf_vcnn_up_4/QP32'
prefix_down_rec_up = 'mf_vcnn_up_rec_mf_vcnn_down_'

# input list
list_yuv_name = ['scene_53.yuv']
#list_yuv_name = sorted(glob.glob(os.path.join(path_label, "*.yuv")))
#list_yuv_name = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(path_label, "*.yuv")))]


# filter estimation
d0 = np.array([[1, 1.0, 1.0**2, 1.0**3],
               [1, 0.0, 0.0**2, 0.0**3],
               [1, 1.0, 1.0**2, 1.0**3],
               [1, 2.0, 2.0**2, 2.0**3]])

d = np.array([[1, 1.5, 1.5**2, 1.5**3],
              [1, 0.5, 0.5**2, 0.5**3],
              [1, 0.5, 0.5**2, 0.5**3],
              [1, 1.5, 1.5**2, 1.5**3]])

d0 = np.array([[1, 1.75, 1.75**2, 1.75**3],
               [1, 0.75, 0.75**2, 0.75**3],
               [1, 0.25, 0.25**2, 0.25**3],
               [1, 1.25, 1.25**2, 1.25**3]])
d1 = np.array([[1, 1.25, 1.25**2, 1.25**3],
               [1, 0.25, 0.25**2, 0.25**3],
               [1, 0.75, 0.75**2, 0.75**3],
               [1, 1.75, 1.75**2, 1.75**3]])

#== order = [1, d, d^2, d^3]
#h = [[-4*a, 8*a, -5*a, a], [1, 0, -(a+3), a+2], [1, 0, -(a+3), a+2], [-4*a, 8*a, -5*a, a]]

w_a = np.array([[-4, 8, -5, 1], [0, 0, -1, 1], [0, 0, -1, 1], [-4, 8, -5, 1]])
w_a_transpose = np.transpose(w_a)
w_c = np.array([[0, 0, 0, 0], [1, 0, -3, 2], [1, 0, -3, 2], [0, 0, 0, 0]])
w_c_transpose = np.transpose(w_c)

d_w_a = []
d_w_c = []
for i in range(4):
    d_w_a.append(np.matmul(d[i], np.transpose(w_a[i])))
    d_w_c.append(np.matmul(d[i], np.transpose(w_c[i])))

d0_w_a = []
d0_w_c = []
for i in range(4):
    d0_w_a.append(np.matmul(d0[i], np.transpose(w_a[i])))
    d0_w_c.append(np.matmul(d0[i], np.transpose(w_c[i])))
d1_w_a = []
d1_w_c = []
for i in range(4):
    d1_w_a.append(np.matmul(d1[i], np.transpose(w_a[i])))
    d1_w_c.append(np.matmul(d1[i], np.transpose(w_c[i])))

# dataframe for saving raw data
list_columns_raw = ['img_name', 'frame_idx', 'x', 'y',
                    'pred_a', 'psnr_cnn_up', 'psnr_bicubic_up_pred',
                    'pred_a_from_label', 'psnr_bicubic_up_pred_from_label',
                    'psnr_bicubic_up', 'psnr_diff',
                    'psnr_bicubic_up_scipy', 'psnr_bicubic_up_matlab']
df_raw = pd.DataFrame(columns=list_columns_raw)

# save each image's PSNR result as file.
for idx, each_yuv in enumerate(list_yuv_name):
    # load label
    each_label = os.path.join(path_label, each_yuv)
    array_label_y, array_label_cbcr = read_yuv420(each_label, w_up, h_up, frame_size, start_frame=start_frame)
    label_y = array_label_y.squeeze(axis=3)

    # input : load LR+HM
    each_down_yuv_rec = os.path.join(path_down_rec, prefix_down_rec + each_yuv)
    array_input_y, array_input_cbcr = read_yuv420(each_down_yuv_rec, w, h, frame_size, start_frame=start_frame)
    input_y_down_rec = array_input_y.squeeze(axis=3)

    # target : load LR+HM+CNN_UP
    each_down_rec_up = os.path.join(path_down_rec_up, prefix_down_rec_up + each_yuv)
    #df_psnr_down_rec_up, df_sse_down_rec_up, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_down_rec_up, start_frame, w_up, h_up, block_size_up, scale, frame_size)
    array_input_y, array_input_cbcr = read_yuv420(each_down_rec_up, w_up, h_up, frame_size, start_frame=start_frame)
    input_y_down_rec_up = array_input_y.squeeze(axis=3)

    # compare 0 : HR
    # compare 1 : LR+HM+bicubic_UP

    w_in_block = int(w / block_size)
    h_in_block = int(h / block_size)

    for each_frame_index in range(0, frame_size):
        df_raw_frm = pd.DataFrame(columns=list_columns_raw)
        input_y_down_rec_frame = input_y_down_rec[each_frame_index, :, :]

        # padding with the edge values of array. (option : edge)
        #input_y_down_rec_frame_padded = np.pad(input_y_down_rec_frame, ((1, 2), (1, 2)), 'edge')
        input_y_down_rec_frame_padded = np.pad(input_y_down_rec_frame, ((2, 2), (2, 2)), 'symmetric')
        #input_y_down_rec_frame_padded = np.pad(input_y_down_rec_frame, ((2, 2), (2, 2)), 'edge')

        # bicubic up using scipy
        input_y_down_rec_frame_bicubic_up = misc.imresize(input_y_down_rec_frame, (h_up, w_up), interp='bicubic')

        # bicubic up using imresize
        input_y_down_rec_frame_bicubic_up_matlab = imresize(input_y_down_rec_frame, 2, (h_up, w_up))

        ## bicubic up using bicubic1
        ## padding
        #input_y_down_rec_frame_padded1 = padding(input_y_down_rec_frame, h, w)
        #input_y_down_rec_frame_bicubic_up1 = bicubic(input_y_down_rec_frame_padded1, h, w, ratio=2, a=-1/2)

        #for y in range(0, h, block_size):
        for h_index in range(0, h_in_block):
            #print("loop-of-y: %d / %d" % (y, h))
            print("loop-of-y: %d / %d" % (h_index, h_in_block))
            #for x in range(0, w, block_size):
            for w_index in range(0, w_in_block):

                x = w_index * block_size
                y = h_index * block_size

                # block index
                y_up = int(y * 2)
                x_up = int(x * 2)

                # block index
                y_in_block = int(y / block_size)
                x_in_block = int(x / block_size)

                data = input_y_down_rec_frame[y:y+block_size, x:x+block_size]
                target = input_y_down_rec_up[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up]
                label = label_y[each_frame_index, y_up:y_up+block_size_up, x_up:x_up+block_size_up]
                data_bicubic_up_scipy = input_y_down_rec_frame_bicubic_up[y_up:y_up+block_size_up, x_up:x_up+block_size_up]

                data_bicubic_up_matlab = input_y_down_rec_frame_bicubic_up_matlab[y_up:y_up+block_size_up, x_up:x_up+block_size_up]

                #data_bicubic_up1 = input_y_down_rec_frame_bicubic_up1[y_up:y_up+block_size_up, x_up:x_up+block_size_up]


                # more pixels are needed [x-1, x+2]
                #data_padded = input_y_down_rec_frame_padded[y:y+block_size+3, x:x+block_size+3]
                # more pixels are needed [x-2, x+2]
                data_padded = input_y_down_rec_frame_padded[y:y+block_size+4, x:x+block_size+4]

                # finding 'a' from CNN_UP
                best_a_from_cnn_up = pred_alpha(block_size, target, data_padded, d0_w_a, d0_w_c, d1_w_a, d1_w_c)

                # finding 'a' from label
                best_a_from_label = pred_alpha(block_size, label, data_padded, d0_w_a, d0_w_c, d1_w_a, d1_w_c)

                # do interpolation with 'best_a'
                #pred_block_2d = bicubic_interpolation(best_a, d_w_a, d_w_c, block_size, data_padded)
                pred_block_2d = bicubic(data_padded, block_size, block_size, 2, best_a_from_cnn_up)
                pred_block_2d = np.clip(pred_block_2d, 0, 255)

                # do interpolation with 'a=-0.5'
                #bicubic_block_2d = bicubic_interpolation(-0.5, d_w_a, d_w_c, block_size, data_padded)
                bicubic_block_2d = bicubic(data_padded, block_size, block_size, 2, -0.5)
                bicubic_block_2d = np.clip(bicubic_block_2d, 0, 255)

                # do interpolation with 'best_a_from_label'
                pred_block_2d_from_label = bicubic(data_padded, block_size, block_size, 2, best_a_from_label)
                pred_block_2d_from_label = np.clip(pred_block_2d_from_label, 0, 255)

                ## bicubic2
                #xx = np.linspace(-2, block_size + 2, block_size + 4)
                #yy = np.linspace(-2, block_size + 2, block_size + 4)
                #xx_new = np.linspace(-2, block_size + 2, block_size_up+4*2)
                #yy_new = np.linspace(-2, block_size + 2, block_size_up+4*2)
                #bicubic_block_2d = bicubic_interpolation2(xx, yy, data_padded, xx_new, yy_new)
                #bicubic_block_2d = bicubic_block_2d[4:4+128, 4:4+128]


                # block-level comparision
                # diff with label
                # diff 1 : cnn-up
                psnr_cnn_up = psnr(target / 255., label / 255., 0)

                # diff 2 : bicubic-up with new 'a'
                psnr_bicubic_up_pred = psnr(pred_block_2d / 255., label / 255., 0)

                # diff 2 : bicubic-up with new 'a'
                psnr_pred_a_from_label = psnr(pred_block_2d_from_label / 255., label / 255., 0)

                # diff 3 : bicubic-up
                psnr_bicubic_up = psnr(bicubic_block_2d / 255., label / 255., 0)

                # diff 4 : bicubic-up by scipy.misc.imresize
                psnr_bicubic_up_scipy = psnr(data_bicubic_up_scipy / 255., label / 255., 0)

                ## diff 5 : bicubic-up by
                #psnr_bicubic_up1 = psnr(data_bicubic_up1 / 255., label / 255., 0)

                # diff 6 : bicubic-up by matlab
                psnr_bicubic_up_matlab = psnr(data_bicubic_up_matlab / 255., label / 255., 0)

                #=== plot
                imgs = []
                psnrs = [None]  # label, lr, enc(lr)
                imgs.append(label)
                imgs.append(target)
                imgs.append(pred_block_2d)
                imgs.append(pred_block_2d_from_label)
                imgs.append(bicubic_block_2d)
                imgs.append(data_bicubic_up_scipy)
                #imgs.append(data_bicubic_up1)
                imgs.append(data_bicubic_up_matlab)

                psnrs.append(psnr_cnn_up)
                psnrs.append(psnr_bicubic_up_pred)
                psnrs.append(psnr_pred_a_from_label)
                psnrs.append(psnr_bicubic_up)
                psnrs.append(psnr_bicubic_up_scipy)
                #psnrs.append(psnr_bicubic_up1)
                psnrs.append(psnr_bicubic_up_matlab)

                xlabels = ['ORG', 'CNN_UP', 'BICUBIC_UP\n(pred_from_cnn_up)', 'BICUBIC_UP\n(pred_from_label)',
                           'BICUBIC_UP\n(a=-0.5)', 'BICUBIC_UP\n(scipy)', 'BICUBIC_UP\n(matlab)']

                # separate dir depending on PSNR
                # psnr diff
                #psnr_diff = psnr_bicubic_up_pred - psnr_bicubic_up
                psnr_diff = psnr_bicubic_up - psnr_bicubic_up_scipy

                if math.isinf(psnr_diff):
                    psnr_diff = 99.0

                psnr_diff_floor = math.floor(psnr_diff)
                str_psnr_diff_floor = str(psnr_diff_floor)
                output_path_psnr = os.path.join(output_path, 'group_psnr_' + str_psnr_diff_floor)

                if not os.path.exists(output_path_psnr):
                    os.makedirs(output_path_psnr)

                plot_diff(imgs, psnrs, xlabels, idx, each_frame_index, x, y, save_dir=output_path_psnr)

                # to save the raw data
                list_raw = [[each_yuv, each_frame_index, x, y,
                             best_a_from_cnn_up, psnr_cnn_up, psnr_bicubic_up_pred,
                             best_a_from_label, psnr_pred_a_from_label,
                             psnr_bicubic_up, psnr_diff, psnr_bicubic_up_scipy, psnr_bicubic_up_matlab]]

                df_raw_frm = df_raw_frm.append(pd.DataFrame(list_raw, columns=list_columns_raw))

        # append to total_df
        df_raw = df_raw.append(df_raw_frm)


# save df_raw as csv
filename_psnr = os.path.join(output_path, 'df_raw')
df_raw.to_csv(filename_psnr + '.txt', sep=' ')

