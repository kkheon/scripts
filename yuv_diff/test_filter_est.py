import numpy as np
import os
import math
from read_yuv import read_yuv420

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

w = 1920
h = 1080

w_up = 3840
h_up = 2160

scale = 2

block_size = 4
block_size_up = block_size * scale

w_up_in_block = math.ceil(w_up / block_size)
h_up_in_block = math.ceil(h_up / block_size)

start_frame = 0
frame_size = 5

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
d = np.array([[1, 1.5, 1.5**2, 1.5**3],
              [1, 0.5, 0.5**2, 0.5**3],
              [1, 0.5, 0.5**2, 0.5**3],
              [1, 1.5, 1.5**2, 1.5**3]])

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

# save each image's PSNR result as file.
for idx, each_yuv in enumerate(list_yuv_name):

    # input : load LR+HM
    each_down_yuv_rec = os.path.join(path_down_rec, prefix_down_rec + each_yuv)
    array_input_y, array_input_cbcr = read_yuv420(each_down_yuv_rec, w, h, frame_size, start_frame=start_frame)
    input_y_down_rec = array_input_y.squeeze(axis=3)

    # target : load LR+HM+CNN_UP
    ## diff label-LR-UP
    each_down_rec_up = os.path.join(path_down_rec_up, prefix_down_rec_up + each_yuv)
    #df_psnr_down_rec_up, df_sse_down_rec_up, label_y, input_y_down_rec_up = yuv_diff_n_frame(each_label, start_frame, each_down_rec_up, start_frame, w_up, h_up, block_size_up, scale, frame_size)
    array_input_y, array_input_cbcr = read_yuv420(each_down_rec_up, w_up, h_up, frame_size, start_frame=start_frame)
    input_y_down_rec_up = array_input_y.squeeze(axis=3)

    # compare 0 : HR
    # compare 1 : LR+HM+bicubic_UP

    w_in_block = int(w / block_size)
    for each_frame_index in range(0, frame_size):
        input_y_down_rec_frame = input_y_down_rec[each_frame_index, :, :]
        input_y_down_rec_up_frame = input_y_down_rec_up[each_frame_index, :, :]

        # padding with the edge values of array. (option : edge)
        input_y_down_rec_frame_padded = np.pad(input_y_down_rec_frame, ((1, 2), (1, 2)), 'edge')

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                # block index
                y_up = int(y * 2)
                x_up = int(x * 2)

                # block index
                y_in_block = int(y / block_size)
                x_in_block = int(x / block_size)

                data = input_y_down_rec_frame[y:y+block_size, x:x+block_size]
                target = input_y_down_rec_up_frame[y_up:y_up+block_size_up, x_up:x_up+block_size_up]

                # more pixels are needed [x-1, x+2]
                data_padded = input_y_down_rec_frame_padded[y:y+block_size+3, x:x+block_size+3]

                list_A = []
                list_B = []
                list_B_minus_P = []
                list_a = []
                list_P = []

                # test 0 : solve a for row 0
                for each_row_index in range(block_size):  # because calculate a from 4x4 block? no. it should iterate based on block size
                    for each_pixel_index in range(block_size):
                        #target_pixel = target[each_row_index, 3]
                        target_pixel = target[each_row_index, 2*each_pixel_index + 1]

                        input = np.transpose(data_padded[each_row_index, each_pixel_index:each_pixel_index+4])

                        d_w_a_p = np.matmul(d_w_a, input)
                        d_w_c_p = np.matmul(d_w_c, input)

                        # solve a
                        a_sol = (target_pixel - d_w_c_p) / d_w_a_p
                        #print(a_sol)

                        # append to list
                        list_A.append(d_w_a_p)
                        list_B.append(d_w_c_p)
                        list_B_minus_P.append(d_w_c_p - target_pixel)
                        list_a.append(a_sol)
                        list_P.append(target_pixel)

                for each_col_index in range(block_size):  # because calculate a from 4x4 block? no. it should iterate based on block size
                    for each_pixel_index in range(block_size):
                        #target_pixel = target[each_row_index, 3]
                        target_pixel = target[2*each_pixel_index + 1, each_col_index]
                        input = np.transpose(data_padded[each_pixel_index:each_pixel_index+4, each_col_index])

                        d_w_a_p = np.matmul(d_w_a, input)
                        d_w_c_p = np.matmul(d_w_c, input)

                        # solve a
                        a_sol = (target_pixel - d_w_c_p) / d_w_a_p
                        #print(a_sol)

                        # append to list
                        list_A.append(d_w_a_p)
                        list_B.append(d_w_c_p)
                        list_B_minus_P.append(d_w_c_p - target_pixel)
                        list_a.append(a_sol)
                        list_P.append(target_pixel)

                # remove redundancy of 'a' and sort 'a'
                period_a = sorted(set(list_a))

                best_a = -0.5
                list_loss = [abs(best_a * A + B) for A, B in zip(list_A, list_B_minus_P)]
                base_loss = sum(list_loss)
                min_loss = base_loss

                # for each period of a, calculate
                for each_a in period_a:
                    list_loss = [abs(each_a * A + B) for A, B in zip(list_A, list_B_minus_P)]
                    loss = sum(list_loss)

                    if loss < min_loss:
                        best_a = each_a
                        min_loss = loss

                ## run with new 'a'
                #list_pred = [best_a * A + B for A, B in zip(list_A, list_B)]
                #list_error = [abs(A - B) for A, B in zip(list_pred, list_P)]
                #print(sum(list_error))

                ## run with base 'a=-0.5'
                #list_pred_base = [(-0.5) * A + B for A, B in zip(list_A, list_B)]
                #list_error_base = [abs(A - B) for A, B in zip(list_pred_base, list_P)]
                #print(sum(list_error_base))

                # do interpolation with new 'a'
                # calculate 'w' with new 'a'
                new_w = [best_a * A + B for A, B in zip(d_w_a, d_w_c)]

                # 1D-interpolation
                pred_block = []
                for each_row_index in range(block_size+3):
                    pred_row = []
                    for each_pixel_index in range(block_size):
                       input = np.transpose(data_padded[each_row_index, each_pixel_index:each_pixel_index+4])

                       pred = np.matmul(new_w, input)
                       pred_row.append(pred)

                    pred_block.append(pred_row)

                # pred_block to array
                pred_block = np.array(pred_block)

                # merge data_padded and pred_block
                pred_block_1d = np.empty(shape=(block_size + 3, 0))

                ## stack padded pixel first.
                #pred_block_1d = np.column_stack([pred_block_1d, data_padded[:, 0]])

                # stack column by column : but is pred_block is padded? this could be a problem.
                for i in range(block_size):
                    pred_block_1d = np.column_stack([pred_block_1d, data_padded[:, i+1], pred_block[:, i]])

                ## stack padded pixel last.
                #pred_block_1d = np.column_stack([pred_block_1d, data_padded[:, -2], data_padded[:, -1]])


                # 1D-interpolation : column-wise
                pred_block = []
                pred_block_row_size, pred_block_col_size = pred_block_1d.shape
                #for each_col_index in range(block_size+3+1):
                for each_col_index in range(pred_block_col_size):
                    pred_col = []
                    for each_pixel_index in range(block_size):
                       input = np.transpose(pred_block_1d[each_pixel_index:each_pixel_index+4, each_col_index])

                       pred = np.matmul(new_w, input)
                       pred_col.append(pred)

                    pred_block.append(pred_col)

                # pred_block to array
                pred_block = np.array(pred_block)
                pred_block_transpose = np.transpose(pred_block)

                # merge data_padded and pred_block
                #pred_block_2d = np.empty(shape=(0, block_size_up))
                pred_block_2d = []

                # stack row by row
                for i in range(block_size):
                    #pred_block_2d = np.stack([pred_block_2d, pred_block_1d[i+1, :], pred_block_transpose[i, :]])
                    #pred_block_2d = np.stack([pred_block_1d[i+1, :], pred_block_transpose[i, :]])
                    pred_block_2d.append(pred_block_1d[i+1, :])
                    pred_block_2d.append(pred_block_transpose[i, :])

                pred_block_2d = np.array(pred_block_2d)

                print(pred_block_2d)



                # block-level comparision

