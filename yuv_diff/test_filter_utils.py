import numpy as np
import math

# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def bicubic_interpolation(a, d_w_a, d_w_c, block_size, data_padded ):
    # do interpolation with new 'a'
    # calculate 'w' with new 'a'
    new_w = [a * A + B for A, B in zip(d_w_a, d_w_c)]

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
        pred_block_2d.append(pred_block_1d[i+1, :])
        pred_block_2d.append(pred_block_transpose[i, :])

    pred_block_2d = np.array(pred_block_2d)
    pred_block_2d_round = np.around(pred_block_2d, decimals=0)
    #print(pred_block_2d)

    return pred_block_2d_round


def bicubic_interpolation1(a, d_w_a, d_w_c, block_size, data_padded ):
    # do interpolation with new 'a'
    # calculate 'w' with new 'a'
    new_w = [a * A + B for A, B in zip(d_w_a, d_w_c)]

    # 1D-interpolation
    pred_block = []
    for each_row_index in range(block_size+3):

        pred_row = []
        for each_pixel_index in range(block_size):
            x1 = each_pixel_index
            mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])

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
        pred_block_2d.append(pred_block_1d[i+1, :])
        pred_block_2d.append(pred_block_transpose[i, :])

    pred_block_2d = np.array(pred_block_2d)
    pred_block_2d_round = np.around(pred_block_2d, decimals=0)
    #print(pred_block_2d)

    return pred_block_2d_round


#Paddnig
def padding(img,H,W):
    zimg = np.zeros((H+4,W+4))
    zimg[2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2]=img[:,0:1]
    zimg[H+2:H+4,2:W+2]=img[H-1:H,:]
    zimg[2:H+2,W+2:W+4]=img[:,W-1:W]
    zimg[0:2,2:W+2]=img[0:1,:]
    #Pad the missing eight points
    zimg[0:2,0:2]=img[0,0]
    zimg[H+2:H+4,0:2]=img[H-1,0]
    zimg[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    zimg[0:2,W+2:W+4]=img[0,W-1]
    return zimg

# Bicubic operation
def bicubic(img, H, W, ratio, a):

    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW))

    h = 1/ratio

    for j in range(dH):
        for i in range(dW):
            #x, y = i * h + 2, j * h + 2
            #x, y = i * h + 1 + h, j * h + 1 + h

            ###==ref : u = x / scale + 0.5 * (1 - 1 / scale)
            x, y = i * h + 1.5 + 0.5 * (1 - h), j * h + 1.5 + 0.5 * (1 - h)

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
            mat_m = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                               [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                               [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                               [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])
            mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
            #dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
            dst[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r)

    return dst


#def pred_alpha(block_size, target, data_padded, d_w_a, d_w_c):
def pred_alpha(block_size, target, data_padded, d0_w_a, d0_w_c, d1_w_a, d1_w_c):

    list_A = []
    list_B = []
    list_B_minus_P = []
    list_a = []
    list_P = []

    # finding 'a' from CNN_UP
    # test 0 : solve a for row 0
    for each_row_index in range(block_size):  # because calculate a from 4x4 block? no. it should iterate based on block size
        for each_pixel_index in range(block_size):
            #target_pixel = target[each_row_index, 3]

            target_pixel = target[2*each_row_index, 2*each_pixel_index + 1]
            #input = np.transpose(data_padded[each_row_index+1, each_pixel_index:each_pixel_index+4])
            input = np.transpose(data_padded[each_row_index+2, each_pixel_index+1:each_pixel_index+5])

            # calculate distance depend on position

            if (each_pixel_index % 2) == 0:
                d_w_a_p = np.matmul(d0_w_a, input)
                d_w_c_p = np.matmul(d0_w_c, input)
            else:
                d_w_a_p = np.matmul(d1_w_a, input)
                d_w_c_p = np.matmul(d1_w_c, input)

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

            target_pixel = target[2*each_pixel_index + 1, 2*each_col_index]
            #input = np.transpose(data_padded[each_pixel_index:each_pixel_index+4, each_col_index+1])
            input = np.transpose(data_padded[each_pixel_index + 1:each_pixel_index+5, each_col_index+2])

            #d_w_a_p = np.matmul(d_w_a, input)
            #d_w_c_p = np.matmul(d_w_c, input)

            if (each_pixel_index % 2) == 0:
                d_w_a_p = np.matmul(d0_w_a, input)
                d_w_c_p = np.matmul(d0_w_c, input)
            else:
                d_w_a_p = np.matmul(d1_w_a, input)
                d_w_c_p = np.matmul(d1_w_c, input)

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

    return best_a
