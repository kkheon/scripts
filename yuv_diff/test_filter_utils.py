import numpy as np
import math

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


# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

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
            x, y = i * h + 2, j * h + 2

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