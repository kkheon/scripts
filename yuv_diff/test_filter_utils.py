import numpy as np
import math

# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def u_a(s):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (abs(s)**3)-(abs(s)**2)
    elif (abs(s) > 1) & (abs(s) <= 2):
        return (abs(s)**3)-5*(abs(s)**2)+8*abs(s)-4
    return 0

def u_c(s):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (2)*(abs(s)**3)-(3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return 0
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

            ## for debug : test partial sum
            #mat_l_a = np.matrix([[u_a(x1), u_a(x2), u_a(x3), u_a(x4)]])
            #mat_r_a = np.matrix([[u_a(y1)], [u_a(y2)], [u_a(y3)], [u_a(y4)]])

            #a2 = np.dot(np.dot(mat_l_a, mat_m), mat_r_a)

            ## term a^1
            #mat_l_c = np.matrix([[u_c(x1), u_c(x2), u_c(x3), u_c(x4)]])
            #mat_r_c = np.matrix([[u_c(y1)], [u_c(y2)], [u_c(y3)], [u_c(y4)]])

            #a1 = np.dot(np.dot(mat_l_a, mat_m), mat_r_c) + np.dot(np.dot(mat_l_c, mat_m), mat_r_a)

            ## term a^0
            #constant = np.dot(np.dot(mat_l_c, mat_m), mat_r_c)

            #curr_pixel = a2 * (a ** 2) + a1 * a + constant

            #diff = dst[j, i] - curr_pixel


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


def solve_quadratic_equation(a, b, c):
    d = b ** 2 - 4 * a * c  # discriminant

    if d < 0:
        #print("This equation has no real solution")
        return []
    elif d == 0:
        x = (-b + math.sqrt(b ** 2 - 4 * a * c)) / 2 * a
        #print(("This equation has one solutions: "), x)
        return [x]
    # add the extra () above or it does not show the answer just the text.
    else:
        x1 = (-b + math.sqrt((b ** 2) - (4 * (a * c)))) / (2 * a)
        x2 = (-b - math.sqrt((b ** 2) - (4 * (a * c)))) / (2 * a)
        #print("This equation has two solutions: ", x1, " or", x2)
        return [x1, x2]

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    #dist_2 = np.sum((nodes - node)**2, axis=1)
    dist_2 = (nodes - node)**2
    return nodes[np.argmin(dist_2)]

def farthest_node(node, nodes):
    nodes = np.asarray(nodes)
    #dist_2 = np.sum((nodes - node)**2, axis=1)
    dist_2 = (nodes - node)**2
    return nodes[np.argmax(dist_2)]

def get_loss(a_candi, a2, a1, constant, target):
    # pred pixel
    curr_pixel = a2 * (a_candi ** 2) + a1 * a_candi + constant
    curr_pixel_clipped = np.clip(curr_pixel, 0, 255)
    curr_pixel_clipped_around = np.around(curr_pixel_clipped).astype(np.uint8)

    # type change : uint8 => int
    curr_pixel_clipped_around_int = curr_pixel_clipped_around.astype(np.int)

    target_clipped = np.clip(target, 0, 255)
    target_int = np.around(target_clipped).astype(np.int)
    curr_pixel_diff = np.subtract(curr_pixel_clipped_around_int, target_int)

    # l1 loss
    #curr_loss_pixel = np.sum(np.absolute(curr_pixel_diff))

    # l2 loss
    curr_loss_pixel = np.sum(np.square(curr_pixel_diff))

    return curr_loss_pixel, curr_pixel_clipped_around_int


# Bicubic operation
def pred_alpha_2d(input, H, W, ratio, target):

    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)

    a2 = np.zeros((dH, dW))
    a1 = np.zeros((dH, dW))
    a0 = np.zeros((dH, dW))

    constant = np.zeros((dH, dW))

    list_period = []

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

            # term a^2
            mat_l_a = np.matrix([[u_a(x1), u_a(x2), u_a(x3), u_a(x4)]])
            mat_m = np.matrix([[input[int(y - y1), int(x - x1)], input[int(y - y2), int(x - x1)], input[int(y + y3), int(x - x1)], input[int(y + y4), int(x - x1)]],
                               [input[int(y - y1), int(x - x2)], input[int(y - y2), int(x - x2)], input[int(y + y3), int(x - x2)], input[int(y + y4), int(x - x2)]],
                               [input[int(y - y1), int(x + x3)], input[int(y - y2), int(x + x3)], input[int(y + y3), int(x + x3)], input[int(y + y4), int(x + x3)]],
                               [input[int(y - y1), int(x + x4)], input[int(y - y2), int(x + x4)], input[int(y + y3), int(x + x4)], input[int(y + y4), int(x + x4)]]])
            mat_r_a = np.matrix([[u_a(y1)], [u_a(y2)], [u_a(y3)], [u_a(y4)]])

            a2[j, i] = np.dot(np.dot(mat_l_a, mat_m), mat_r_a)

            # term a^1
            mat_l_c = np.matrix([[u_c(x1), u_c(x2), u_c(x3), u_c(x4)]])
            mat_r_c = np.matrix([[u_c(y1)], [u_c(y2)], [u_c(y3)], [u_c(y4)]])

            a1[j, i] = np.dot(np.dot(mat_l_a, mat_m), mat_r_c) + np.dot(np.dot(mat_l_c, mat_m), mat_r_a)

            # term a^0
            constant[j, i] = np.dot(np.dot(mat_l_c, mat_m), mat_r_c)
            a0[j, i] = np.dot(np.dot(mat_l_c, mat_m), mat_r_c) - target[j, i]

            # solve quadratic equation of a.
            if a2[j, i] == 0:  # linear equation
                list_sol = [- a0[j, i] / a1[j, i]]
            else:
                list_sol = solve_quadratic_equation(a2[j, i], a1[j, i], a0[j, i])

            # add to period list
            list_period = list_period + list_sol

    # after 2D loop,
    # remove redundancy of 'a' and sort 'a'
    list_period = [x for x in list_period if str(x) != 'nan']
    list_period_sorted = sorted(set(list_period))

    # bicubic alpha
    best_a = -0.5

    # calculate current period's sum
    sum = (best_a ** 2) * a2 + (best_a * a1) + a0
    sum_abs = np.absolute(sum)
    bicubic_loss = np.sum(sum_abs)
    bicubic_loss_pixel, pred_bicubic = get_loss(best_a, a2, a1, constant, target)
    pred_best_a = pred_bicubic

    ## for debug
    #bicubic_block_2d = bicubic(input, H, W, 2, -0.5)
    #bicubic_loss_pixel_with_bicubic = get_loss(best_a, a2, a1, constant, bicubic_block_2d)


    # defalult alpha
    min_loss = bicubic_loss
    min_loss_pixel = bicubic_loss_pixel

    # TODO : add last period
    # for each period, (a= period < each_period)
    for period_index, each_period in enumerate(list_period_sorted):

        # last period
        if period_index == 0:
            #last_period = None
            last_period = - np.inf
            each_period_a = each_period - 0.01
        else:
            last_period = list_period_sorted[period_index - 1]
            each_period_a = (each_period + last_period) / 2

        if each_period_a == np.inf:
            continue

        list_period_curr = [last_period, each_period]
        list_period_curr = [v for v in list_period_curr if not (math.isinf(v) or math.isnan(v))]

        # determine each equation's sign and sum the coefficients.
        each_period_a2 = each_period_a ** 2

        # calculate current period's sum    # TODO : does it work?
        sum = each_period_a2 * a2 + each_period_a * a1 + a0

        # extract sign of each element
        sign = np.sign(sum)

        # get abs value of current period by multiplying each element's sign.
        a2_abs = np.multiply(sign, a2)
        a1_abs = np.multiply(sign, a1)
        a0_abs = np.multiply(sign, a0)

        # sum all the values.
        a2_abs_sum = np.sum(a2_abs)
        a1_abs_sum = np.sum(a1_abs)
        a0_abs_sum = np.sum(a0_abs)

        axis_of_symmetry = - a1_abs_sum / (2 * a2_abs_sum)

        # if a2 > 0, calculate derivative. and calculate loss for the point where derivative=0.
        # check if axis_of_symmetry is in this period.
        if a2_abs_sum > 0:
            axis_of_symmetry_is_here = False
            if (last_period == None):
                if (axis_of_symmetry < each_period):
                    axis_of_symmetry_is_here = True
            elif (axis_of_symmetry > last_period) and (axis_of_symmetry < each_period):
                axis_of_symmetry_is_here = True

            if axis_of_symmetry_is_here:
                # check the loss of center
                a_candi = axis_of_symmetry
            else:
                # find close one from center
                a_candi = closest_node(axis_of_symmetry, list_period_curr)

        elif a2_abs_sum < 0:
            # find far one from center
            a_candi = farthest_node(axis_of_symmetry, list_period_curr)
        else: #a2_abs_sum == 0
            # how can I treat this?
            if a1_abs_sum >= 0:
                a_candi = last_period
            else:
                a_candi = each_period

        # calculate loss
        curr_loss_pixel, pred_a_candi = get_loss(a_candi, a2, a1, constant, target)

        # curr loss from equation
        curr_loss = a2_abs_sum * (a_candi ** 2) + a1_abs_sum * a_candi + a0_abs_sum

        # diff because of clipping and rounding
        diff_clipping = curr_loss - curr_loss_pixel
        diff_bicubic_loss = curr_loss - bicubic_loss
        diff_bicubic_loss_pixel = curr_loss_pixel - bicubic_loss_pixel

        #if curr_loss < min_loss:
        #    min_loss = curr_loss
        if curr_loss_pixel < min_loss_pixel:
            min_loss = curr_loss
            min_loss_pixel = curr_loss_pixel

            min_diff_bicubic_loss = diff_bicubic_loss
            min_diff_bicubic_loss_pixel = diff_bicubic_loss_pixel

            best_a = a_candi

            pred_best_a = pred_a_candi

    return best_a, pred_best_a

