#!/usr/bin/python3

from yuv_tools import *

def make_block_replacement():

    # set list of files. from label(?)
    DIR_ROOT = "/data/kkheon/dataset/SJTU_4K_test"
    DIR_LABEL = "/data/kkheon/dataset/SJTU_4K_test/label"
    DIR_LABEL_HM = "/data/kkheon/dataset/SJTU_4K_test/label_hm"

    # set benchmark : as QP32
    DIR_LABEL_HM_BENCH = DIR_LABEL_HM + '/QP32'

    out_dir = './dataset_replace'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    list_qp = [
        "27",
        #"32",
        "37",
        "42",
        "47",
    ]

    # set list of test
    list_video = [
        "Campfire_Party",
        "Fountains",
        "Runners",
        "Rush_Hour",
        "Traffic_Flow",
    ]
    w = 3840
    h = 2160
    n_frame = 5

    block_size_h = 64
    block_size_w = 64

    # loop of file
    for each_video in list_video:
        print('video : ' + each_video)

        # read benchmark
        filename_bench = DIR_LABEL_HM_BENCH + '/rec_' + each_video + '.yuv'
        video_bench_y, video_bench_uv = read_yuv420(filename_bench, w, h, n_frame, start_frame=0)

        # read label
        filename_label = DIR_LABEL + '/' + each_video + '.yuv'
        video_label_y, video_label_uv = read_yuv420(filename_label, w, h, n_frame, start_frame=0)

        # calculate SSE of bench
        list_bench_sse_y = calculate_sse_frame(n_frame, video_bench_y, video_label_y)
        list_bench_sse_u = calculate_sse_frame(n_frame, video_bench_uv[:, :, :, 0], video_label_uv[:, :, :, 0])
        list_bench_sse_v = calculate_sse_frame(n_frame, video_bench_uv[:, :, :, 1], video_label_uv[:, :, :, 1])

        n_sample = 5
        list_y_idx = np.random.choice(range(0, h-block_size_h+1, block_size_h), n_sample)
        list_x_idx = np.random.choice(range(0, w-block_size_w+1, block_size_w), n_sample)

        #list_y_idx = [256]
        #list_x_idx = [576]

        # for each frame
        for each_frame in range(n_frame):
            print('\tframe : %d' % each_frame)

            # loop of block
            for y in list_y_idx:
            #for y in range(0, h-block_size_h+1, block_size_h):
                print('\t\ty=%4d' % y)

                for x in list_x_idx:
                #for x in range(0, w-block_size_w+1, block_size_w):

                    y_uv = int(y/2)
                    x_uv = int(x/2)
                    block_size_h_uv = int(block_size_h/2)
                    block_size_w_uv = int(block_size_w/2)

                    # make temporary frame from video bench
                    video_temp_y = video_bench_y.copy()
                    video_temp_uv = video_bench_uv.copy()

                    # loop of qp
                    for each_qp in list_qp:

                        # gen path
                        filename = DIR_LABEL_HM + '/QP' + each_qp + '/rec_' + each_video + '.yuv'

                        # read yuv
                        input_y, input_uv = read_yuv420(filename, w, h, n_frame, start_frame=0)

                        # replace_block
                        video_temp_y[each_frame, y:y + block_size_h, x:x + block_size_w] = input_y[each_frame, y:y + block_size_h, x:x + block_size_w]
                        video_temp_uv[each_frame, y_uv:y_uv + block_size_h_uv, x_uv:x_uv + block_size_w_uv, :] = input_uv[each_frame, y_uv:y_uv + block_size_h_uv, x_uv:x_uv + block_size_w_uv, :]

                        # merge yuv
                        video_temp = merge_yuv(n_frame, video_temp_y, video_temp_uv)

                        # save temporary frame
                        #out_filename = out_dir + '/' + each_video + '_' + str(x) + '_' + str(y) + '_qp' + each_qp
                        out_filename = out_dir + '/' + each_video + '_frm{0:02d}'.format(each_frame) + '_{0:04d}'.format(x) + 'x{0:04d}'.format(y) + '_qp' + each_qp
                        array_save_as_yuv(video_temp, out_filename)

                        # need to save SSE
                        #calculate sse with label

                        # calculate SSE
                        list_temp_sse_y = calculate_sse_frame(n_frame, video_temp_y, video_label_y)
                        list_temp_sse_u = calculate_sse_frame(n_frame, video_temp_uv[:, :, :, 0], video_label_uv[:, :, :, 0])
                        list_temp_sse_v = calculate_sse_frame(n_frame, video_temp_uv[:, :, :, 1], video_label_uv[:, :, :, 1])

                        # calculate SSE diff
                        list_sse_diff_y = np.subtract(list_temp_sse_y, list_bench_sse_y)
                        list_sse_diff_u = np.subtract(list_temp_sse_u, list_bench_sse_u)
                        list_sse_diff_v = np.subtract(list_temp_sse_v, list_bench_sse_v)

                        out_filename = out_dir + '/' + each_video + '_frm{0:02d}'.format(each_frame) + '_{0:04d}'.format(x) + 'x{0:04d}'.format(y) + '_qp' + each_qp + '_sse.txt'
                        with open(out_filename, 'w') as file_handler:
                            for item in list_sse_diff_y:
                                file_handler.write("{}\n".format(item))
                            for item in list_sse_diff_u:
                                file_handler.write("{}\n".format(item))
                            for item in list_sse_diff_v:
                                file_handler.write("{}\n".format(item))


def main():
    make_block_replacement()

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)
