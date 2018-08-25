

from stat_hevc import stat_hevc


if __name__ == '__main__':

    list_dir = [
        #'/home/kkheon/VCNN-Tensorflow/data_vsr/val'
    ]


    filename = 'result_scene_53.txt'
    stat_scene_53 = stat_hevc(filename)

    poc_53 = stat_scene_53.get_poc()
    print(poc_53)
    summary_53 = stat_scene_53.get_summary()
    print(summary_53)

