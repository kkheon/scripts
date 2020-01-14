#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_psnr(object):

    def __init__(self, filename):

        self.parse_filename(filename)
        self.parse_psnr(filename)

        # gen id : video + qp
        self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp

    def get_frame_table(self):
        return self.df_frame

    def parse_filename(self, filename):

        path, name = filename.rsplit('/', 1)
        name, _ = name.split('.', 1)
        if 'vcnn_down_' in name:
          _, name = name.split('vcnn_down_', 1)
        else:
          _, name = name.split('rec_', 1)

        self.video_name = name

        if 'up' in name:
            list_numbers = re.findall('up_[0-9]+', path)
        else:
            list_numbers = re.findall('_[0-9]+', path)
        _, self.loop_idx = list_numbers[0].split('_', 1)

        _, qp = path.rsplit('QP', 1)
        self.qp = qp


    def parse_psnr(self, filename):

        self.list_frame = []

        f = open(filename, "r")
        data = f.readlines()

        for each_line in data:
            if 'Frm' in each_line:
                list_numbers = re.findall('[.0-9]+', each_line)
                self.list_frame.append(list_numbers)

        f.close()

        # generate table from list_frame, list_summary
        self.df_frame = pd.DataFrame(self.list_frame)

        # select essential columns
        self.df_frame = self.df_frame[[0, 2, 4]]

        # add column name
        self.df_frame.columns = ['frm', 'psnr_y_up', 'ssim_up']


