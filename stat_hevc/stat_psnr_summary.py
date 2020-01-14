#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_psnr_summary(object):

    def __init__(self, filename, id_type=None):

        self.parse_filename(filename)
        self.parse_psnr(filename)

        # gen id : video + qp
        #self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp
        self.df_frame['epoch'] = self.epoch

    def get_frame_table(self):
        return self.df_frame

    def parse_filename(self, filename):

        path, filename = filename.rsplit('/', 1)
        name, _ = filename.split('.', 1)

        if 'up' in name:
            list_numbers = re.findall('up_[0-9]+', name)
        else:
            list_numbers = re.findall('_[0-9]+', name)
        _, self.loop_idx = list_numbers[0].split('_', 1)

        # QP
        list_qp = re.findall('QP[0-9]+', filename)
        _, qp = list_qp[0].rsplit('QP', 1)
        self.qp = qp

        # epoch
        list_epoch = re.findall('epoch_[0-9]+', filename)
        _, epoch = list_epoch[0].rsplit('_', 1)
        self.epoch = epoch


    def parse_psnr(self, filename):

        self.list_frame = []

        f = open(filename, "r")
        data = f.readlines()

        for each_line in data:
            if 'label' in each_line:
                # split
                each_line_name, each_line_numbers = each_line.split('frame', 1)

                # name
                list_name = re.findall('[_A-z]+.yuv', each_line_name)
                name, _ = list_name[0].split('.', 1)

                # PSNR
                list_numbers = re.findall('[.0-9]+', each_line_numbers)

                #
                list_numbers[0] = str(int(list_numbers[0]))

                self.list_frame.append([name] + list_numbers)

        f.close()

        # generate table from list_frame, list_summary
        self.df_frame = pd.DataFrame(self.list_frame)

        # select essential columns
        self.df_frame = self.df_frame[[0, 1, 2, 4, 3, 5]]

        # add column name
        self.df_frame.columns = ['name', 'frm', 'psnr_y_up_bicubic', 'ssim_up_bicubic', 'psnr_y_up', 'ssim_up']
        # type change
        self.df_frame[['psnr_y_up_bicubic', 'ssim_up_bicubic', 'psnr_y_up', 'ssim_up']] = self.df_frame[['psnr_y_up_bicubic', 'ssim_up_bicubic', 'psnr_y_up', 'ssim_up']].astype(float)
        #self.df_frame['frm'] = self.df_frame['frm'].astype(int)




