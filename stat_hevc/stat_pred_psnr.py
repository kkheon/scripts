#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_pred_psnr(object):

    def __init__(self, filename):

        self.parse_filename(filename)
        self.parse_psnr(filename)

        # gen id : video + qp
        self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp

        # add id to table
        #self.df_frame['id'] = self.df_frame['name'] + '_QP' + self.df_frame['qp'] + '_frm_' + self.df_frame['frm']
        #self.df_frame['id'] = 'loop_'+ str(self.loop_idx) + '_' + self.df_frame['name'] + '_QP' + self.df_frame['qp'] + '_frm_' + self.df_frame['frm']
        self.df_frame['id'] = 'loop_'+ str(self.loop_idx) + '_' + self.df_frame['name'] + '_QP' + self.df_frame['qp']

    def get_frame_table(self):
        return self.df_frame

    def parse_filename(self, filename):

        path, name = filename.rsplit('/', 1)
        name, _ = name.split('.', 1)

        list_numbers = re.findall('up_[0-9]+', path)
        _, self.loop_idx = list_numbers[0].split('_', 1)

        _, qp = name.rsplit('QP', 1)
        self.qp = qp


    def parse_psnr(self, filename):

        self.list_frame = []

        f = open(filename, "r")
        data = f.readlines()

        for each_line in data:
            if 'label' in each_line:
                list_numbers = re.findall('[.0-9]+', each_line)
                vsr_psnr = list_numbers[-1]

                # filename
                each_label, each_input, _ = each_line.split(',', 2)
                _, filename = each_label.split('/', 1)
                filename, _ = filename.split('.', 1)

                self.list_frame.append([filename, vsr_psnr])

        f.close()

        # generate table from list_frame, list_summary
        self.df_frame = pd.DataFrame(self.list_frame)

        # add column name
        self.df_frame.columns = ['name', 'psnr_y_up']


