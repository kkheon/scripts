#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_sse(object):

    def __init__(self, filename):

        self.parse_filename(filename)
        self.parse_sse(filename)

        self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp
        self.df_frame['frm_replaced'] = self.frm_replaced
        self.df_frame['frm'] = int(self.frm_replaced)
        self.df_frame['x'] = self.pos_x
        self.df_frame['y'] = self.pos_y

    def get_frame_table(self):
        return self.df_frame

    def parse_filename(self, filename):

        path, name = filename.rsplit('/', 1)
        name, info = name.split('_frm', 1)

        self.video_name = name

        list_numbers = re.findall('[0-9]+', info)

        self.frm_replaced = list_numbers[0]
        self.pos_x = list_numbers[1]
        self.pos_y = list_numbers[2]
        self.qp = list_numbers[3]

    def parse_sse(self, filename):

        self.list_frame = []

        f = open(filename, "r")
        data = f.readlines()

        for each_line in data:
            self.list_frame.append(int(each_line))

        f.close()

        self.list_frame = [self.list_frame[int(self.frm_replaced)]]
        # generate table from list_frame, list_summary
        self.df_frame = pd.DataFrame(self.list_frame)
        self.df_frame.columns = ['sse']


