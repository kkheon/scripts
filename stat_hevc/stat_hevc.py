#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_hevc(object):

    def __init__(self, filename):

        self.parse_result(filename)

        # generate table from list_poc, list_summary
        self.df_poc = pd.DataFrame(self.list_poc)
        self.df_summary = pd.DataFrame(self.list_summary)

        # make the table minimal. just essential part.
        # add column name
        self.make_poc_minimal()
        self.make_summary_minimal()

        # calculate bitrate from bits and fps

        # gen id : video + qp
        # add id to table


    def get_poc(self):
        return self.df_poc

    def get_summary(self):
        return self.df_summary

    def parse_result(self, filename):
        try:
            self.list_poc = []
            self.list_summary = []

            with open(filename) as data:

                lines = data.readlines()
                #for each_line in data:
                for i in range(0, len(lines)):
                    each_line = lines[i]
                    if 'Input          File' in each_line:
                        (text, video_name) = each_line.rsplit('/', 1)
                        (video_name, text) = video_name.split('.', 1)

                        self.video_name = video_name

                    if each_line.startswith("QP"):
                        if re.match('QP[ ][ :.0-9]+', each_line):
                            (text, qp_str) = each_line.split(':', 1)
                            (qp, text) = qp_str.split('.', 1)

                            self.qp = qp

                    if 'Real     Format' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        self.fps = int(list_numbers[2])


                    if ' a ' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        list_summary_line = ['a'] + list_numbers
                        self.list_summary.append(list_summary_line)
                    if ' i ' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        list_summary_line = ['i'] + list_numbers
                        self.list_summary.append(list_summary_line)
                    if ' b ' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        list_summary_line = ['b'] + list_numbers
                        self.list_summary.append(list_summary_line)
                    if ' p ' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        list_summary_line = ['p'] + list_numbers
                        self.list_summary.append(list_summary_line)

                    if 'POC' in each_line:
                        list_slice_type = re.findall('[A-z]+-SLICE', each_line)
                        slice_type, _ = list_slice_type[0].split('-', 1)

                        list_numbers = re.findall('[.0-9]+', each_line)

                        list_poc_line = [slice_type] + list_numbers
                        self.list_poc.append(list_poc_line)

        except IOError as err:
            print('File error'+str(err))

    def make_poc_minimal(self):

        # select essential columns : 0, 1, 3, 5, 6, 7, 8
        self.df_poc = self.df_poc[[0, 1, 3, 5, 6, 7, 8]]

        # add column name
        self.df_poc.columns = ['slice_type', 'poc', 'qp', 'bits', 'psnr_y', 'psnr_u', 'psnr_v']

        # type change
        self.df_poc['bits'] = self.df_poc['bits'].astype(float)

        # add bitrate
        self.df_poc['bitrate'] = self.df_poc['bits'] * self.fps / 1000

    def make_summary_minimal(self):
        # add column name
        self.df_summary.columns = ['slice_type', 'n_frames', 'bitrate', 'psnr_y', 'psnr_u', 'psnr_v', 'psnr_yuv']

