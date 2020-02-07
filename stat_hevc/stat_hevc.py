#!/usr/bin/python
import re
import pandas as pd
import numpy as np

class stat_hevc(object):

    def __init__(self, filename):

        self.parse_filename(filename)
        self.parse_result(filename)

        # generate table from list_frame, list_summary
        self.df_frame = pd.DataFrame(self.list_frame)
        self.df_summary = pd.DataFrame(self.list_summary)

        # make the table minimal. just essential part.
        # add column name
        self.make_poc_minimal()
        self.make_summary_minimal()

        # calculate bitrate from bits and fps

        # gen id : video + qp
        self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp

        self.df_summary['name'] = self.video_name
        self.df_summary['qp'] = self.qp

        # add loop to table
        self.df_frame['loop'] = self.loop_idx
        self.df_summary['loop'] = self.loop_idx

        # add resolution to table (just height)
        self.df_frame['resolution'] = self.resolution
        self.df_summary['resolution'] = self.resolution

    def get_frame_table(self):
        return self.df_frame

    def get_summary_table(self):
        return self.df_summary

    def parse_filename(self, filename):

        path, name = filename.rsplit('/', 1)
        list_numbers = re.findall('down_[0-9]+', path)
        if list_numbers:
            _, loop_idx = list_numbers[0].split('_', 1)
            self.loop_idx = int(loop_idx) + 1
        else:
            self.loop_idx = 1

    def parse_result(self, filename):
        try:
            self.list_frame = []
            self.list_summary = []

            with open(filename) as data:

                lines = data.readlines()
                #for each_line in data:
                for i in range(0, len(lines)):
                    each_line = lines[i]
                    if 'Input          File' in each_line:
                        (text, video_name) = each_line.rsplit('/', 1)
                        (video_name, text) = video_name.split('.', 1)
                        if 'vcnn_down_' in video_name:
                          _, video_name = video_name.split('vcnn_down_', 1)

                        video_name = re.sub('_[0-9]+x[0-9]+', '', video_name)

                        self.video_name = video_name

                    if each_line.startswith("QP"):
                        if re.match('QP[ ][ :.0-9]+', each_line):
                            (text, qp_str) = each_line.split(':', 1)
                            (qp, text) = qp_str.split('.', 1)

                            self.qp = qp.strip()

                    if 'Real     Format' in each_line:
                        list_numbers= re.findall('[.0-9]+', each_line)
                        self.resolution = int(list_numbers[1])
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
                        self.list_frame.append(list_poc_line)

        except IOError as err:
            print('File error'+str(err))

    def make_poc_minimal(self):

        # select essential columns : 0, 1, 3, 5, 6, 7, 8
        self.df_frame = self.df_frame[[0, 1, 3, 5, 6, 7, 8]]

        # add column name
        self.df_frame.columns = ['slice_type', 'frm', 'qp', 'bits', 'psnr_y', 'psnr_u', 'psnr_v']

        # type change
        self.df_frame['bits'] = self.df_frame['bits'].astype(float)
        #self.df_frame['frm'] = self.df_frame['frm'].astype(int)

        # add bitrate
        self.df_frame['bitrate'] = self.df_frame['bits'] * self.fps / 1000

    def make_summary_minimal(self):
        # add column name
        self.df_summary.columns = ['slice_type', 'n_frames', 'bitrate', 'psnr_y', 'psnr_u', 'psnr_v', 'psnr_yuv']

