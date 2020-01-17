
import json
import pandas as pd
import re

class stat_vmaf(object):
    def __init__(self, filename, replaced=False):

        self.parse_filename(filename, replaced)

        self.list_frame = self.read_file(filename)
        self.df_frame = pd.DataFrame(self.list_frame)
        self.df_frame.columns = ['frm', 'VMAF']
        self.df_frame[['VMAF']] = self.df_frame[['VMAF']].astype(float)

        self.df_frame['name'] = self.video_name
        self.df_frame['qp'] = self.qp

        if replaced:
            self.df_frame['frm_replaced'] = self.frm_replaced
            self.df_frame['x'] = self.pos_x
            self.df_frame['y'] = self.pos_y

        ## add loop to table
        #self.df_frame['loop'] = self.loop_idx


    def parse_filename(self, filename, replaced=False):

        # ex) lanczos_2160_to_1080_vmaf/QP22/result_mf_vcnn_up_rec_Campfire_Party_1920x1080_3840x2160.json

        path, qp, name = filename.rsplit('/', 2)
        if replaced:
            name, info = name.split('_frm', 1)
        else:
            name, _ = name.split('.', 1)

        name = re.sub('result_', '', name)
        name = re.sub('rec_', '', name)
        name = re.sub('mf_vcnn_up_', '', name)
        #name = re.sub('_[0-9]+x[0-9]+', '', name)
        self.video_name = name

        if replaced:
            list_numbers = re.findall('[0-9]+', info)

            self.frm_replaced = list_numbers[0]
            self.pos_x = list_numbers[1]
            self.pos_y = list_numbers[2]
            self.qp = list_numbers[3]
        else:
            _, self.qp = qp.split('P', 1)

    def read_file(self, filename):
        list_frame = []
        with open(filename) as json_file:

            json_data = json.load(json_file)
            for each_frame in json_data['frames']:
                frm = int(each_frame['frameNum'])
                vmaf = each_frame['VMAF_score']

                list_frame.append([frm, vmaf])

            #avg_vmaf = json_data['aggregate']['VMAF_score']


        return list_frame


    def get_frame_table(self):
        return self.df_frame
