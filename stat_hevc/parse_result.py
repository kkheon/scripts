#!/usr/bin/python
import re

def parse_result(filename):
    try:
        list_result = []
        with open(filename) as data:
            list_sr = []
            list_info = []

            lines = data.readlines()
            #for each_line in data:
            for i in range(0, len(lines)):
                each_line = lines[i]
                if 'Input' in each_line:
                    (text, video_name) = each_line.rsplit('/', 1)
                    (video_name, text) = video_name.split('.', 1)
                    list_info.append(video_name)
                if each_line.startswith("QP"):
                    if re.match('QP[ ][ :.0-9]+', each_line):
                        (text, qp_str) = each_line.split(':', 1)
                        (qp, text) = qp_str.split('.', 1)
                        list_info.append(qp)

                if 'Slices' in each_line:
                    (frame_type, text) = each_line.split('Slices', 1)
                    frame_type = frame_type.strip()

                    result_line = lines[i+2]
                    list_numbers= re.findall('[.0-9]+', result_line)
                    if frame_type is 'P':
                        list_result.append(list_info+list_numbers)

        return list_result
    except IOError as err:
        print('File error'+str(err))
