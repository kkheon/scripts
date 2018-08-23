#!/usr/bin/python
from parse_result import *

import fnmatch
import os

### os check
#print os.name
import platform
os_info = platform.system()
#platform.release()

if os_info == 'Windows':
    directory_separator = '\\'
else:
    directory_separator = '/'

def get_filename(item):
    #return item[0]
    return item[1][0]

list_total_result = []
with open('out_result.txt','w') as db_file:
    ### make the list of stat files.
    matches = []
    path = '.'
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, 'result_*.txt'):
            matches.append(os.path.join(root, filename))

    ### get data from each file.
    for each_file in matches:
        list_result = parse_result(each_file)
        list_total_result.append((each_file, list_result))

    list_total_result_sorted = sorted(list_total_result, key=get_filename)
    for each_list_result in list_total_result_sorted:
        each_file = each_list_result[0]
        list_result = each_list_result[1]
        print >> db_file, each_file,
        for each_result in list_result:
            for each_data in each_result:
                print >> db_file, each_data,
        print >> db_file, ""

