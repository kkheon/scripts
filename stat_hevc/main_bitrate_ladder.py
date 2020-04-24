
import pickle

from bitrate_ladder import *


# load
with open('df_total.pickle', 'rb') as f:
    df_total = pickle.load(f)

out_dir = './result_bitrate_ladder'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

build_bitrate_ladder(df_total, out_dir)

