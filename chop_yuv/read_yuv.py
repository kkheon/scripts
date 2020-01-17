import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
import scipy.io
import pickle
from skimage.measure import compare_ssim as ssim


def read_yuv420(filename, w, h, n_frame, start_frame=0):

    y_size = int(w*h)
    uv_size = int(w/2*h/2)
    y_sequence = []
    uv_sequence = []

    # file read
    f = open(filename, "rb")

    # start_frame offset
    if start_frame > 0:
        frame_size = y_size + (2*uv_size)
        f.seek(start_frame * frame_size)

    for i in range(0, n_frame):
        plane_y = np.fromfile(f, dtype=np.uint8, count=y_size).reshape(h, w)
        plane_u = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))
        plane_v = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))

        ## normalize
        #plane_y = plane_y / 255.
        #plane_u = plane_u / 255.
        #plane_v = plane_v / 255.

        # reshape
        plane_y = plane_y.reshape((plane_y.shape[0], plane_y.shape[1], 1))
        plane_u = plane_u.reshape((plane_u.shape[0], plane_u.shape[1], 1))
        plane_v = plane_v.reshape((plane_v.shape[0], plane_v.shape[1], 1))

        # make UV of shape [height, width, color_plane]
        uv = np.concatenate((plane_u, plane_v), axis=2)

        # append to list
        y_sequence.append(plane_y)
        uv_sequence.append(uv)

    f.close()

    # Make list to numpy array. With this transform
    y_array = np.asarray(y_sequence) # [n_frames, h, w, 3]
    uv_array = np.asarray(uv_sequence) # [n_frames, h, w, 3]

    return y_array, uv_array



