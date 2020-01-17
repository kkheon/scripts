import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
from scipy import ndimage
import numpy as np
from math import log10

from skimage.feature import hog

import re


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

def read_yuv420_to_yuv444(filename, w, h, n_frame):

    y_size = int(w*h)
    uv_size = int(w/2*h/2)

    yuv_sequence = []
    # file read
    f = open(filename, "rb")
    for i in range(0, n_frame):
        plane_y = np.fromfile(f, dtype=np.uint8, count=y_size).reshape(h, w)
        plane_u = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))
        plane_v = np.fromfile(f, dtype=np.uint8, count=uv_size).reshape(int(h/2), int(w/2))

        # 4:2:0 to 4:4:4
        plane_u = plane_u.repeat(2, axis=0).repeat(2, axis=1)
        plane_v = plane_v.repeat(2, axis=0).repeat(2, axis=1)

        # reshape
        plane_y = plane_y.reshape((plane_y.shape[0], plane_y.shape[1], 1))
        plane_u = plane_u.reshape((plane_u.shape[0], plane_u.shape[1], 1))
        plane_v = plane_v.reshape((plane_v.shape[0], plane_v.shape[1], 1))

        # make YUV of shape [height, width, color_plane]
        yuv = np.concatenate((plane_y, plane_u, plane_v), axis=2)

        # append to list
        yuv_sequence.append(yuv)

    f.close()

    # Make list to numpy array. With this transform
    yuv_array = np.asarray(yuv_sequence) # [n_frames, h, w, 3]

    return yuv_array


def list_save_as_yuv(list_yuv, out_filename):
      arr_yuv = np.asarray(list_yuv)
      arr_yuv_1d = arr_yuv.ravel()

      arr_yuv_1d.tofile(out_filename + '.yuv')


def array_save_as_yuv(arr_yuv, out_filename):
    arr_yuv_1d = arr_yuv.ravel()
    arr_yuv_1d.tofile(out_filename + '.yuv')


def merge_yuv(n_frame, y, uv):
    list_merged = []
    for each_frame in range(n_frame):
        merged = np.concatenate([y[each_frame, :, :].ravel(), uv[each_frame, :, :, 0].ravel(), uv[each_frame, :, :, 1].ravel()])
        list_merged.append(merged)

    arr_yuv = np.asarray(list_merged)
    arr_yuv_1d = arr_yuv.ravel()

    return arr_yuv_1d

def calculate_sse(pred, gt):
    diff = pred - gt
    sse = np.sum(diff ** 2)

    return sse

def calculate_sse_frame(n_frame, pred, gt):

    list_sse = []
    for each_frame in range(0, n_frame):
        each_sse = calculate_sse(pred[each_frame, :, :], gt[each_frame, :, :])
        list_sse.append(each_sse.astype(np.int64))

    return list_sse

def PSNR(pred, gt):
    #pred = pred.clamp(0, 1)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt
    #mse = np.mean(diff.numpy() ** 2)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100, mse
    #return 10 * log10(1.0 / mse)
    try:
        psnr = 10 * log10(1.0 / mse)
    except Exception as e:
        print(type(e))
        psnr = 100

    return psnr, mse


