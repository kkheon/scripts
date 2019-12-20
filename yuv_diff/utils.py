"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

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

import re
import pandas as pd

def plot_psnr_diff(imgs, psnrs, img_num, pos_x, pos_y, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr) in enumerate(zip(axes.flatten(), imgs, psnrs)):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        list_img_shape = list(img.shape)
        if len(list_img_shape) >= 3 and list_img_shape[2] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        elif list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            #img *= 255.0
            #img = img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)

            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            #img = img.squeeze().clamp(0, 1).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('ORG')
            elif i == 1:
                ax.set_xlabel('HM (PSNR: %.2fdB)' % psnr)
            elif i == 2:
                ax.set_xlabel('ARCNN (PSNR: %.2fdB)' % psnr)
            elif i == 3:
                ax.set_xlabel('diff (PSNR: %.2fdB)' % psnr)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    ## save figure
    #result_dir = os.path.join(save_dir, 'plot')
    #if not os.path.exists(result_dir):
    #    os.makedirs(result_dir)

    #save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    save_fn = save_dir + '/Test_result_%d_pos_%04dx%04d.png' % (img_num, pos_x, pos_y)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def plot_ssim_diff(imgs, psnrs, ssims, bitrates, xlabels, img_num, pos_x, pos_y, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, ssim, bitrate, xlabel) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims, bitrates, xlabels)):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        list_img_shape = list(img.shape)
        if len(list_img_shape) >= 3 and list_img_shape[2] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        elif list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            #img *= 255.0
            #img = img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)

            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            #img = img.squeeze().clamp(0, 1).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                #ax.set_xlabel('frame: %d' % xlabel)
                ax.set_xlabel('frame: %d\n bit: %d' % (xlabel, bitrate))
            else:
                #ax.set_xlabel('frame: %d (PSNR: %.2fdB)' % (xlabel, psnr))
                #ax.set_xlabel('frame: %d\n (PSNR: %.2fdB, SSIM: %.4f)' % (xlabel, psnr, ssim))
                ax.set_xlabel('frame: %d\n (PSNR: %.2fdB, SSIM: %.4f,\n bit: %d)' % (xlabel, psnr, ssim, bitrate))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    ## save figure
    #result_dir = os.path.join(save_dir, 'plot')
    #if not os.path.exists(result_dir):
    #    os.makedirs(result_dir)

    #save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    save_fn = save_dir + '/Test_result_%d_pos_%04dx%04d.png' % (img_num, pos_x, pos_y)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def plot_framework_diff(imgs, psnrs, bitrates, xlabels, img_num, frm, pos_x, pos_y, save_dir='', is_training=False, show_label=True, show=False, img_name=None):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, bitrate, xlabel) in enumerate(zip(axes.flatten(), imgs, psnrs, bitrates, xlabels)):
        ax.axis('off')
        #ax.set_adjustable('box-forced')
        ax.set_adjustable('box')
        list_img_shape = list(img.shape)
        if len(list_img_shape) >= 3 and list_img_shape[2] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        elif list(img.shape)[0] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            #if i <= 1:
            #    ax.set_xlabel('%s' % xlabel)
            #elif i == 2:
            #    ax.set_xlabel('%s, \n bit: %d' % (xlabel, bitrate))
            #else:
            #    #ax.set_xlabel('%s, PSNR: %.2fdB, SSIM: %.4f,\n bit: %d)' % (xlabel, psnr, ssim, bitrate))
            #    ax.set_xlabel('%s, PSNR: %.2fdB, \n bit: %d)' % (xlabel, psnr, bitrate))

            string_label = '%s' % xlabel
            if psnr != None:
                string_label += ', PSNR: %.2fdB' % psnr
            if bitrate != None:
                string_label += ', bit : %d' % bitrate
            ax.set_xlabel(string_label)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    ## save figure
    #result_dir = os.path.join(save_dir, 'plot')
    #if not os.path.exists(result_dir):
    #    os.makedirs(result_dir)

    #save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    if img_name == None:
        save_fn = save_dir + '/Test_result_img_%d_frm_%d_pos_%04dx%04d.png' % (img_num, frm, pos_x, pos_y)
    else:
        save_fn = save_dir + '/Test_result_img_%s_frm_%d_pos_%04dx%04d.png' % (img_name, frm, pos_x, pos_y)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def plot_diff(imgs, psnrs, xlabels, img_num, frm, pos_x, pos_y, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, xlabel) in enumerate(zip(axes.flatten(), imgs, psnrs, xlabels)):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        list_img_shape = list(img.shape)
        if len(list_img_shape) >= 3 and list_img_shape[2] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        elif list(img.shape)[0] == 3:
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i < 1:
                ax.set_xlabel('%s' % xlabel)
            else:
                #ax.set_xlabel('%s, PSNR: %.2fdB, SSIM: %.4f,\n bit: %d)' % (xlabel, psnr, ssim, bitrate))
                ax.set_xlabel('%s\n PSNR: %.2fdB)' % (xlabel, psnr))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    ## save figure
    #result_dir = os.path.join(save_dir, 'plot')
    #if not os.path.exists(result_dir):
    #    os.makedirs(result_dir)

    #save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    save_fn = save_dir + '/Test_result_img_%d_frm_%d_pos_%04dx%04d.png' % (img_num, frm, pos_x, pos_y)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def parse_dec_bit_lcu(filename):
    # example : <0,0> 47
    try:
        with open(filename) as data:
            list_result = []
            lines = data.readlines()
            for each_line in lines:

                list_numbers = re.findall('[0-9]+', each_line)
                #list_numbers[0] # frame
                #list_numbers[1] # lcu
                #list_numbers[2] # bis

                list_result.append(list_numbers)

            df_result = pd.DataFrame(list_result)
            df_result.columns = ['frame', 'lcu', 'bit']

            # set type
            df_result = df_result.astype(int)

            return df_result

    except IOError as err:
        print('File error'+str(err))

def draw_scatter_plot_from_df(filename, df, x, y):
    # scatter plot
    ax = df.plot(kind='scatter', x=x, y=y, color='Red')
    fig = ax.get_figure()
    fig.savefig(filename + '.png')
