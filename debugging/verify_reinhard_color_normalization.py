# -*- coding: utf-8 -*-
# @__ramraj__


import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from lxml import etree as ET
from xml.etree import ElementTree as ElementTree
from xml.dom import minidom
import matplotlib.pyplot as plt
import histomicstk as htk
import scipy as sp
import skimage.io
import skimage.transform
import skimage.measure
import skimage.color

MU = [8.63234435, -0.11501964, 0.03868433]
STD = [0.57506023, 0.10403329, 0.01364062]


def preprocess(im_input, im_reference):

    # ++++++++++++++++++++++++ Color Normalization +++++++++++++++++++++++++++
    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(
        im_reference)
    print 'Mean Ref : ', mean_ref
    print 'Stf Ref : ', std_ref

    # im_input = cv2.resize(im_input, (512, 512))
    im_input = skimage.transform.resize(
        im_input, (512, 512), anti_aliasing=True)
    print 'type :', type(im_input)
    print im_input

    # perform reinhard color normalization
    im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input,
                                                             MU, STD,
                                                             # )
                                                             src_mu=mean_ref,
                                                             src_sigma=std_ref)
    cv2.imwrite('img_afterNormalized.png', im_nmzd)

    # ++++++++++++++++++++++++ Color Deconvolution +++++++++++++++++++++++++++

    # # create stain to color map
    # stainColorMap = {
    #     'hematoxylin': [0.65, 0.70, 0.29],
    #     'eosin': [0.07, 0.99, 0.11],
    #     'dab': [0.27, 0.57, 0.78],
    #     'null': [0.0, 0.0, 0.0]
    # }

    # # specify stains of input image
    # stain_1 = 'hematoxylin'   # nuclei stain
    # stain_2 = 'eosin'         # cytoplasm stain
    # stain_3 = 'null'          # set to null of input contains only two stains

    # # create stain matrix
    # W = np.array([stainColorMap[stain_1],
    #               stainColorMap[stain_2],
    #               stainColorMap[stain_3]]).T

    # # perform standard color deconvolution
    # im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd,
    # W).Stains

    # im_nuclei_stain = im_stains[:, :, 0]
    # print 'Dst Img stats : '
    # print im_nuclei_stain.shape
    # print np.unique(im_nuclei_stain)
    # cv2.imwrite('img_Hchannel.png', im_nuclei_stain)
    # ++++++++++++++++++++++++ Image Resizing +++++++++++++++++++++++++++++++

    # return im_nuclei_stain


def main_loop(src_dir, dst_dir):

    # files = ['img_src1.jpg']
    files = ['./src_data/77.0.jpg', './src_data/77.3.jpg']

    for file in files[:1]:
        print 'File : ', file

        im_input = skimage.io.imread(file)[:, :, :3]
        # plt.imshow(im_input)
        # _ = plt.title('Input Image', fontsize=16)
        # plt.show()

        # Load reference image for normalization
        # ref_image_file = ('https://data.kitware.com/api/v1/file/'
        #                   '57718cc28d777f1ecd8a883c/download')  # L1.png
        # im_reference = skimage.io.imread(ref_image_file)[:, :, :3]
        ref_image_file = ('./src_data/77.tif')  # L1.png
        im_reference = cv2.imread(ref_image_file)[:, :, :3]
        cv2.imwrite('img_ref.png', im_reference)
        # plt.imshow(im_reference)
        # _ = plt.title('Input Reference Image', fontsize=16)
        # plt.show()

        preprocess(im_input, im_reference)
        # output_img = preprocess(im_input, im_reference)
        # cv2.imwrite('img_dst.png', output_img)
        # plt.imshow(output_img)
        # _ = plt.title('Output  Image', fontsize=16)
        # plt.show()


if __name__ == '__main__':

    SRC_DIR = './SRC_DATA/warwick_data/Train/rawJPEGImages/'
    DST_DIR = './SRC_DATA/warwick_data/Train/JPEGImages/'

    main_loop(SRC_DIR, DST_DIR)
