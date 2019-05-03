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
import random
import shutil


TRAIN_TEST_SPLIT = 0.2

# RUN these 2 bash commands before running this script
#           COPY ALL IMG & ANNOT FOR TRAIN
# cp ../nucleus_images/images/* LuminothStyle_Data/JPEGImages/
# cp DST_DIR/XML/* LuminothStyle_Data/Annotations/
#           THIS WILL REMOVE TEST IMG FOLDER


def load_filenames(src_img_path, src_xml_path, dst_path):
    xml_files = os.listdir(src_xml_path)

    n_test = int(len(xml_files) * TRAIN_TEST_SPLIT)
    test_samples = random.sample(population=xml_files, k=n_test)
    print 'Num of Test Samples : ', n_test

    JPEG_folder = os.path.join(dst_path, 'JPEGImages')
    Annotations_folder = os.path.join(dst_path, 'Annotations')
    test_JPEG_folder = os.path.join(dst_path, 'test_JPEGImages')

    # shutil.rmtree(JPEG_folder)
    # shutil.rmtree(Annotations_folder)
    # shutil.rmtree(test_JPEG_folder)
    # if os.path.exists(dst_path):
    #     shutil.rmtree(dst_path)

    # os.makedirs(dst_path)
    # os.makedirs(JPEG_folder)
    # os.makedirs(Annotations_folder)
    # os.makedirs(test_JPEG_folder)

    os.makedirs(dst_path)
    shutil.copytree(src_img_path, JPEG_folder, symlinks=False, ignore=None)
    shutil.copytree(src_xml_path, Annotations_folder,
                    symlinks=False, ignore=None)

    # if not os.path.exists(JPEG_folder):
    #     os.makedirs(JPEG_folder)
    # if not os.path.exists(Annotations_folder):
    #     os.makedirs(Annotations_folder)
    # if not os.path.exists(test_JPEG_folder):
    # os.makedirs(test_JPEG_folder)

    # ============== Copying ONLY test IMGs to test folder =============
    # TEST DATA
    os.makedirs(test_JPEG_folder)
    for test_sample in test_samples:
        full_src_test_img_filename = os.path.join(
            src_img_path, '%s.png' % test_sample[:-4])
        dest_img_filename = os.path.join(
            test_JPEG_folder, '%s.png' % test_sample[:-4])
        # print 'Copying test data >>>>> ', full_src_test_img_filename
        shutil.copy(full_src_test_img_filename, dest_img_filename)

    # ==================================================================
    # TRAIN DATA
    train_JPEGs = os.listdir(JPEG_folder)
    for train_JPEG in train_JPEGs:
        if '%s.xml' % train_JPEG[:-4] in test_samples:
            # print 'Found A test data  :  ', train_JPEG
            os.remove(os.path.join(JPEG_folder, train_JPEG))

    train_Annots = os.listdir(Annotations_folder)
    for train_Annot in train_Annots:
        if train_Annot in test_samples:
            # print 'Found A test data  :  ', train_Annot
            os.remove(os.path.join(Annotations_folder, train_Annot))

    # =========== Verify Test & Train data sample numbers ==============
    veri_JPEGs = os.listdir(JPEG_folder)
    veri_Annots = os.listdir(Annotations_folder)
    veri_test_JPEGs = os.listdir(test_JPEG_folder)
    print 'Train JPEGs # : ', len(veri_JPEGs)
    print 'Train Annots # : ', len(veri_Annots)
    print 'Test JPEGs # : ', len(veri_test_JPEGs)

    # =========== Write train filenames in txt file ====================
    train_record_path = os.path.join(dst_path, 'ImageSets', 'Main')
    if not os.path.exists(train_record_path):
        os.makedirs(train_record_path)
    filename = train_record_path + '/train.txt'
    classfilename = train_record_path + '/objectness_train.txt'

    with open(filename, "w") as file:
        for png_file in JPEG_folder:
            file.write(png_file[:-4] + '\n')

    with open(classfilename, "w") as file:
        for png_file in JPEG_folder:
            file.write(png_file[:-4] + ' 1 \n')


if __name__ == '__main__':
    SRC_XML_DIR = './DST_DIR/XML/'
    SRC_IMG_DIR = '../nucleus_images/images/'
    DST_DIR = './LuminothStyle_Data/'

    # # Remove unlabled images from src
    # xml_files = os.listdir(SRC_XML_DIR)
    # img_files = os.listdir(SRC_IMG_DIR)
    # for img_file in img_files:
    #     if '%s.xml' % img_file[:-4] not in xml_files:
    #         print 'Found unlabled img file : ', img_file
    #         os.remove(os.path.join(SRC_IMG_DIR, img_file))

    # print 'Verifying Img sampels : ', len(os.listdir(SRC_IMG_DIR))

    load_filenames(SRC_IMG_DIR, SRC_XML_DIR, DST_DIR)
