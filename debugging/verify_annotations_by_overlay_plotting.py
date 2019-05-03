# -*- coding: utf-8 -*-
# @__ramraj__


import cv2
import os
import numpy as np
from glob import glob
from lxml import etree as et
from xml.etree.ElementTree import ElementTree


# JPEG_PATH = '../SRC_DATA/LuminothStyle1D/JPEGImages/'
# XML_PATH = '../SRC_DATA/LuminothStyle1D/Annotations/'
# DST_JPEG_PATH = '../SRC_DATA/LuminothStyle1D/OverlayAnnotationJPEGs/'

JPEG_PATH = '../SRC_DATA/LuminothStyle1D/JPEGImages/'
XML_PATH = '../SRC_DATA/LuminothStyle1D/S1CorrectedAnnotations/'
DST_JPEG_PATH = '../SRC_DATA/LuminothStyle1D/S1CorrectedAnnotations_doublecheck/'


if not os.path.exists(DST_JPEG_PATH):
    os.makedirs(DST_JPEG_PATH)


def plot_annotations(img_file, xml_file, slide_index,
                     do_verbose=False, do_plot=True):

    img = cv2.imread(img_file)

    print xml_file
    lxml_root = et.parse(xml_file)

    print 'Total # of bboxes Found : ', len(lxml_root.findall('.//bndbox'))
    for bbox in lxml_root.findall('.//bndbox'):
        e_bbox = ElementTree(bbox)

        x_min = bbox.find('xmin')
        x_max = bbox.find('xmax')
        y_min = bbox.find('ymin')
        y_max = bbox.find('ymax')

        cv2.rectangle(img,
                      (int(x_min.text), int(y_min.text)),
                      (int(x_max.text), int(y_max.text)),
                      (0, 255, 0), 1)

    cv2.imwrite(os.path.join(DST_JPEG_PATH,
                             '%s_%s.png' % (img_file.split('/')[-1][:-4], slide_index)),
                img)


if __name__ == '__main__':

    fulloath_filenames = glob(JPEG_PATH + 'image*.png')
    # fulloath_filenames = glob(JPEG_PATH + 'TCGA*.png')
    fulloath_filenames.sort()
    for slide_i, file in enumerate(fulloath_filenames):
        print file

        xml_filename = XML_PATH + '%s.xml' % file.split('/')[-1][:-4]
        print xml_filename

        plot_annotations(file, xml_filename, slide_i)
