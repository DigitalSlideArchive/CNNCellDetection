# -*- coding: utf-8 -*-
# @__ramraj__


import cv2
import os
import numpy as np
from glob import glob
from lxml import etree as et
from xml.etree.ElementTree import ElementTree


JPEG_PATH = '../SRC_DATA/LuminothStyle1D/JPEGImages/'
XML_PATH = '../SRC_DATA/LuminothStyle1D/Annotations/'

DST_JPEG_PATH = '../SRC_DATA/LuminothStyle1D/S3_main_croptiles_TCGA_JPEGImages/'
DST_XML_PATH = '../SRC_DATA/LuminothStyle1D/S3_main_croptiles_TCGA_Annotations/'


if not os.path.exists(DST_JPEG_PATH):
    os.makedirs(DST_JPEG_PATH)
if not os.path.exists(DST_XML_PATH):
    os.makedirs(DST_XML_PATH)


"""
Primary tile size: 512 x 512 - which should have quite number of nuclies

- but if tile either h or w <=1025 and >= 512 can be left as it is depending on
    its number of nuclie content.
- only image*.png series images have this variable size problem.
- Also TCGS-* series images have constant size 1000x1000, but depending on the
    number of nuclie content, we need to crop them into 512x512 tiles.
- some specific image*.png series
    image01.png - 1286 x 976 - LEAVE IT NOW - not much nuclie


* I don't see any quite more number of nuclie in the image*.png series.
* so go for TCGA-*.png series
"""


def crop_tiles(img_file, xml_file, start_x, end_x, start_y, end_y, slide_index, tile_num,
               do_verbose=False, do_plot=True):

    img = cv2.imread(img_file)

    print xml_file
    lxml_root = et.parse(xml_file)

    print 'Total # of bboxes Found : ', len(lxml_root.findall('.//bndbox'))
    for bbox in lxml_root.findall('.//bndbox'):
        e_bbox = ElementTree(bbox)

        x_min = int(bbox.find('xmin').text)
        x_max = int(bbox.find('xmax').text)
        y_min = int(bbox.find('ymin').text)
        y_max = int(bbox.find('ymax').text)

        if (x_min < start_x) or (y_min < start_y) \
                or (x_min > end_x) or (y_min > end_y):
            if do_verbose:
                print 'Removing : ', x_min, y_min, bbox.getparent()
            bbox.getparent().remove(bbox)
            continue

        if x_max >= end_x:
            """End_X falling outside the cropping region."""
            bbox.find('xmax').text = str(end_x)
            if do_verbose:
                print 'Assigning X : ', x_max, ' : ', bbox.find('xmax').text

        if y_max >= end_y:
            """End_Y falling outside the cropping region."""
            bbox.find('ymax').text = str(end_y)
            if do_verbose:
                print 'Assigning Y : ', y_max, ' : ', bbox.find('ymax').text

        # Remove very thin bndboxes.
        if (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) <= 2 \
                or (int(bbox.find('ymax').text) - int(bbox.find('ymin').text)) <= 2:
            bbox.getparent().remove(bbox)
            if do_verbose:
                print 'Removing **** VERY THIN BNDBOX : ', bbox.getparent()
            continue

        cv2.rectangle(img,
                      (int(bbox.find('xmin').text),
                       int(bbox.find('ymin').text)),
                      (int(bbox.find('xmax').text),
                       int(bbox.find('ymax').text)),
                      (0, 255, 0), 1)
        # ================================================================================
        # === Finally do a 0 shift to index the annotation coordinates for tile 1,2,3 ====
        # ================================================================================
        bbox.find('xmin').text = str(int(bbox.find('xmin').text) - start_x)
        bbox.find('xmax').text = str(int(bbox.find('xmax').text) - start_x)
        bbox.find('ymin').text = str(int(bbox.find('ymin').text) - start_y)
        bbox.find('ymax').text = str(int(bbox.find('ymax').text) - start_y)

    print 'Remaining # of bboxes Found : ', len(lxml_root.findall('.//bndbox'))

    # Saving the sample overlay annotated images
    cv2.imwrite('sample_%s_%s.png' % (slide_index, tile_num), img)
    # Saving the actual cropped image for training. Beaware ! this should be gray image
    dst_img = cv2.imread(img_file, 0)
    dst_img = dst_img[start_x: end_x, start_y: end_y]
    cv2.imwrite(os.path.join(DST_JPEG_PATH,
                             '%s_%s.png' % (img_file.split('/')[-1][:-4], tile_num)),
                dst_img)

    lxml_root.write(os.path.join(DST_XML_PATH,
                                 '%s_%s.xml' % (xml_file.split('/')[-1][:-4], tile_num)))
    """
    if we find any bbox.coord >=512, then delete those whole bndbox entry
    if we find any bbox.coord.x_max >=512 then change its value to 512
    """


if __name__ == '__main__':

    # ====================================================================================
    # ============================ TCGA Type Images ======================================
    # ====================================================================================

    fulloath_filenames = glob(JPEG_PATH + 'TCGA*.png')
    for slide_i, file in enumerate(fulloath_filenames):

        xml_filename = XML_PATH + '%s.xml' % file.split('/')[-1][:-4]
        print xml_filename

        # ++++++++++++++++++++++++++++++ CASE = 1 ++++++++++++++++++++++++++++++
        tile_number = 1
        crop_tiles(file, xml_filename, 0, 512, 0, 512, slide_i, tile_num=tile_number)
        # Checking the xml deleted bnbboxes.
        lxml_root = et.parse(os.path.join(DST_XML_PATH,
                                          '%s_%s.xml' % (xml_filename.split('/')[-1][:-4],
                                                         tile_number)))
        print 'Total # : ', len(lxml_root.findall('.//bndbox'))

        # ++++++++++++++++++++++++++++++ CASE = 2 ++++++++++++++++++++++++++++++
        tile_number = 2
        crop_tiles(file, xml_filename, 0, 512, 512, 1000, slide_i, tile_num=tile_number)
        # Checking the xml deleted bnbboxes.
        lxml_root = et.parse(os.path.join(DST_XML_PATH,
                                          '%s_%s.xml' % (xml_filename.split('/')[-1][:-4],
                                                         tile_number)))
        print 'Total # : ', len(lxml_root.findall('.//bndbox'))

        # ++++++++++++++++++++++++++++++ CASE = 3 ++++++++++++++++++++++++++++++
        tile_number = 3
        crop_tiles(file, xml_filename, 512, 1000, 0, 512, slide_i, tile_num=tile_number)
        # Checking the xml deleted bnbboxes.
        lxml_root = et.parse(os.path.join(DST_XML_PATH,
                                          '%s_%s.xml' % (xml_filename.split('/')[-1][:-4],
                                                         tile_number)))
        print 'Total # : ', len(lxml_root.findall('.//bndbox'))

        # ++++++++++++++++++++++++++++++ CASE = 4 ++++++++++++++++++++++++++++++
        tile_number = 4
        crop_tiles(file, xml_filename, 512, 1000, 512, 1000, slide_i, tile_num=tile_number)
        # Checking the xml deleted bnbboxes.
        lxml_root = et.parse(os.path.join(DST_XML_PATH,
                                          '%s_%s.xml' % (xml_filename.split('/')[-1][:-4],
                                                         tile_number)))
        print 'Total # : ', len(lxml_root.findall('.//bndbox'))
