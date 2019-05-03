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


PROJECT_NAME = 'NucleiDetection'
DATABASE_NAME = 'The Nuclie Bounding Box Database-from Mohamed'
AUTHOR = 'ramraj'


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_object(class_name, xmin, ymin, xmax, ymax, objectness):
    """
    Function that will create sub-root of each object labels.
    Each object in each ROI.

    objectness == 1 : means we only have one class - 'objectness'
    objectness == 0 : means we have multi-class
    """

    root = ET.Element('root')
    object = ET.SubElement(root, 'object')

    name = ET.SubElement(object, 'name')
    if objectness == 1:
        # If we are need objectness detection, then 'objectness' would already be
        # set to 1
        name.text = "objectness"
    else:
        # Else the each cell class label "basophil" or "blast" ... would be
        # set.
        name.text = class_name
    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    sub_xmin = ET.SubElement(bndbox, 'xmin')
    sub_xmin.text = str(xmin)
    sub_ymin = ET.SubElement(bndbox, 'ymin')
    sub_ymin.text = str(ymin)
    sub_xmax = ET.SubElement(bndbox, 'xmax')
    sub_xmax.text = str(xmax)
    sub_ymax = ET.SubElement(bndbox, 'ymax')
    sub_ymax.text = str(ymax)

    return root


def custom_XML_write(image_name, height, width, depth,
                     class_names, xmins, xmaxs, ymins, ymaxs, objectness, dst_file):

    top = ET.Element('annotation')
    top.set('version', '1.0')
    # -----------------------------------------
    folder = ET.SubElement(top, 'folder')
    folder.text = PROJECT_NAME
    # -----------------------------------------
    filename = ET.SubElement(top, 'filename')
    filename.text = image_name
    # -----------------------------------------
    source = ET.SubElement(top, 'source')

    database = ET.SubElement(source, 'database')
    database.text = DATABASE_NAME

    annotation = ET.SubElement(source, 'annotation')
    annotation.text = PROJECT_NAME

    image = ET.SubElement(source, 'image')
    image.text = 'None'

    flickrid = ET.SubElement(source, 'flickrid')
    flickrid.text = 'None'

    # -----------------------------------------
    owner = ET.SubElement(top, 'owner')
    owner.text = AUTHOR
    # -----------------------------------------
    size = ET.SubElement(top, 'size')
    sub_width = ET.SubElement(size, 'width')
    sub_width.text = str(width)
    sub_height = ET.SubElement(size, 'height')
    sub_height.text = str(height)
    sub_depth = ET.SubElement(size, 'depth')
    sub_depth.text = str(depth)
    # -----------------------------------------
    segmented = ET.SubElement(top, 'segmented')
    segmented.text = '0'

    # ====================================================
    # Create each object
    # ====================================================
    for e in range(len(class_names)):
        each_object = create_object(class_names[e],
                                    xmins[e], ymins[e], xmaxs[e], ymaxs[e],
                                    objectness)
        top.extend(each_object)

    # -----------------------------------------

    tree = ET.ElementTree(top)

    # tree.write(save_xml, pretty_print=True, xml_declaration=True, encoding="utf-8")
    tree.write(dst_file, pretty_print=False,
               xml_declaration=True, encoding="utf-8")
    # print(prettify(top))


def draw_bbox(XY_coords, dstimgname, srcimgname):
    """
    """

    img = cv2.imread(srcimgname)
    print img.shape
    # img = img[::-1]
    # img = np.transpose(img, axes=(1, 0, 2))
    XY_coords = np.asarray(XY_coords, np.int32)
    for XY_coord in XY_coords:
        cv2.rectangle(img,
                      (XY_coord[0], XY_coord[1]), (XY_coord[2], XY_coord[3]),
                      (255, 0, 0), 1)
        cv2.imwrite(dstimgname, img)


def load_annot_xml(df):
    rmins = df['rmin']
    rmaxs = df['rmax']
    cmins = df['cmin']
    cmaxs = df['cmax']

    xmins = cmins
    ymins = rmins
    xmaxs = cmaxs
    ymaxs = rmaxs

    obj_coords = zip(xmins, ymins, xmaxs, ymaxs)

    return obj_coords


def loop_run(src_path, dst_path):

    files = glob(
        src_path + 'Annotation_images_2019-04-14_20_52_15.831294/bboxes/*.csv')

    for file in files:
        print 'File : ', file
        src_imagename = os.path.join(src_path, 'images',
                                     '%s.png' % file.split('/')[-1][:-4])
        dst_imagename = os.path.join(dst_path, 'IMG_OVRLAY',
                                     '%s.png' % file.split('/')[-1][:-4])
        XMLfilename = os.path.join(dst_path, 'XML',
                                   '%s.xml' % file.split('/')[-1][:-4])
        if not os.path.exists(os.path.join(dst_path, 'IMG_OVRLAY')):
            os.makedirs(os.path.join(dst_path, 'IMG_OVRLAY'))
        if not os.path.exists(os.path.join(dst_path, 'XML')):
            os.makedirs(os.path.join(dst_path, 'XML'))
        print src_imagename
        print dst_imagename
        df = pd.read_csv(file)

        obj_coords = load_annot_xml(df)

        # ==============================================================
        # ================= Writing BBOX Overlay Images ================
        # ==============================================================

        draw_bbox(obj_coords, dst_imagename, srcimgname=src_imagename)

        # ==============================================================
        # ==================== Writing XML Annotations =================
        # ==============================================================

        XY_coords = np.asarray(obj_coords, np.int32)
        width = int(df['fov_xmax'][0]) - int(df['fov_xmin'][0])
        height = int(df['fov_ymax'][0]) - int(df['fov_ymin'][0])
        with open(XMLfilename, "w") as file:

            nuclei_classes = ['objectness'] * len(XY_coords)
            objectness = 1

            custom_XML_write(image_name=src_imagename,
                             height=height, width=width,
                             depth=3,
                             class_names=nuclei_classes,
                             xmins=XY_coords[:, 0], xmaxs=XY_coords[:, 2],
                             ymins=XY_coords[:, 1], ymaxs=XY_coords[:, 3],
                             objectness=objectness,
                             dst_file=XMLfilename)


if __name__ == '__main__':
    SRC_DIR = './src_data/'
    DST_DIR = './pascalvoc_format_data/'

    loop_run(SRC_DIR, DST_DIR)
