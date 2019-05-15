# -*- coding: utf-8 -*-
# @__ramraj__


import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from statistics import mode
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import argparse


osp = os.path


'''
Structure of the each functions:
|
|-- load PREDICTIONS Json
|
|-- load GROUND TRUTH XML
|
|-- numpy jit IoU calculation
|
|-- main file

'''


def draw_rectangle(draw,  coordinates, color, width=1, fill=0):
    fill = color + (fill,)
    outline = color + (255,)

    for i in range(int(width)):
        coords = [
            coordinates[0] - i,
            coordinates[1] - i,
            coordinates[2] + i,
            coordinates[3] + i,
        ]

        if i == 0:
            draw.rectangle(coords, fill=fill, outline=outline)
        else:
            draw.rectangle(coords, outline=outline)


def load_pred_json(name, JSON_dict, img_size, pad_size=0):

    print 'Entering into Pred Loading ....'

    img_name = '%s.png' % name.split('/')[-1][5:-5]
    W = img_size[0]
    H = img_size[1]

    preds = json.loads(open(name).read()).get('objects')

    pred_list = []
    for obj in preds:
        value = [
            obj['label'],       # object class - objectness
            obj['bbox'][0],     # x1 coordinate
            obj['bbox'][1],     # y1 coordinate
            obj['bbox'][2],     # x2 coordinate
            obj['bbox'][3],     # y2 coordinate
            obj['prob']         # Confidence scor of this detected objectness
        ]
        pred_list.append(value)

    print('Total Number of Un-Filtered Label Counts per ROI : ', len(pred_list))

    # ++++++++++++++++++++++++++ fILTERING ++++++++++++++++++++++++++
    # Renove the cells, which are close to borders within the pad_size pixels
    pred_filtered_list = []
    for obj in preds:

        x1 = obj['bbox'][0]
        y1 = obj['bbox'][1]
        x2 = obj['bbox'][2]
        y2 = obj['bbox'][3]

        x_cent = int((x1 + x2) / 2)
        y_cent = int((y1 + y2) / 2)

        # Check if bbox center is inside the valid error measurable region of
        # ROI.
        if (x_cent >= pad_size) and (x_cent <= W - pad_size)\
                and (y_cent >= pad_size) and (y_cent <= H - pad_size):
            # if True:

            value = [obj['label'],      # object class - objectness
                     x1,                # x1 coordinate
                     y1,                # y1 coordinate
                     x2,                # x2 coordinate
                     y2,                # y2 coordinate
                     obj['prob']]       # Confidence scor of this detected objectness
            pred_filtered_list.append(value)

    print('Total Number of Filtered Prediction Counts per ROI : ',
          len(pred_filtered_list))

    json_coords = np.asarray(np.asarray(pred_filtered_list)[:, 1:5],
                             dtype=np.int32).\
        tolist()
    json_scores = np.asarray(np.asarray(pred_filtered_list)[:, 5],
                             dtype=np.float32).\
        tolist()
    tmp_dict = {}
    tmp_dict['boxes'] = json_coords
    tmp_dict['scores'] = json_scores
    JSON_dict['%s' % img_name] = tmp_dict

    return pred_filtered_list, JSON_dict


def load_gt_xml(name, JSON_dict, pad_size=0):

    print 'Entering into GT Loading ....'

    img_name = '%s.png' % name.split('/')[-1][:-4]

    xml_list = []
    tree = ET.parse(name)
    root = tree.getroot()

    W = int(root.findall('size/width')[0].text)
    H = int(root.findall('size/height')[0].text)
    D = int(root.findall('size/depth')[0].text)

    for member in root.findall('object'):
        value = [member[0].text,            # object class - objectness
                 int(member[4][0].text),    # x1 coordinate
                 int(member[4][1].text),    # y1 coordinate
                 int(member[4][2].text),    # x2 coordinate
                 int(member[4][3].text),    # y2 coordinate
                 1.0                        # object's confidence - ofcourse its 1.0
                 ]
        xml_list.append(value)

    print('Total Number of Un-Filtered Label Counts per ROI : ', len(xml_list))

    size = root.findall('size')[0]
    size = [int(si.text) for si in size]

    # ++++++++++++++++++++++++++ fILTERING ++++++++++++++++++++++++++
    # Renove the cells, which are close to borders within the pad_size pixels
    xml_FILTERED_list = []
    for member in root.findall('object'):
        x1 = int(member[4][0].text)
        y1 = int(member[4][1].text)
        x2 = int(member[4][2].text)
        y2 = int(member[4][3].text)

        x_cent = int((x1 + x2) / 2)
        y_cent = int((y1 + y2) / 2)

        # Check if bbox center is inside the valid error measurable region of
        # ROI.
        if (x_cent >= pad_size) and (x_cent <= W - pad_size)\
                and (y_cent >= pad_size) and (y_cent <= H - pad_size):
            # if True:

            value = [member[0].text,    # object class - objectness
                     x1,                # x1 coordinate
                     y1,                # y1 coordinate
                     x2,                # x2 coordinate
                     y2,                # y2 coordinate
                     1.0]               # object's confidence - ofcourse its 1.0
            xml_FILTERED_list.append(value)

    print('Total Number of Fitlered Label Counts per ROI : ', len(xml_FILTERED_list))
    xml_coords = np.asarray(np.asarray(xml_FILTERED_list)[:, 1:5],
                            dtype=np.int32).\
        tolist()
    JSON_dict['%s' % img_name] = xml_coords

    return xml_FILTERED_list, JSON_dict, size


def np_vec_no_jit_iou(bboxes1, bboxes2):
    """
    Fast, vectorized IoU.
    Source: https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d
    """
    bboxes1 = np.asarray(bboxes1, dtype=np.float32)
    bboxes2 = np.asarray(bboxes2, dtype=np.float32)
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def calculation(json_path, xml_path, final_results_table, collective_table, args):

    image_folder = args["image_path"]
    dst_folder = args["dest_path"]
    threshold = args["threshold"]
    pad_size = args["pad_size"]
    scale = args["scale"]
    fill = args["fill"]

    print('++++++++++++++++++++++++++++++++++++++++++++++')

    image_name = osp.join(image_folder, '%s.png' %
                          osp.basename(json_path)[5:-5])

    img = cv2.imread(image_name)
    image = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(image, 'RGBA')

    json_dict = {}
    labels_list, _, img_size = load_gt_xml(xml_path, json_dict, pad_size)
    preds_list, _ = load_pred_json(json_path, json_dict, img_size, pad_size)

    n_preds = len(preds_list)
    n_labels = len(labels_list)

    labels = pd.DataFrame(labels_list, columns=[
                          'label', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])
    preds = pd.DataFrame(preds_list, columns=[
                         'label', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])

    iou = np_vec_no_jit_iou(
        bboxes1=np.concatenate((
            np.array(preds["xmin"])[:, None], np.array(preds["ymin"])[:, None],
            np.array(preds["xmax"])[:, None], np.array(preds["ymax"])[:, None]),
            axis=1),
        bboxes2=np.concatenate((
            np.array(labels["xmin"])[:, None], np.array(
                labels["ymin"])[:, None],
            np.array(labels["xmax"])[:, None], np.array(labels["ymax"])[:, None]),
            axis=1))
    iou = pd.DataFrame(iou, index=list(preds.index),
                       columns=list(labels.index))

    iou_matchable = iou.loc[iou.sum(axis=1) > 0, iou.sum(axis=0) > 0]

    row_ind, col_ind = linear_sum_assignment(1 - iou_matchable.values)

    threshold_FP = 0

    mapping = pd.DataFrame()
    img_wise_score_table = {}
    img_wise_score_table['PredScore'] = []
    img_wise_score_table['GTLabel'] = []
    for pairidx in range(len(row_ind)):
        tmp_iou_value = iou_matchable.iloc[row_ind[pairidx],
                                           col_ind[pairidx]]

        pred_idx = int(iou_matchable.index[row_ind[pairidx]])

        if (tmp_iou_value < THRESH_IOU):

            threshold_FP += 1

            collective_table['GTLabel'].append(0)
            collective_table['PredScore'].append(preds_list[pred_idx][5])
            img_wise_score_table['PredScore'].append(preds_list[pred_idx][5])
            img_wise_score_table['GTLabel'].append(0)

            continue

        mapping.loc[pairidx, "predidx"] = iou_matchable.index[
            row_ind[pairidx]]
        mapping.loc[pairidx, "labelidx"] = iou_matchable.columns[
            col_ind[pairidx]]
        mapping.loc[pairidx, "Pred Confidence"] = preds_list[pred_idx][5]
        mapping.loc[pairidx, "IoU"] = tmp_iou_value

        collective_table['GTLabel'].append(1)
        collective_table['PredScore'].append(preds_list[pred_idx][5])
        img_wise_score_table['GTLabel'].append(1)
        img_wise_score_table['PredScore'].append(preds_list[pred_idx][5])

        color = (0, 255, 0)
        pred_coord = preds_list[pred_idx][1:5]
        draw_rectangle(
            draw, pred_coord, color, width=round(3 * scale), fill=fill
        )

    # =========================================================================
    TP = mapping.shape[0]
    # =========================================================================

    # =========================================================================
    iou_Unmatched_index = iou[~iou.index.isin(mapping.index)].index
    FP = len(iou.index) - TP
    # =========================================================================

    # =========================================================================
    iou_unmatched_columns = iou.columns[~np.in1d(np.int32(iou.columns),
                                                 np.int32(mapping["labelidx"]))].values
    iou_unmatched_col_coords = labels[['xmin', 'ymin', 'xmax', 'ymax']].\
        loc[iou_unmatched_columns]
    FN = len(iou_unmatched_col_coords)
    # =========================================================================

    P = (np.float32(TP) / (TP + FP))
    R = (np.float32(TP) / (TP + FN))
    F1 = 2 * (P * R) / (P + R)

    # FPs
    iou_unmatched_rows = iou.index[~np.in1d(np.int32(iou.index),
                                            np.int32(mapping["predidx"]))].values
    iou_unmatched_row_coords = preds[['xmin', 'ymin', 'xmax', 'ymax']].\
        loc[iou_unmatched_rows]
    for i in iou_unmatched_row_coords.index.values:
        # tmp_iou_unmatched_row_coords = preds[['xmin', 'ymin', 'xmax', 'ymax']].\
        #     loc[i]
        pred_coord = iou_unmatched_row_coords.loc[i]
        color = (0, 0, 255)

        draw_rectangle(
            draw, pred_coord, color, width=round(3 * scale), fill=fill
        )

        collective_table['GTLabel'].append(0)
        collective_table['PredScore'].append(preds['confidence'].loc[i])
        img_wise_score_table['GTLabel'].append(0)
        img_wise_score_table['PredScore'].append(preds['confidence'].loc[i])

    # FNs
    for j in iou_unmatched_col_coords.index.values:
        collective_table['GTLabel'].append(1)
        collective_table['PredScore'].append(0.0)
        img_wise_score_table['GTLabel'].append(1)
        img_wise_score_table['PredScore'].append(0.0)

        color = (255, 0, 0)
        label_coord = iou_unmatched_col_coords.loc[j]

        draw_rectangle(
            draw, label_coord, color, width=round(3 * scale), fill=fill
        )

    print("TP = %d, FP = %d, FN = %d" % (TP, FP, FN))
    print("P = %.2f, R = %.2f, F1 = %.2f" % (P, R, F1))
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++')

    noGT_FP = FP - threshold_FP

    ROI_name = '%s.png' % xml_path.split('/')[-1][:-4]
    final_results_table['ROI Name'].append(ROI_name)
    final_results_table['TP'].append(TP)
    final_results_table['FP'].append(FP)
    final_results_table['FN'].append(FN)
    final_results_table['noGT_FP'].append(noGT_FP)
    final_results_table['threshold_FP'].append(threshold_FP)
    final_results_table['Precision'].append('%.2f' % P)
    final_results_table['Recall'].append('%.2f' % R)
    final_results_table['F1-Score'].append('%.2f' % F1)

    # =================================================================
    # ++++++++++++++++ Each Image wise Score Table +++++++++++++++++++++++++
    # =================================================================

    df = pd.DataFrame(img_wise_score_table,
                      columns=['PredScore', 'GTLabel'])
    sav_file = os.path.join(
        dst_folder, 'Img-wise-Tables/table-%s.csv' % ROI_name[:-4])
    df.to_csv(sav_file,
              index=True)

    image.save(os.path.join(dst_folder, 'PNGs', 'overlay_%s' % ROI_name))

    # =================================================================
    # =================================================================
    # =================================================================

    return final_results_table, mapping_table


def eval(args):

    gt_path = args["groundtruth_path"]
    pred_path = args["prediction_path"]
    dst_folder = args["dest_path"]

    if not os.path.exists(os.path.join(dst_folder, 'Img-wise-Tables')):
        os.makedirs(os.path.join(dst_folder, 'Img-wise-Tables'))
    if not os.path.exists(os.path.join(dst_folder, 'PNGs')):
        os.makedirs(os.path.join(dst_folder, 'PNGs'))

    xml_files = os.listdir(gt_path)
    json_files = os.listdir(pred_path)
    xml_files = [xml_file for xml_file in xml_files if xml_file != '.DS_Store']
    json_files = [
        json_file for json_file in json_files if json_file != '.DS_Store']

    print 'Total XML files : ', len(xml_files)
    print 'Total JSON files : ', len(json_files)

    final_results_table = {}
    final_results_table['ROI Name'] = []
    final_results_table['TP'] = []
    final_results_table['FP'] = []
    final_results_table['FN'] = []
    final_results_table['noGT_FP'] = []
    final_results_table['threshold_FP'] = []
    final_results_table['Precision'] = []
    final_results_table['Recall'] = []
    final_results_table['F1-Score'] = []

    mapping_table = {}
    mapping_table['GTLabel'] = []
    mapping_table['PredScore'] = []

    for json_file in json_files:
        for xml_file in xml_files:
            if json_file[5:-5] == xml_file[:-4]:

                print 'File : ', json_file

                final_results_table,\
                    mapping_table = calculation(os.path.join(pred_path,
                                                             json_file),
                                                os.path.join(
                                                    gt_path,
                                                    xml_file),
                                                final_results_table,
                                                mapping_table,
                                                args)

            # =================================================================
            # =================== Saving Final Results Table ==================
            # =================================================================

            df = pd.DataFrame(final_results_table,
                              columns=['ROI Name', 'TP', 'FP', 'FN',
                                       'noGT_FP', 'threshold_FP',
                                       'Precision', 'Recall', 'F1-Score'])
            sav_file = os.path.join(dst_folder, 'final_results_stats.csv')
            df.to_csv(sav_file,
                      index=True)

            df = pd.DataFrame(mapping_table,
                              columns=['GTLabel',
                                       'PredScore',
                                       ])
            sav_file = os.path.join(dst_folder, 'mapped_table.csv')
            df.to_csv(sav_file,
                      index=True)

            df = pd.DataFrame(final_results_table,
                              columns=['ROI Name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score'])
            df = df.sort_values(by=['ROI Name'], ascending=True)
            sav_file = dst_folder + 'TP-FP-FN-stats.csv'
            df.to_csv(sav_file,
                      index=True)

            # =================================================================
            # =================================================================
            # =================================================================


if __name__ == '__main__':
    """
    prediction json file should have 'pred_<image_id>.json' file name, and
    ground truth annotations should have '<image_id>.xml' file name.
    """

    parser = argparse.ArgumentParser(
        description='Eval the predictions based on confidence score.')

    parser.add_argument(
        '--gt', '--groundtruth_path', help='ground truth files path in xml format', required=True)
    parser.add_argument(
        '--p', '--prediction_path', help='prediction files path in json format', required=True)
    parser.add_argument(
        '--i', '--image_path', help='input image file paths', required=True)
    parser.add_argument(
        '--d', '--dest_path', help='destination path to save results and overlays', required=True)

    parser.add_argument(
        '--ps', '--pad_size', type=int, default=0, help='pixels left from all 4 borders')
    parser.add_argument(
        '--t', '--threshold', type=float, default=0.1, help='IoU threshold to filterout robust TP detections')
    parser.add_argument(
        '--s', '--scale', type=int, default=1, help='scale at which draw bounding boxes on overlays')
    parser.add_argument(
        '--f', '--fill', type=int, default=0, help='alpha value to fill the region inside bounding box overlays')

    args = vars(parser.parse_args())

    eval(args)
