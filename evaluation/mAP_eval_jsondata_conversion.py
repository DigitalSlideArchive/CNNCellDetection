# -*- coding: utf-8 -*-
# @__ramraj__


import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse


"""
GT JSON specs
{"img_id": [[ ... ]],
    ...
}


PRED JSON specs

    {"img_id": {"boxes": [[ ... ]],
                "scores": [...]
                },
        ...
    }
"""


def load_pred_json(name, JSON_dict, pad_size):
    """
    This function will load each predicted JSON files.
    Read each predicted object's 4 coordinates and its labels with confidence score/probability to be an object.
    Return them as a 2-dim list.
    """

    print 'Entering into Pred Loading ....'

    # img_name = '%s.png' % name.split('/')[-1][5:-5]
    # W = img_size[0]
    # H = img_size[1]

    _, jsonname = os.path.split(name)
    img_name = '%s.png' % jsonname[5:-5]

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
        # if (x_cent >= pad_size) and (x_cent <= W - pad_size)\
        #         and (y_cent >= pad_size) and (y_cent <= H - pad_size):
        if True:

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
    """
    This function will load each XML ground truth files.
    Read each objects 4 coordinates and their labels on a ROI.
    Return them as a 2-dim list.
    """

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


def convert(args):

    gt_path = args["groundtruth_path"]
    pred_path = args["prediction_path"]
    dest_path = args["dest_path"]
    mode = args["mode"]
    pad_size = args["pad_size"]

    if mode == 'Local':
        if not os.path.exists(os.path.join(dest_path, 'GT')):
            os.makedirs(os.path.join(dest_path, 'GT'))
        if not os.path.exists(os.path.join(dest_path, 'Pred')):
            os.makedirs(os.path.join(dest_path, 'Pred'))

    # ==========================================================================
    # ==================== Ground Truth XML 2 JSON conversion ================
    # ==========================================================================

    gt_files = os.listdir(gt_path)
    gt_files = [gt_file for gt_file in gt_files if gt_file != '.DS_Store']

    if MODE != 'Local':
        json_dict = {}

    for xmlfile in gt_files:
        if MODE == 'Local':
            json_dict = {}

        xmlfilename = os.path.join(gt_path, xmlfile)
        _, json_gt_dict, _ = load_gt_xml(xmlfilename, json_dict, pad_size)

        if MODE == 'Local':
            print json_gt_dict
            with open(os.path.join(dest_path, 'GT', 'formatted-%s.json' % xmlfile[:-4], 'w')) as gt_json:
                gt_json.write(json.dumps(json_gt_dict))

    if MODE != 'Local':
        print json_gt_dict
        with open(os.path.join(dest_path, 'GT', 'collective-formatted-GT.json', 'w')) as gt_json:
            gt_json.write(json.dumps(json_gt_dict))

    # ====================================================================
    # ==================== Predictions JSON 2 JSON conversion ============
    # ====================================================================

    pred_files = os.listdir(pred_path)
    pred_files = [
        pred_file for pred_file in pred_files if pred_file != '.DS_Store']

    if mode != 'Local':
        json_dict = {}
    for jsonfile in pred_files:
        if mode == 'Local':
            json_dict = {}

        jsonfilename = os.path.join(pred_path, jsonfile)
        print(jsonfilename)
        _, json_pred_dict = load_pred_json(jsonfilename, json_dict, pad_size)

        if mode == 'Local':
            print json_pred_dict
            with open(os.path.join(dest_path, 'Pred', 'formatted-%s.json' % jsonfile[5:-5], 'w')) as pred_json:
                pred_json.write(json.dumps(json_pred_dict))

    if mode != 'Local':
        print json_pred_dict
        with open(os.path.join(dest_path, 'Pred', 'collective-formatted-Pred.json', 'w')) as pred_json:
            pred_json.write(json.dumps(json_pred_dict))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Eval the predictions based on confidence score.')

    parser.add_argument(
        '--gt', '--groundtruth_path', help='ground truth files path in xml format', required=True)
    parser.add_argument(
        '--p', '--prediction_path', help='prediction files path in json format', required=True)
    parser.add_argument(
        '--d', '--dest_path', help='destination path to save results and overlays', required=True)

    parser.add_argument(
        '--m', '--mode', default='Global', choices=['Global', 'Local'], help='mode in which the csv files are run, either as individual experiments or single experiment')
    parser.add_argument(
        '--ps', '--pad_size', type=int, default=0, help='pixels left from all 4 borders')

    args = vars(parser.parse_args())

    convert(args)
