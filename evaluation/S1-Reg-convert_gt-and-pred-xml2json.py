# -*- coding: utf-8 -*-
# @__ramraj__


import json
import os
import xml.etree.ElementTree as ET
import numpy as np


MODE = 'COLLECTIVE'  # ['COLLECTIVE', 'SEPARATE']
PAD_SIZE = 0


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


def load_pred_json(name, JSON_dict):
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
    # Renove the cells, which are close to borders within the PAD_SIZE pixels
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
        # if (x_cent >= PAD_SIZE) and (x_cent <= W - PAD_SIZE)\
        #         and (y_cent >= PAD_SIZE) and (y_cent <= H - PAD_SIZE):
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


def load_gt_xml(name, JSON_dict):
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
    # Renove the cells, which are close to borders within the PAD_SIZE pixels
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
        if (x_cent >= PAD_SIZE) and (x_cent <= W - PAD_SIZE)\
                and (y_cent >= PAD_SIZE) and (y_cent <= H - PAD_SIZE):
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


if __name__ == '__main__':

    if MODE == 'SEPARATE':
        if not os.path.exists('./Results/Reg-Eval/GT'):
            os.makedirs('./Results/Reg-Eval/GT')
        if not os.path.exists('./Results/Reg-Eval/Pred'):
            os.makedirs('./Results/Reg-Eval/Pred')

    # ==========================================================================
    # ==================== Ground Truth XML 2 JSON conversion ================
    # ==========================================================================

    XML_DIR = './test_XMLs/'
    XML_files = os.listdir(XML_DIR)
    XML_files = [XML_file for XML_file in XML_files if XML_file != '.DS_Store']

    if MODE != 'SEPARATE':
        json_dict = {}
    for xmlfile in XML_files:
        if MODE == 'SEPARATE':
            json_dict = {}

        xmlfilename = os.path.join(XML_DIR, xmlfile)
        _, json_gt_dict, _ = load_gt_xml(xmlfilename, json_dict)

        if MODE == 'SEPARATE':
            print json_gt_dict
            with open('./Results/Reg-Eval/GT/formatted-%s.json' % xmlfile[:-4], 'w') as gt_json:
                gt_json.write(json.dumps(json_gt_dict))

    if MODE != 'SEPARATE':
        print json_gt_dict
        # with open('./Results/Reg-Eval/GT/formatted-%s.json' % xmlfile[:-4],
        # 'w') as gt_json:
        with open('./Results/Reg-Eval/collective-formatted-GT.json', 'w') as gt_json:
            gt_json.write(json.dumps(json_gt_dict))

    # # ====================================================================
    # # ==================== Predictions JSON 2 JSON conversion ============
    # # ====================================================================

    JSON_DIR = './outputs/JSON/'
    JSON_files = os.listdir(JSON_DIR)
    JSON_files = [
        JSON_file for JSON_file in JSON_files if JSON_file != '.DS_Store']

    if MODE != 'SEPARATE':
        json_dict = {}
    for jsonfile in JSON_files:
        if MODE == 'SEPARATE':
            json_dict = {}

        jsonfilename = os.path.join(JSON_DIR, jsonfile)
        print(jsonfilename)
        _, json_pred_dict = load_pred_json(jsonfilename, json_dict)

        if MODE == 'SEPARATE':
            print json_pred_dict
            with open('./Results/Reg-Eval/Pred/formatted-%s.json' % jsonfile[5:-5], 'w') as pred_json:
                pred_json.write(json.dumps(json_pred_dict))

    if MODE != 'SEPARATE':
        print json_pred_dict
        # with open('./Results/Reg-Eval/Pred/formatted-%s.json' %
        # jsonfile[5:-5], 'w') as pred_json:
        with open('./Results/Reg-Eval/collective-formatted-Pred.json', 'w') as pred_json:
            pred_json.write(json.dumps(json_pred_dict))

    print len(XML_files)
    print len(JSON_files)
