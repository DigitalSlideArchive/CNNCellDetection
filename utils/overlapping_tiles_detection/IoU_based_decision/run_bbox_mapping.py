# -*- coding: utf-8 -*-
# @__ramraj__


import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


OFFSET = 448
TILE_SIZE = 512
THRESHOLD = 0.5
PAD_SIZE = 64
INPUT_JSON = 'detection_objects.json'


def load_json(name='detection.json'):
    preds = json.loads(open(name).read()).get('objects')

    # Create the 2-dim list
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

    # ++++++++++++++++++++++++++ fILTERING ++++++++++++++++++++++++++
    # Renove the cells, which are close to borders within the PAD_SIZE pixels
    pred_filtered_list = []
    for obj in preds:

        x1 = obj['bbox'][0]
        y1 = obj['bbox'][1]
        x2 = obj['bbox'][2]
        y2 = obj['bbox'][3]

        # Get center bbox coordintes
        x_cent = int((x1 + x2) / 2)
        y_cent = int((y1 + y2) / 2)

        # Check if bbox center is inside the valid error measurable region of ROI.
        # if (x_cent >= PAD_SIZE) and (x_cent <= W - PAD_SIZE)\
        #         and (y_cent >= PAD_SIZE) and (y_cent <= H - PAD_SIZE):
        if (x_cent >= OFFSET) and (x_cent <= TILE_SIZE):

            value = [obj['label'],      # object class - objectness
                     x1,                # x1 coordinate
                     y1,                # y1 coordinate
                     x2,                # x2 coordinate
                     y2,                # y2 coordinate
                     obj['prob']]       # Confidence scor of this detected objectness
            pred_filtered_list.append(value)

    print('Total Number of Filtered Prediction Counts per ROI : ',
          len(pred_filtered_list))
    return pred_filtered_list


def compute_box_centers(df):
    df['width'] = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']
    df['xcenter'] = (df['xmax'] + df['xmin']) / 2
    df['ycenter'] = (df['ymax'] + df['ymin']) / 2
    return df


def np_vec_no_jit_iou(bboxes1, bboxes2):
    """
    Fast, vectorized IoU.
    Source: https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d
    """
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


def main(verbose=False):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Loading the annotations inside the overlapping region
    tile1_list = load_json('detection_objects.json')
    tile1_list = pd.DataFrame(tile1_list,
                              columns=['label', 'xmin', 'ymin',
                                       'xmax', 'ymax', 'confidence'])
    tile2_list = tile1_list.copy()
    if verbose:
        print 'Tile 1 List: '
        print tile1_list

    # Addibg Guassian noise to the copied second tile
    noise = np.asarray(np.random.normal(loc=0, scale=5, size=(len(tile1_list), 4)),
                       np.int32)
    tile2_list.iloc[:, 1:5] = tile2_list.iloc[:, 1:5] + noise
    if verbose:
        print 'Tile 2 List:'
        print tile2_list

    # Cast them into Pandas frame
    tile1 = pd.DataFrame(tile1_list,
                         columns=['label', 'xmin', 'ymin',
                                  'xmax', 'ymax', 'confidence'])
    tile2 = pd.DataFrame(tile2_list,
                         columns=['label', 'xmin', 'ymin',
                                  'xmax', 'ymax', 'confidence'])

    # Throwing away some elements from tile 2 to generate non-sqare cost matrix

    tile2 = tile2.iloc[:-2, :]
    print 'New Tile 2 List'
    print tile2

    # get IoU for predictions and labels
    bboxes1 = np.asarray(np.concatenate((
        np.array(tile1["xmin"])[:, None], np.array(tile1["ymin"])[:, None],
        np.array(tile1["xmax"])[:, None], np.array(tile1["ymax"])[:, None]),
        axis=1), np.float32)
    bboxes2 = np.asarray(np.concatenate((
        np.array(tile2["xmin"])[:, None], np.array(tile2["ymin"])[:, None],
        np.array(tile2["xmax"])[:, None], np.array(tile2["ymax"])[:, None]),
        axis=1), np.float32)
    iou = np_vec_no_jit_iou(bboxes1, bboxes2)
    iou = pd.DataFrame(iou, index=list(tile1.index), columns=list(tile2.index))
    if verbose:
        print 'IoU Matrix : '
        print iou

    # only keep predictions which intersect with some ground truth and vice
    # versa
    iou_NoUnmatched = iou.loc[iou.sum(axis=1) > 0, iou.sum(axis=0) > 0]
    print 'Thresold=0 Filtered IoU Matrix'
    print iou_NoUnmatched

    # # ++++++++++++++++++++++++++++ Additional Filtering Using 0.5 ++++++++++

    # # Apply 0.5 Threshold for IoU values
    # filtered_index = np.where(np.any(iou_NoUnmatched > THRESHOLD, axis=1))
    # if verbose:
    #     print 'Remaining Objs Index After Filtering with Actual Threshold'
    #     print filtered_index
    # iou_NoUnmatched = iou_NoUnmatched.loc[filtered_index]
    # print 'Remaining Objs After Filtering with Actual Threshold (0.5)'
    # print iou_NoUnmatched

    # ++++++++++++++++++++++++++++ Applying Hangarian Algorithm ++++++++++++++

    # Use linear sum assignment (Hungarian algorithm) to match tile1
    # and tile2 to each other in a 1:1 mapping
    # IMPORTANT NOTE: the resultant indices are relative to the
    # iou_NoUnmatched dataframe

    # which are NOT NECESSARILY THE SAME as the indices relative to the iou
    # matrix
    row_ind, col_ind = linear_sum_assignment(1 - iou_NoUnmatched.values)
    print 'Mapped Row & Column from tile 1 & tile 2'
    print row_ind
    print col_ind

    # # a = iou_NoUnmatched.iloc[row_ind, col_ind] > 0.5
    # # print a
    # filtered_row_ind = []
    # filtered_col_ind = []
    # for i in range(len(row_ind)):
    #     if iou_NoUnmatched.loc[row_ind[i], col_ind[i]] > 0.5:
    #         filtered_row_ind.append(row_ind[i])
    #         filtered_col_ind.append(col_ind[i])
    # row_ind = filtered_row_ind
    # col_ind = filtered_col_ind

    # print row_ind
    # print col_ind

    # ++++++++++++++ Differentiate Boxes = (unique boxes, mapped boxes) ++++++

    unique_tile1_objs = tile1[~tile1.index.isin(row_ind)].copy()
    # if verbose:
    #     print 'Tile 1 Unmapped objects : '
    #     print unique_tile1_objs
    unique_tile2_objs = tile2[~tile2.index.isin(col_ind)].copy()
    # if verbose:
    #     print 'Tile 2 Unmapped objects : '
    #     print unique_tile2_objs
    unique_objs = pd.concat([unique_tile1_objs, unique_tile2_objs], axis=0)
    if verbose:
        print 'Total Unmapped objects : '
        print unique_objs
    print 'Unmapped Objects : ', len(unique_objs)

    tile1_mapped_objs = tile1.loc[row_ind]
    tile2_mapped_objs = tile2.loc[col_ind]
    print 'Mapped Objects : ', len(tile1_mapped_objs)

    # ++++++++++++++++++++++ Finding width & heigh of each mapped bbox +++++++

    tile1_mapped_objs = compute_box_centers(tile1_mapped_objs)
    tile2_mapped_objs = compute_box_centers(tile2_mapped_objs)
    # if verbose:
    #     print tile1_mapped_objs
    #     print tile2_mapped_objs
    tile_mappend_objs_w = np.maximum(tile1_mapped_objs['width'],
                                     tile2_mapped_objs['width'])
    tile_mappend_objs_h = np.maximum(tile1_mapped_objs['height'],
                                     tile2_mapped_objs['height'])
    # ++++++++++++++++++++++ Finding x_c & y_c of each mapped bbox +++++++++++
    tile_mappend_objs_xc = (tile1_mapped_objs['xcenter'] +
                            tile2_mapped_objs['xcenter']) / 2
    tile_mappend_objs_yc = (tile1_mapped_objs['ycenter'] +
                            tile2_mapped_objs['ycenter']) / 2

    ROI = cv2.imread('91315_leica_at2_40x.svs.98728.44031.1649.892.jpg')
    # +++++++++++++++++++ Draw Unique Bbox +++++++++++++++++++++++++++++++++
    for i in unique_objs.index:
        x1 = unique_objs.loc[i, 'xmin']
        y1 = unique_objs.loc[i, 'ymin']
        x2 = unique_objs.loc[i, 'xmax']
        y2 = unique_objs.loc[i, 'ymax']
        cv2.rectangle(ROI, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # +++++++++++++++++++ Draw Mapped Bbox +++++++++++++++++++++++++++++++++
    for i in range(tile_mappend_objs_xc.shape[0]):
        x1 = tile_mappend_objs_xc[i] - (tile_mappend_objs_w[i] / 2)
        y1 = tile_mappend_objs_yc[i] - (tile_mappend_objs_h[i] / 2)
        x2 = tile_mappend_objs_xc[i] + (tile_mappend_objs_w[i] / 2)
        y2 = tile_mappend_objs_yc[i] + (tile_mappend_objs_h[i] / 2)
        cv2.rectangle(ROI, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 1)

    cv2.rectangle(ROI, (0, 0), (512, 512), (255, 0, 0), 2)
    cv2.rectangle(ROI, (448, 0), (960, 512), (0, 0, 255), 2)

    cv2.imwrite('final_ROI_with_output.png', ROI)


if __name__ == '__main__':
    main()
