

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


OFFSET = 448
THRESHOLD = 50


BBOX_LIST1 = [[445, 117, 493, 175], [472, 438, 523, 487], [473, 403, 508, 448], [451, 351, 481, 391], [430, 267, 476, 307]]
BBOX_LIST2 = [[448, 116, 492, 179], [471, 433, 526, 482], [452, 352, 480, 387], [434, 263, 472, 307]]


def get_centers(obj_list, offset):
    obj_cen_list = []
    for obj in obj_list:
        x1 = obj[0]
        y1 = obj[1]
        x2 = obj[2]
        y2 = obj[3]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y1) / 2
        w = (x2 - x1)
        h = (y2 - y1)
        obj_cen_list.append([x_c - offset, y_c, w, h])
    return obj_cen_list


def visualize(list1, list2):
    ROI = cv2.imread('91315_leica_at2_40x.svs.98728.44031.1649.892.jpg')
    cv2.rectangle(ROI, (0, 0), (512, 512), (255, 0, 0), 3)
    cv2.rectangle(ROI, (448, 0), (960, 512), (0, 0, 255), 3)
    d = 3
    for obj in list1:
        cv2.circle(ROI,
                   (OFFSET + obj[0], obj[1]),
                   d * 2, (0, 255, 0), -1)
    cv2.imwrite('another.png', ROI)


def main():

    tile1_coord = get_centers(BBOX_LIST1, OFFSET)
    tile2_coord = get_centers(BBOX_LIST2, OFFSET)
    print tile1_coord
    print tile2_coord
    tile1_coord = np.asarray(tile1_coord)
    tile2_coord = np.asarray(tile2_coord)

    visualize(tile1_coord, None)

    # dist_matrix = distance.cdist(tile1_coord[:, :2], tile2_coord[:, :2])
    # print dist_matrix
    # row_ind, col_ind = linear_sum_assignment(np.asarray(dist_matrix, np.float32))
    # print row_ind
    # print col_ind

    # unique_objs = [x for i, x in enumerate(tile1_coord) if i not in row_ind]
    # unique_objs += [x for i, x in enumerate(tile2_coord) if i not in col_ind]
    # print unique_objs

    # tile1_mapped_objs = tile1_coord[row_ind]
    # tile2_mapped_objs = tile2_coord[col_ind]

    # # ++++++++++++++++++++++ Finding width & heigh of each mapped bbox +++++++++++++++++++
    # tile_mappend_objs_w = np.maximum(tile1_mapped_objs, tile2_mapped_objs)[:, 2]
    # tile_mappend_objs_h = np.maximum(tile1_mapped_objs, tile2_mapped_objs)[:, 3]
    # # ++++++++++++++++++++++ Finding x_c & y_c of each mapped bbox +++++++++++++++++++++++
    # tile_mappend_objs_aveg = (tile1_mapped_objs + tile2_mapped_objs) / 2
    # tile_mappend_objs_xc = tile_mappend_objs_aveg[:, 0]
    # tile_mappend_objs_yc = tile_mappend_objs_aveg[:, 1]

    # ROI = cv2.imread('91315_leica_at2_40x.svs.98728.44031.1649.892.jpg')
    # # +++++++++++++++++++ Draw Unique Bbox +++++++++++++++++++++++++++++++++
    # for obj in unique_objs:
    #     x1 = obj[0] + OFFSET - (obj[2] / 2)
    #     y1 = obj[1] - (obj[3] / 2)
    #     x2 = obj[0] + OFFSET + (obj[2] / 2)
    #     y2 = obj[1] + (obj[3] / 2)
    #     cv2.rectangle(ROI, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # # +++++++++++++++++++ Draw Mapped Bbox +++++++++++++++++++++++++++++++++
    # for i in range(tile_mappend_objs_xc.shape[0]):
    #     x1 = tile_mappend_objs_xc[i] + OFFSET - (tile_mappend_objs_w[i] / 2)
    #     y1 = tile_mappend_objs_yc[i] - (tile_mappend_objs_h[i] / 2)
    #     x2 = tile_mappend_objs_xc[i] + OFFSET + (tile_mappend_objs_w[i] / 2)
    #     y2 = tile_mappend_objs_yc[i] + (tile_mappend_objs_h[i] / 2)
    #     cv2.rectangle(ROI, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # cv2.imwrite('del.png', ROI)


if __name__ == '__main__':
    main()
