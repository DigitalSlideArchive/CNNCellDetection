# -*- coding: utf-8 -*-
# @__ramraj__


import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc
import os
import argparse


def get_stats_and_plot(filename, args):

    mode = args["mode"]
    dst_path = args["dst_path"]
    do_plot = args["do_plot"]

    df = pd.read_csv(filename)

    probabilities = df['PredScore']
    classes = df['GTLabel']

    # calculate precision-recall curve
    thresholds = np.sort(probabilities)
    precision = []
    recall = []
    for t in thresholds:
        precision.append(metrics.precision_score(classes, probabilities >= t))
        recall.append(metrics.recall_score(classes, probabilities >= t))

    recall.append(0)
    precision.append(1)

    print[recall[0], precision[0]]
    print[recall[-1], precision[-1]]

    tmp_PR = list(set(zip(precision, recall)))
    dff = pd.DataFrame(tmp_PR, columns=['Precision', 'Recall'])
    dff = dff.sort_values(by=['Recall'])

    auc_val = auc(dff['Recall'], dff['Precision'])
    print 'AUC score : ', auc_val

    fig_size = (9, 9)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_aspect('equal')
    plt.step(dff['Recall'], dff['Precision'], where='post', color='b')
    # plt.step(recall, precision, where='post', color='b')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if do_plot:
        plt.show()

    # =========================================================================
    # ============================ Saving SVG Files ===========================
    # =========================================================================

    if mode == 'Local':
        fig.savefig(os.path.join(dst_path,
                                 'Plots',
                                 '%s - PRCurve-AUC=%.2f.png' % (
                                     filename[:-4], auc_val)),
                    bbox_inches='tight')
    else:
        fig.savefig(os.path.join(dst_path,
                                 'Plots',
                                 'Overall_PRCurve-AUC=%.2f.png'.format(auc_val)),
                    bbox_inches='tight')

    return recall, precision


def run(args):

    mode = args["mode"]
    csv_path = args["csv_path"]
    dst_path = args["dst_path"]

    if mode == 'Local':
        csv_path = os.path.join(csv_path, 'Img-wise-Tables')

    if not os.path.exists(os.path.join(dst_path, 'Plots')):
        os.makedirs(os.path.join(dst_path, 'Plots'))

    if MODE == 'Local':
        files = os.listdir(csv_path)
        files = [os.path.join(csv_path, file)
                 for file in files if file != '.DS_Store']
    else:
        files = [os.path.join(csv_path, 'mapped_table.csv')]

    files.sort()
    for file in files:
        P, R = get_stats_and_plot(file, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Eval the predictions based on confidence score.')

    parser.add_argument(
        '--c', '--csv_path', help='path which has csv files prepared with confidence scores', required=True)
    parser.add_argument(
        '--d', '--dst_path', help='desination path which saves the results', required=True)

    parser.add_argument(
        '--p', '--do_plot', default=False, help='boolean to either plot or not the results')

    parser.add_argument(
        '--m', '--mode', default='Global', choices=['Global', 'Local'], help='mode in which the csv files are run, either as individual experiments or single experiment')

    args = vars(parser.parse_args())

    run(args)
