# -*- coding: utf-8 -*-
# @__ramraj__


import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc
import os

MODE = 'Global'  # ['Global', 'Local']


if MODE == 'Local':
    PATH = './Results/Eval_regular/Img-wise-Tables/'
else:
    PATH = './Results/Eval_regular/'


if not os.path.exists(
        os.path.join(os.path.abspath(os.path.join(PATH, '..')), 'Plots')):
    os.makedirs(os.path.join(os.path.abspath(
        os.path.join(PATH, '..')), 'Plots'))


def get_stats_and_plot(filename, do_plot=False):

    df = pd.read_csv(os.path.join(PATH, filename))

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

    # auc_val = auc(recall, precision)
    # print 'AUC score : ', auc_val

    tmp_PR = list(set(zip(precision, recall)))
    dff = pd.DataFrame(tmp_PR, columns=['Precision', 'Recall'])
    dff = dff.sort_values(by=['Recall'])

    auc_val = auc(dff['Recall'], dff['Precision'])
    print 'AUC score : ', auc_val

    FigSize = (9, 9)
    fig = plt.figure(figsize=FigSize)
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

    if MODE == 'Local':
        fig.savefig(os.path.join(os.path.abspath(os.path.join(PATH, '..')),
                                 'Plots',
                                 # 'PRCurve-AUC=%s - %s.svg' %
                                 # 'PRCurve-AUC=%s - %s.png' %
                                 # (auc_val, filename)),
                                 # '{} - PRCurve-AUC={0:.2f}.png'.format(
                                 #     filename[:-4], auc_val)),
                                 '%s - PRCurve-AUC=%.2f.png' % (
                                     filename[:-4], auc_val)),
                    bbox_inches='tight')
    else:
        fig.savefig(os.path.join(os.path.abspath(os.path.join(PATH, '..')),
                                 # 'PRCurve-AUC=%s - %s.svg' %
                                 # 'Overall_PRCurve-AUC=%s - %s.png' %
                                 # (auc_val, filename)),
                                 'Overall_PRCurve-AUC=%.2f.png'.format(auc_val)),
                    bbox_inches='tight')

    return recall, precision


if __name__ == '__main__':

    if MODE == 'Local':
        files = os.listdir(PATH)
        files = [file for file in files if file != '.DS_Store']
    else:
        files = ['mapped_table.csv']

    files.sort()
    # files = files[:-2]
    # files = files[-2:]
    for file in files:
        P, R = get_stats_and_plot(file)
