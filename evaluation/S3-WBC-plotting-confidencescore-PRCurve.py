# -*- coding: utf-8 -*-
# @__ramraj__


import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc
import os

if not os.path.exists('./PerImgPRTables/'):
    os.makedirs('./PerImgPRTables/')

DIR_PATH = 'Results/Eval_regular/Img-wise-Tables/'


def get_PR_values(score_file, auc_dict):

    # Results thresholded at 0.5
    df = pd.read_csv(DIR_PATH + score_file)
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

    auc_val = auc(recall, precision)
    print 'AUC score : ', auc_val

    auc_dict['AUC'].append(auc_val)
    auc_dict['ImageID'].append('%s.png' % score_file[9:-4])

    tmp_PR = list(set(zip(precision, recall)))
    dff = pd.DataFrame(tmp_PR, columns=['Precision', 'Recall'])
    dff = dff.sort_values(by=['Recall'])

    # plt.step(dff['Recall'], dff['Precision'], where='post', color='b')
    # plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    # plt.show()

    return recall, precision, auc_dict


auc_dict = {}
auc_dict['AUC'] = []
auc_dict['ImageID'] = []

score_files = os.listdir(DIR_PATH)
score_files = [
    score_file for score_file in score_files if score_file != '.DS_Store']

all_recalls = pd.DataFrame()

for score_file in score_files:
    csv_dict = {}
    recall, precision, auc_dict = get_PR_values(score_file, auc_dict)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++ Rounding Mechanism ++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Round to 2 decimals
    recall = np.around(recall, decimals=2)
    precision = np.around(precision, decimals=2)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    PR = list(set(zip(precision, recall)))
    dff = pd.DataFrame(PR, columns=['Precision', 'Recall'])
    dff = dff.sort_values(by=['Recall'])
    dff.to_csv('./PerImgPRTables/PRvalues_img_%s.csv' % score_file)
    # plt.step(dff['Recall'], dff['Precision'], where='post', color='b')
    # plt.show()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    all_recalls = pd.concat([all_recalls, dff['Recall']])
    all_recalls.index = pd.RangeIndex(len(all_recalls.index))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    df = pd.DataFrame(auc_dict, columns=['ImageID', 'AUC'])
    df.to_csv('./AUC_values.csv')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++ Concatanating the Results ++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


all_recalls['SampleNumber'] = pd.RangeIndex(len(all_recalls.index))
all_recalls.columns = ['Recall', 'SampleNumber']
all_recalls.set_index('Recall', inplace=True)
all_recalls = all_recalls[~all_recalls.index.duplicated(keep='first')]


whole_PR_Table = pd.DataFrame()
for score_file in score_files:
    df = pd.read_csv('./PerImgPRTables/PRvalues_img_%s.csv' % score_file)

    precision = df['Precision']
    recall = df['Recall']

    tmp_csv = pd.DataFrame({'Recall': recall, 'Precision': precision})
    tmp_csv.set_index('Recall', inplace=True)
    tmp_csv = tmp_csv[~tmp_csv.index.duplicated(keep='first')]

    whole_PR_Table = pd.concat([whole_PR_Table, tmp_csv], axis=1, join='outer')


df = pd.DataFrame(whole_PR_Table)
df.to_csv('WithoutNaNFilling0_0.csv')
df = df.fillna(method='ffill')
# df = df.fillna(method='bfill')
df.to_csv('FilledPRValues0_0.csv')

sample_std = df.std(axis=1)
sample_mean = df.mean(axis=1)


plt.step(recall, precision, where='post', color='b')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++ Plotting Section ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


FigSize = (10, 10)
fig = plt.figure(figsize=FigSize)
ax = fig.add_subplot(111, aspect='equal')
ax.set_aspect('equal')

# lw = 2
lw = 0
x_axis = df.index.values

plt.plot(x_axis, sample_mean, 'b')
plt.fill_between(x_axis, sample_mean - sample_std,
                 sample_mean + sample_std,
                 # alpha=0.2,
                 alpha=0.3,
                 color="navy", lw=lw,
                 step='mid',
                 facecolor='green',
                 interpolate=True,
                 )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.show()
# fig.savefig('auto1111.png', bbox_inches='tight')
fig.savefig('auto11110_0.svg', bbox_inches='tight')
