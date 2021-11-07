import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv


fileName = '500len_16b_20e_multilabel_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
print('micro: ', binary_f1_score_micro, 'macro: ', binary_f1_score_macro)


roc = roc_auc_score(targets_array, prob_array)
print('roc: ', roc)


def one_label_f1(label_index):
    label_name = list_of_label[label_index]
    pred_label = pred_array[:, label_index]
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    # print(len(true_label))
    # print(true_label)
    # print(len(pred_label))
    # print(pred_label)
    recall = recall_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    brier = brier_score_loss(true_label, prob)
    return label_name, f1, recall, precision, brier

all_brier = []
print('---------------------')
for i, label in enumerate(list_of_label):
    label_name, f1, recall, precision, brier = one_label_f1(i)
    print(label_name, '  ', 'F1', f1, 'recall', recall, 'precision', precision)
    all_brier.append(brier)
    print('biere: ', brier)

avg_brier = sum(all_brier)/len(all_brier)
print('avg brier :', avg_brier)

binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
print(binary_f1_score_micro, binary_f1_score_macro)
