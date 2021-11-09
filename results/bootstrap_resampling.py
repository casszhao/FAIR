import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv
from random import choices
import matplotlib.pyplot as plt


fileName = 'scibert_500len_8b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

total = len(pred_label)

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

# binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
# binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
# roc = roc_auc_score(targets_array, prob_array)
# print('micro: ', binary_f1_score_micro, 'macro: ', binary_f1_score_macro, 'roc: ', roc)

def one_label(label_index):
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    pred = pred_array[:, label_index]
    return brier_score_loss(true_label, prob), roc_auc_score(true_label, pred)


all_brier = []
all_auc = []
for i in range(1000):
    idx = np.random.choice(np.arange(total), 300, replace=True)
    pred_sample = pred_array[idx]
    prob_sample = prob_array[idx]
    target_sample = targets_array[idx]

    brier_list_of_9 = []
    auc_of_9 = []
    for i in range(9):
        one_brier, one_auc = one_label(i)
        brier_list_of_9.append(one_brier)
        auc_of_9.append(one_auc)
    temp_brier = sum(brier_list_of_9)/len(brier_list_of_9)
    temp_auc = sum(auc_of_9) / len(auc_of_9)
    all_brier.append(temp_brier)
    all_auc.append(temp_auc)


print(all_brier, all_auc)
