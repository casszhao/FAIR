import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from ast import literal_eval
import csv


fileName = '500len_16b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']


binary_pred_array = np.array([np.array(xi) for xi in pred_label])
all_targets_array = np.array([np.array(xi) for xi in target])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

def one_label_f1(label_index):
    label_name = list_of_label[label_index]
    pred_label = binary_pred_array[:, label_index]
    true_label = all_targets_array[:, label_index]
    # print(len(true_label))
    # print(true_label)
    # print(len(pred_label))
    # print(pred_label)
    recall = recall_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    return label_name, f1, recall, precision

print('---------------------')
for i, label in enumerate(list_of_label):
    label_name, f1, recall, precision = one_label_f1(i)
    print(label_name, '  ', 'F1', f1, 'recall', recall, 'precision', precision)