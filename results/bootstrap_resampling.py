import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv
from random import choices
import matplotlib.pyplot as plt


fileName = '512len_16b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

total = len(pred_label)

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])

auc = roc_auc_score(targets_array,pred_array)


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

# binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
# binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
# roc = roc_auc_score(targets_array, prob_array)
# print('micro: ', binary_f1_score_micro, 'macro: ', binary_f1_score_macro, 'roc: ', roc)

def one_label(targets_array,prob_array,pred_array,label_index):
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    pred = pred_array[:, label_index]
    return brier_score_loss(true_label, prob), roc_auc_score(true_label, pred)

def one_label_brier(targets_array,prob_array,label_index):
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    return brier_score_loss(true_label, prob)

all_brier = []
all_auc = []
for i in range(1000):
    idx = np.random.choice(np.arange(total), total, replace=True)
    pred_sample = pred_array[idx]
    prob_sample = prob_array[idx]
    target_sample = targets_array[idx]


    brier_list_of_9 = []
    auc_of_9 = []
    for label_index_of_9 in range(9):
        try:
            one_brier, one_auc = one_label(targets_array=target_sample,prob_array=prob_sample,pred_array=pred_sample,
                                       label_index=label_index_of_9)
        except:
            one_auc = 0
            one_brier = one_label_brier(targets_array=target_sample,
                                        prob_array=prob_sample,
                                        label_index=label_index_of_9)
        brier_list_of_9.append(one_brier)
        auc_of_9.append(one_auc)

    temp_brier = sum(brier_list_of_9)/len(brier_list_of_9)
    temp_auc = sum(auc_of_9) / len(auc_of_9)
    all_brier.append(temp_brier)
    all_auc.append(temp_auc)


print(all_brier, all_auc)



plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
# Plot Histogram on x
plt.hist(all_auc, bins=50)
plt.gca().set(title='Metrics bootstrapping', ylabel='Frequency')
plt.show()

confidence_level = 0.95

def get_CI(array):
    degrees_freedom = len(array) - 1
    sample_mean = np.mean(array)
    sample_standard_error = scipy.stats.sem(array)
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom,
                                                 sample_mean, sample_standard_error)
    return confidence_interval

auc_CI = get_CI(all_auc)
brier_CI = get_CI(all_brier)
print('AUC', auc)
print('auc ci',auc_CI)
print('brier ci',brier_CI)