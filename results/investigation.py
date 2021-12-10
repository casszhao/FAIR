import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


sns.set(style="darkgrid", font_scale=1.2)

fileName = 'JN_512len_16b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']


# load dataset
# values = data.values

# configure bootstrap
n_iterations = 1000
n_size = int(len(df) * 0.8)
# run bootstrap
stats = list()
for i in range(n_iterations):

    temp_df = df.sample(n_size, replace=True)

    pred_label = df['Prediction']
    target = df['Ground truth']
    probability = df['Probability']

    pred_array = np.array([np.array(xi) for xi in pred_label])
    targets_array = np.array([np.array(xi) for xi in target])
    prob_array = np.array([np.array(xi) for xi in probability])

    # roc = roc_auc_score(targets_array, prob_array)
    # f1_score_micro = f1_score(targets_array, pred_array, average='micro')
    # f1_score_macro = f1_score(targets_array, pred_array, average='macro')
    metrics = precision_score(targets_array, pred_array, average='macro')
    stats.append(metrics)

# plot scores
pyplot.hist(stats, bins=100)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


stop

f1_score_micro = f1_score(targets_array, pred_array, average='micro')
f1_score_macro = f1_score(targets_array, pred_array, average='macro')
roc = roc_auc_score(targets_array, prob_array)
print('roc: ', roc)
print('micro: ', f1_score_micro, 'macro: ', f1_score_macro)


def one_label_f1(label_index):
    label_name = list_of_label[label_index]
    pred_label = pred_array[:, label_index]
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    recall = recall_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    brier = brier_score_loss(true_label, prob)


    return label_name, f1, recall, precision, brier, prob


all_brier = []
print('---------------------')
plt_list = []
for i, label in enumerate(list_of_label):
    label_name, f1, recall, precision, brier, prob = one_label_f1(i)
    print(label_name, '  ', 'F1', f1, 'recall', recall, 'precision', precision)
    all_brier.append(brier)
    plt_list.append(prob)
    print('biere: ', brier)


avg_brier = sum(all_brier)/len(all_brier)
print('avg brier :', avg_brier)



'''
for plot, do not delete
'''
# print(plt_list)
# print(len(plt_list))
# print(len(plt_list[1]))
# temp_df = pd.DataFrame(plt_list).transpose()
# temp_df.columns = list_of_label

# for i, label in enumerate(list_of_label):
#
#     sns.displot(
#       data=temp_df,
#       x=temp_df[label],
#       kind="hist",
#       aspect=1.4,
#       log_scale=10
#     )
#     plt.show()

