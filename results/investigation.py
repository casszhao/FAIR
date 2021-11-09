import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)

fileName = 'scibert_500len_8b_20e_binary_results.csv'
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

# print(plt_list)
print(len(plt_list))
print(len(plt_list[1]))
temp_df = pd.DataFrame(plt_list).transpose()
temp_df.columns = list_of_label

for i, label in enumerate(list_of_label):

    sns.displot(
      data=temp_df,
      x=temp_df[label],
      kind="hist",
      aspect=1.4,
      log_scale=10
    )
    plt.show()
# ax = temp_df.plot.hist(bins=12, alpha=0.5)
# sns.displot(data=temp_df, col='Days', col_wrap=4, x='Visitors')


# temp_df.plot(kind='hist',
#         alpha=0.7,
#         bins=30,
#         title='Histogram Of Test Scores',
#         rot=45,
#         grid=True,
#         figsize=(12,8),
#         fontsize=15,
#         )
# plt.xlabel('Test Score')
# plt.ylabel("Number Of Students")

# print(plt_list)
# fig, axs = plt.subplots(3, 3)
# for i, label in enumerate(list_of_label):
#     if i < 3: # 0, 1, 2,
#         axs_x = 0
#         axa_y = 0+i
#     elif i < 6: # 3,4,5
#         axs_x = 1
#         axa_y = i-3
#     else: # 6,7,8
#         axs_x = 2
#         axa_y = i-6
#
#     print(axs_x, axa_y)
#     print(len(plt_list[i]))
#     axs[axs_x, axa_y].plot(plt_list[i])
#     axs[axs_x, axa_y].set_title(list_of_label[i])
# fig.tight_layout()
# plt.show()


binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
print(binary_f1_score_micro, binary_f1_score_macro)
