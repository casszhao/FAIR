import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from ast import literal_eval
import csv
import matplotlib.pyplot as plt


fileName = '500len_16b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']


Rank = range(len(df))
print('rank', Rank)



def recall_correct_for_per_label(label_index, label_name):
    prob_label = prob_array[:, label_index]
    true_label = targets_array[:, label_index]
    temp_df = pd.DataFrame({'bert_score': prob_label, 'true_label': true_label})
    sorted_df = temp_df.sort_values(by=['bert_score'], ascending=False)
    total = len(pred_array)

    recall_count = []
    correct_count = []
    for i in range(len(sorted_df)):

        # print(' ++++++ rank ', i+1)
        temp_recall_count = sorted_df['true_label'][:i+1].sum()
        recall_count.append(temp_recall_count)

        temp_pred_array_one = [1.0 for i in range(i)]
        temp_pred_array_zero = [0.0 for i in range(total-i)]
        sorted_df['temp_pred'] = temp_pred_array_one + temp_pred_array_zero
        sorted_df['temp'] = sorted_df['true_label'] - sorted_df['temp_pred']
        sorted_df['temp'] = sorted_df['temp'].abs()
        temp_correct_count = total - sorted_df['temp'].sum()
        correct_count.append(temp_correct_count)

    all_counts = [recall_count, correct_count]
    # Plot
    plt.stackplot(Rank, all_counts, labels=['Recall @K','Correct @K'])
    plt.legend(loc='upper left')
    #Adding the aesthetics
    title = str(label_name) + '- Correct prediction and recall counts at K'
    plt.title(title)
    plt.xlabel('At K Ranking')
    plt.ylabel('Correct_prediction / Recall counts')
    # Show the plot
    plt.show()

for n, label in enumerate(list_of_label):
    label_index = n
    label_name = label
    recall_correct_for_per_label(label_index, label_name)

