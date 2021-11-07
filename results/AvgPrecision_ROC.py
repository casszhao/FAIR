import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve, ndcg_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


fileName = 'scibert_500len_8b_20e_binary_results.csv'

df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

# pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

# pred_array = np.array([np.array(xi) for xi in pred_label])
prob_array = np.array([np.array(xi) for xi in probability])
targets_array = np.array([np.array(xi) for xi in target])
# print(targets_array)
# print(prob_array)

list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']
Rank = range(len(df))

ndcg_score = ndcg_score(targets_array, prob_array)
# Compute Normalized Discounted Cumulative Gain.
# Sum the true scores ranked in the order induced by the predicted scores,
# after applying a logarithmic discount.
# Then divide by the best possible score (Ideal DCG, obtained for a perfect ranking)
# to obtain a score between 0 and 1.
# This ranking metric yields a high value if true labels are ranked high by y_score.
print('the Higher the Better')
print('the Normalized Discounted Cumulative Gain: ', ndcg_score)


def get_AvgPrecision_ROC_perLabel(label_index, label_name):
    prob_label = prob_array[:, label_index]
    true_label = targets_array[:, label_index]
    AP = average_precision_score(true_label, prob_label)

    label_name = label_name
    print(label_name)
    fpr, tpr, thresholds = roc_curve(true_label, prob_label)
    auc = roc_auc_score(true_label, prob_label)
    print('AUC:', auc, 'Avg Precision:', AP)
    #
    # fname = '../images/roc/' + str(label_name) + '_ROC'
    #
    # pyplot.savefig(fname)

    return auc, AP, fpr, tpr, thresholds

AUC = []
AP = []
fpr = []
tpr = []
thresholds = []

for n, label in enumerate(list_of_label):
    label_index = n
    label_name = label
    temp_AUC, temp_AP, temp_fpr, temp_tpr, temp_thresholds = get_AvgPrecision_ROC_perLabel(label_index, label_name)

    AUC.append(temp_AUC)
    AP.append(temp_AP)
    fpr.append(temp_fpr)
    tpr.append(temp_tpr)
    # thresholds.append(temp_thresholds)
    # plot the roc curve for the model


pyplot.plot(fpr[0], tpr[0], linestyle='--', color='orange', label=list_of_label[0])
pyplot.plot(fpr[1], tpr[1], linestyle='--', color='black', label=list_of_label[1])
pyplot.plot(fpr[2], tpr[2], linestyle='--', color='red', label=list_of_label[2])
pyplot.plot(fpr[3], tpr[3], linestyle='--', color='blue', label=list_of_label[3])
pyplot.plot(fpr[4], tpr[4], linestyle='--', color='purple', label=list_of_label[4])
pyplot.plot(fpr[5], tpr[5], linestyle='--', color='green', label=list_of_label[5])
pyplot.plot(fpr[6], tpr[6], linestyle='--', color='grey', label=list_of_label[6])
pyplot.plot(fpr[7], tpr[7], linestyle='--', color='yellow', label=list_of_label[7])
pyplot.plot(fpr[8], tpr[8], linestyle='--', color='brown', label=list_of_label[8])


# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



AUC_array = np.array(AUC)
AP_array = np.array(AP)
avgAUC = sum(AUC_array)/len(AUC_array)
avgAP = sum(AP_array)/len(AP_array)
print('the avg AUC and AP for the model are: ', avgAUC, avgAP)

