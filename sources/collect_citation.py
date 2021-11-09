import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertModel, BertConfig, BertTokenizer, BertForSequenceClassification
from torch import cuda
import torch.nn.functional as F
import seaborn as sns



df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                 usecols=['PaperTitle', 'Abstract', 'Citations', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                          'Education', 'Socioeconomic', 'Social', 'Plus'])

text_list = df.PaperTitle + ' ' + df.Abstract
print(len(text_list))
text_list.dropna()
print(len(text_list))
# pd.set_option('display.max_columns', None)
# print(df)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=False)
length = []
for i, text in enumerate(text_list):
    try:
        seq = tokenizer(text)['input_ids']
        seq_len = len(seq)
        # print(seq_len)
        length.append(seq_len)
    except:
        print(text)


# print(length)
print(max(length))
plt.hist(length, bins=20)
plt.xlabel("Token sequence length")
plt.ylabel("Number of Text")
plt.show()