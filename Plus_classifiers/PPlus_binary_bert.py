import argparse
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


device = 'cuda' if cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--test", default = False, action='store_true')
parser.add_argument("--epoch", "-e", default=3, type=int)
parser.add_argument("--max_len", "-m", default=500, type=int)
parser.add_argument("--learning_rate", "-l", default=1e-05, action = 'store_true')
parser.add_argument("--train_batch_size", "-t", default=16, type=int)
parser.add_argument("--bert_model", "-b", default='bert-base-uncased')
# other option allenai/scibert_scivocab_uncased
# parser.add_argument("--", "-t", default=16, type=int, action = 'store_true')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate

test = True


if args.test == True:

    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[2:11]].values.tolist()
    new_df = df[['text', 'list']].copy()
    new_df = new_df.sample(200)
    results_directory = '../results/'
    VALID_BATCH_SIZE = 4
    TRAIN_BATCH_SIZE = 8
    MAX_LEN = 20

else:
    df = pd.read_csv('./sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[2:11]].values.tolist()
    new_df = df[['text', 'list']].copy()
    results_directory = './results/'
    VALID_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = args.train_batch_size

LABEL_NUM = 9
if args.bert_model == 'allenai/scibert_scivocab_uncased':
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
else:
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)




list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']
print(new_df.head())


# define hypeparameters



class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'text': text
        }
# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}, training batch size: {}".format(train_dataset.shape, TRAIN_BATCH_SIZE))
print("TEST Dataset: {}, testing batch size: {}".format(test_dataset.shape, VALID_BATCH_SIZE))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


model = BertForSequenceClassification.from_pretrained(args.bert_model)
model.to(device)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def train_binary(epoch, label_index, model_name, optimizer_name):
    model = model_name
    optimizer = optimizer_name
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets_all = data['targets'].to(device, dtype=torch.float)

        targets_1d =  torch.empty(targets_all.size()[0])
        targets_1d[:] = targets_all[:, label_index] # i
        targets = torch.stack([1-targets_1d, targets_1d], dim = 1).to(device)


        outputs = model(ids, mask, token_type_ids, labels=targets)
        logits = outputs.logits
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: ', epoch, 'Loss: ', loss)

def validation_binary(epoch,model_name,label_index):
    model = model_name
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    text_list = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            text = data['text']
            text_list = text_list + text
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

            targets_all = data['targets'].to(device, dtype=torch.float)

            targets_1d = torch.empty(targets_all.size()[0])
            targets_1d[:] = targets_all[:, label_index]  # i
            fin_targets.extend(targets_1d)
            # targets = torch.stack([1 - targets_1d, targets_1d], dim=1).to(device)
            outputs = model(ids, mask, token_type_ids)
            # fin_targets.extend(targets.cpu().detach().numpy().tolist())
            logits = F.softmax(outputs.logits, dim=-1)[:,1]
            fin_outputs.extend(logits.cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets, text_list


f1_list = []
for i, label in enumerate(list_of_label):
    model = BertForSequenceClassification.from_pretrained(args.bert_model)
    model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    label_index = i

    for epoch in range(EPOCHS):
        train_binary(epoch, label_index, model, optimizer)

    binary_prob, targets, text_list = validation_binary(epoch, model, label_index)
    # print('-----------')
    # print(binary_prob)
    # print(len(binary_prob))
    # print(targets)
    # print(len(targets))
    # print(text_list)
    # print(len(text_list))
    binary_pred = np.array(binary_prob) >= 0.5
    binary_pred = binary_pred.astype(int)
    # print('new binary_prob')
    # print(binary_prob)

    if i ==0:
        binary_pred_array = binary_pred
        binary_prob_array = binary_prob
        all_targets_array = targets
        # print(len(all_targets_array))
        # print(len(binary_prob_array))

            # print('binary_pred ',binary_pred)
            # print('binary_prob ',binary_prob)
    else:
        binary_pred_array = np.concatenate([binary_pred_array, binary_pred])
        binary_prob_array = np.concatenate([binary_prob_array, binary_prob])
        all_targets_array = np.concatenate([all_targets_array, targets])
        # print(len(all_targets_array))
        # print(len(binary_prob_array))

# print(all_targets_array)
binary_pred_list = np.transpose(binary_pred_array.reshape(len(list_of_label), len(test_dataset))).tolist()
binary_prob_list = np.transpose(binary_prob_array.reshape(len(list_of_label), len(test_dataset))).tolist()
all_targets_list = np.transpose(all_targets_array.reshape(len(list_of_label), len(test_dataset))).tolist()

results_df = pd.DataFrame(list(zip(text_list,all_targets_list,binary_pred_list,binary_prob_list)),
                               columns =['Text', 'Ground truth', 'Prediction', 'Probability'])


results_df_name = str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'binary_results.csv'
results_df.to_csv(results_directory + results_df_name)


binary_f1_score_micro = metrics.f1_score(all_targets_array, binary_pred_array, average='micro')
binary_f1_score_macro = metrics.f1_score(all_targets_array, binary_pred_array, average='macro')

binary_pred_array = np.array(binary_pred_list)
all_targets_array = np.array(all_targets_list)


def one_label_f1(label_index):
    label_name = list_of_label[label_index]
    pred_label = binary_pred_array[:, label_index]
    true_label = all_targets_array[:, label_index]
    # print(len(true_label))
    # print(true_label)
    # print(len(pred_label))
    # print(pred_label)
    f1 = f1_score(true_label, pred_label)
    return label_name, f1

print('---------------------')
for i, label in enumerate(list_of_label):
    label_name, f1 = one_label_f1(i)
    print(label_name, '  ', f1)


# usecols list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion',
#            'Education', 'Socioeconomic', 'Social', 'Plus']

print(f"binary F1 Score (Micro) = {binary_f1_score_micro}")
print(f"binary F1 Score (Macro) = {binary_f1_score_macro}")
