import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification
from torch import cuda
import torch.nn.functional as F
device = 'cuda' if cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--test", default = False, action='store_true')
parser.add_argument("--epoch", "-e", default=3, type=int)
parser.add_argument("--max_len", "-m", default=500, type=int)
parser.add_argument("--learning_rate", "-l", default=1e-05, action = 'store_true')
parser.add_argument("--train_batch_size", "-t", default=16, type=int)
# parser.add_argument("--", "-t", default=16, type=int, action = 'store_true')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate



if args.test == True:

    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[2:11]].values.tolist()
    new_df = df[['text', 'list']].copy()
    new_df = new_df.sample(200)
    results_directory = '../results/binary_pred_results.csv'
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
    results_directory = './results/binary_pred_results.csv'
    VALID_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = args.train_batch_size

LABEL_NUM = 9
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


list_of_label = ['PlaceOfResidence','RaceEthnicity','Occupation','GenderSex','Religion', 'Education','SocioeconomicStatus', 'SocialCapital','Plus']
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
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}, testing batch size: {}, testing batch size: {}".format(train_dataset.shape, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE))
print("TEST Dataset: {}, testing batch size: {}, testing batch size: {}".format(test_dataset.shape, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
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

def validation_binary(epoch,model_name,label_index):
    model = model_name
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

            targets_all = data['targets'].to(device, dtype=torch.float)

            targets_1d = torch.empty(targets_all.size()[0])
            targets_1d[:] = targets_all[:, label_index]  # i
            targets = torch.stack([1 - targets_1d, targets_1d], dim=1).to(device)

            outputs = model(ids, mask, token_type_ids)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())

            logits = F.softmax(outputs.logits, dim=-1)[:,1]
            fin_outputs.extend(logits.cpu().detach().numpy().tolist())
    return fin_outputs, targets_1d.cpu().detach().numpy().tolist()


f1_list = []
for i, label in enumerate(list_of_label):

  model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  model.to(device)
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
  label_index = i

  for epoch in range(EPOCHS):
    train_binary(epoch, label_index, model, optimizer)

  outputs, targets = validation_binary(epoch, model, label_index)

  pred = np.array(outputs) >= 0.5
  pred = pred.astype(int)

  if i ==0:
    binary_pred = pred
    binary_prob = outputs
    all_targets = targets
    # print('binary_pred ',binary_pred)
    # print('binary_prob ',binary_prob)
  else:
    binary_pred = np.concatenate([binary_pred, pred])
    binary_prob = np.concatenate([binary_prob, outputs])
    print('-------before concatenate---------')
    print(all_targets)
    all_targets = np.concatenate([all_targets, targets])
    print('-------after---------')
    print(all_targets)
    # print('binary_pred ',binary_pred)
    # print('binary_prob ',binary_prob)

print(all_targets)
binary_pred = binary_pred.reshape(len(test_dataset),len(list_of_label)).tolist()
binary_prob = binary_prob.reshape(len(test_dataset),len(list_of_label)).tolist()
all_targets = all_targets.reshape(len(test_dataset),len(list_of_label)).tolist()

test_dataset['binary_pred'] = binary_pred
test_dataset['binary_prob'] = binary_prob

binary_f1_score_micro = metrics.f1_score(all_targets, binary_pred, average='micro')
binary_f1_score_macro = metrics.f1_score(all_targets, binary_pred, average='macro')
print(f"binary F1 Score (Micro) = {binary_f1_score_micro}")
print(f"binary F1 Score (Macro) = {binary_f1_score_macro}")
test_dataset.to_csv(results_directory)
