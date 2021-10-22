import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

test = False

# define hypeparameters



# laod data
# url = 'https://raw.githubusercontent.com/casszhao/FAIR/main/sources/PROGRESSSample.tsv'
df = pd.read_csv('./sources/ProgressTrainingCombined.tsv', sep='\t',
                 usecols=['PaperTitle', 'Abstract', 'Place','Race','Occupation','Gender','Religion',
                 'Education','Socioeconomic', 'Social','Plus'])
df['text'] = df.PaperTitle + ' ' + df.Abstract
print(df.head())


df['list'] = df[df.columns[2:11]].values.tolist()
LABEL_NUM = len(df['list'][5])
print('label numbers: ', LABEL_NUM)
# bool_list = list(map(bool,int_list))

new_df = df[['text', 'list']].copy()
print(new_df.head())

list_of_label = ['PlaceOfResidence','RaceEthnicity','Occupation','GenderSex','Religion', 'Education','SocioeconomicStatus', 'SocialCapital','Plus']


if test == True:
    MAX_LEN = 20
    EPOCHS = 1
    new_df=new_df.sample(20)
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
else:
    MAX_LEN = 500
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8

LABEL_NUM = 9
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

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


class BERT_multilabel(torch.nn.Module):
    def __init__(self):
        super(BERT_multilabel, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, LABEL_NUM)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = output_1[1]
        # print(output_1) # dropout(): argument 'input' (position 1) must be Tensor, not str
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        return output


model = BERT_multilabel()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def train_multilabel(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train_multilabel(epoch)


# define validating
def validation_multilabel(model):
    model = model
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


multilabel_prod, targets = validation_multilabel(model)
multilabel_pred = [[np.round(float(i)) for i in nested] for nested in multilabel_prod]

test_dataset['multilabel_prod'] = pd.Series(multilabel_prod)
test_dataset['multilabel_pred'] = multilabel_pred

test_dataset.to_csv('./results/multilabel_pred_results.csv')


multilabel_f1_score_micro = metrics.f1_score(targets, multilabel_pred, average='micro')
multilabel_f1_score_macro = metrics.f1_score(targets, multilabel_pred, average='macro')
print(f"multilabel F1 Score (Micro) = {multilabel_f1_score_micro}")
print(f"multilabel F1 Score (Macro) = {multilabel_f1_score_macro}")
