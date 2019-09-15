import os
import torch
import torch.utils.data as data
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

def _parse( file, only_supporting=False):
        data, story = [], []
        for line in file:
            tid, text = line.rstrip('\n').split(' ', 1)
            if tid == '1':
                story = []
            if text.endswith('.'):
                story.append(text[:])
            else:
                query, answer, supporting = (x.strip() for x in text.split('\t'))
                if only_supporting:
                    substory = [story[int(i) - 1] for i in supporting.split()]
                else:
                    substory = [x for x in story if x]
                data.append((substory, query[:-1], answer))
                story.append("")
        return data
    
def build_trg_dics(tenK=True, path="tasks_1-20_v1-2", train=True):
    
    if tenK:
        dirname = os.path.join(path, 'en-10k')
    else:
        dirname = os.path.join(path, 'en')

    for (dirpath, dirnames, filenames) in os.walk(dirname):
        filenames = filenames

    if train:
        filenames = [filename for filename in filenames if  "train.txt" in filename]
    else:
        filenames = [filename for filename in filenames if  "test.txt" in filename]

    temp = []
    for filename in filenames:
        f = open(os.path.join(dirname, filename), 'r')
        parsed =_parse(f)
        temp.extend([d[2] for d in parsed])
    temp = set(temp)
    
    trg_word2id = {word:i for i, word in enumerate(temp)}
    trg_id2word = {i:word for i, word in enumerate(temp)}
    return trg_word2id, trg_id2word


class bAbi_Dataset(data.Dataset):
    
    def __init__(self, trg_word2id, tenK=True, path = "tasks_1-20_v1-2", train=True):
        # joint is Default
        
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if tenK:
            dirname = os.path.join(path, 'en-10k')
        else:
            dirname = os.path.join(path, 'en')
            
        for (dirpath, dirnames, filenames) in os.walk(dirname):
            filenames = filenames
         
        if train:
            filenames = [filename for filename in filenames if  "train.txt" in filename]
        else:
            filenames = [filename for filename in filenames if  "test.txt" in filename]
        
        self.src = []
        self.trg = []
        
        for filename in filenames:
            f = open(os.path.join(dirname, filename), 'r')
            parsed = _parse(f)
            self.src.extend([d[:2] for d in parsed])
            self.trg.extend([trg_word2id[d[2]] for d in parsed])
        self.trg = torch.tensor(self.trg)
            
            
    def __getitem__(self, index):
        src_seq = self.src[index]
        trg = self.trg[index]
        src_seq, seg_seq, att_mask, pos_id = self.preprocess_sequence(src_seq)
        
        return src_seq, seg_seq, att_mask, pos_id, trg

    def __len__(self):
        return len(self.trg)
        
    def preprocess_sequence(self, seq):

        text =  ["[CLS]"] + list(seq[0]) + ["[SEP]"] + [seq[1]] + ["[SEP]"]

        tokenized_text = self.tokenizer.tokenize(" ".join(text))
        indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        where_is_sep = indexed_text.index(102) + 1
        segment_ids = [0 ]* (where_is_sep) + [1] * (len(indexed_text)- where_is_sep)
        attention_mask = [1] *len(indexed_text)
        pos_id = [i for i in range(len(indexed_text))]
        
        return torch.tensor(indexed_text), torch.tensor(segment_ids), torch.tensor(attention_mask), torch.tensor(pos_id)
    
    

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), 512).long()
        for i, seq in enumerate(sequences):
            
            
            end = lengths[i]
            if end <= 512:
                padded_seqs[i, :end] = seq[:end]
            else:
                padded_seqs[i] = seq[-512:]

        return padded_seqs
      
    def pos_merge(sequences):
        
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), 512).long()
        for i, seq in enumerate(sequences):
            
            padded_seqs[i] = torch.tensor([i for i in range(512)])

        return padded_seqs
    
    src_seqs, seg_seqs, att_mask, pos_id, trgs = zip(*data)
    src_seqs = merge(src_seqs)
    seg_seqs = merge(seg_seqs)
    att_mask = merge(att_mask)
    pos_id = pos_merge(pos_id)
    trgs = torch.tensor(trgs)
    return src_seqs, seg_seqs, att_mask, pos_id, trgs


import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn as nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(768, 60)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, tokens_tensor, segments_tensors, att_mask, pos_ids):
        x = self.bert(tokens_tensor, token_type_ids=segments_tensors, position_ids= pos_ids, attention_mask=att_mask)[1]
        x = self.out(x)
        return self.softmax(x)

from pytorch_transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:" + str(torch.cuda.get_device_name(0)))
    
my_model = model()
my_model.to(device)

optimizer = AdamW(my_model.parameters())
criterion = nn.NLLLoss()

from tqdm import tqdm

EPOCHS = 10
for epoch in range(1, EPOCHS+1):
    
    my_model.train()
    
    train_loss = 0
    length = 0
    for tokens_tensor, segments_tensors, att_mask, pos_id, trg in data_loader:
        output = my_model(tokens_tensor.to(device), segments_tensors.to(device), att_mask.to(device), pos_id.to(device))
        loss = criterion(output, trg.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        length+=1
        train_loss += loss.item()
        if length % 10 == 0:
          print("\t\t{:3}/25000 : {}".format(length, train_loss / length))
        
    epoch_loss = train_loss / length
    print("##################")
    print("{} epoch Loss : {:.4f}".format(epoch, epoch_loss))
  