from pytorch_transformers import BertModel
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