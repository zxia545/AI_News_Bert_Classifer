import torch
from torch import nn 
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self,dropout=0.5, model_path= 'bert-base-uncased'):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768,26)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        output = self.relu(output)
        return output