import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import random

# NOTE: Need to change this to the path of the model you want to use
model_path = '/home/data/zhengyuhu/bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)

labels = {
    'federal_budget': 0,
    'surveillance': 1,
    'the_affordable_care_act': 2,
    'immigration': 3,
    'law_enforcement': 4,
    'gay_rights': 5,
    'gun_control': 6,
    'military': 7,
    'abortion': 8,
    'dance': 9,
    'tv_show': 10,
    'music': 11,
    'movies': 12,
    'stocks_and_bonds': 13,
    'energy_companies': 14,
    'economy': 15,
    'international_business': 16,
    'astronomy': 17,
    'environment': 18,
    'hockey': 19,
    'basketball': 20,
    'tennis': 21,
    'golf': 22,
    'american_football': 23,
    'baseball': 24,
    'soccer': 25
}

class BertClassifierRealDataset(Dataset):
    def __init__(self, input_jsonl_file_path_list, tokenizer, labels, num_items=4000):

        all_texts = []
        all_labels = []

        for input_jsonl_file_path in input_jsonl_file_path_list:
            with open(input_jsonl_file_path, 'r') as file:
                for line in file:
                    json_dict = json.loads(line)
                    context_text = json_dict['text']
                    # Assume the label is provided elsewhere; modify as needed
                    label_name = json_dict['_id']
                    
                    converted_text = tokenizer(context_text, 
                                    padding='max_length', 
                                    max_length=512, 
                                    truncation=True,
                                    return_tensors="pt")
                    converted_label = label_name
                    all_texts.append(converted_text)
                    all_labels.append(converted_label)

        # Shuffle the data
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        all_texts[:], all_labels[:] = zip(*combined)

        # Select the first num_items instances
        self.texts = all_texts[:num_items]
        self.labels = all_labels[:num_items]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassiferDataset(Dataset):
    def __init__(self, input_json_file_path_list, num_items=4000):
        all_texts = []
        all_labels = []

        for input_json_file_path in input_json_file_path_list:
            f = open(input_json_file_path)
            json_dict = json.load(f)
            
            for label_name in json_dict:
                for id in json_dict[label_name]:
                    context_text = json_dict[label_name][id]['context']
                    coverted_text = tokenizer(context_text, 
                                    padding='max_length', 
                                    max_length = 512, 
                                    truncation=True,
                                    return_tensors="pt")
                    coverted_label = labels[label_name]
                    all_texts.append(coverted_text)
                    all_labels.append(coverted_label)
        # Shuffle the data
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        all_texts[:], all_labels[:] = zip(*combined)

        # Select the first num_items instances
        self.texts = all_texts[:num_items]
        self.labels = all_labels[:num_items]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

def load_bert_classifer_data(dataset_path_list, num_workers=16, batch_size=32, shuffle=True, num_item=4000, **kwargs):
    dataset = BertClassiferDataset(dataset_path_list, num_item, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

def load_bert_classifer_real_data(dataset_path_list, num_workers=16, batch_size=32, shuffle=True, num_item=4000, **kwargs):
    dataset = BertClassifierRealDataset(dataset_path_list, tokenizer, labels, num_items=num_item, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
