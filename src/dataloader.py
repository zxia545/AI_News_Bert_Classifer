import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json

# NOTE: Need to change this to the path of the model you want to use
model_path = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_path)

labels = {'abortion': 0, 'the_affordable_care_act': 1, 'stocks_and_bonds': 2, 'music': 3, 'soccer': 4, 'energy_companies': 5, 'golf': 6, 'law_enforcement': 7, 'economy': 8, 'basketball': 9, 'immigration': 10, 'american_football': 11, 'surveillance': 12, 'gay_rights': 13, 'dance': 14, 'hockey': 15, 'international_business': 16, 'environment': 17, 'federal_budget': 18, 'baseball': 19, 'tennis': 20, 'movies': 21, 'tv_show': 22, 'astronomy': 23, 'military': 24, 'gun_control': 25}

class BertClassiferDataset(Dataset):
    def __init__(self, input_json_file_path_list):
        self.labels = []
        self.texts = []

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
                    self.texts.append(coverted_text)
                    self.labels.append(coverted_label)


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

def load_bert_classifer_data(dataset_path_list, num_workers=8, batch_size=4, shuffle=True, **kwargs):
    dataset = BertClassiferDataset(dataset_path_list, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)


