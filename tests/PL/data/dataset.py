# %%
from pathlib import Path
from typing import Tuple, Callable
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
import os

class TitlesDataset(Dataset):
    def __init__(self, 
                 annotations_file_path : str, 
                 data_dir : str, 
                 transform : Callable = None, 
                 target_transform : Callable = None):
        
        self.annotations = pd.read_csv(annotations_file_path)
        self.titles = self.annotations['Article Title']
        self.label_columns = self.annotations[['Positive', 'Neutral', 'Negative']].fillna(0)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform 
        self.labels = [self.target_transform(item) for item in self.label_columns.iloc]
        self.balance = Counter(self.labels)
        
    def __len__(self):
        return len(self.label_columns)
    
    def __getitem__(self, 
                    index : int):
        title = self.titles[index]
        str_labels = self.label_columns.iloc[index]
        if self.transform:
            tokenizer_output = self.transform(title, padding = 'max_length', max_length=100)
            title_input_ids = torch.tensor(tokenizer_output['input_ids'])
            title_token_type_ids = torch.tensor(tokenizer_output['token_type_ids'])
            title_attention_mask = torch.tensor(tokenizer_output['attention_mask'])
        if self.target_transform:
            numerical_labels = self.target_transform(labels_dict = str_labels)
        return title_input_ids, title_token_type_ids, title_attention_mask, numerical_labels

def make_dataset(path : str,
                 tokenizer : BertTokenizer,
                 data_dir : str,
                 target_transforms : Callable, 
                 split_ratio : float = 0.8,
                 train_batch_size: int = 16,
                 val_batch_size: int = 1,
                 test_batch_size: int = 1,
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    titles_dataset = TitlesDataset(annotations_file_path=path, data_dir=data_dir, transform=tokenizer, target_transform=target_transforms)
    
    train_ds, test_ds = torch.utils.data.random_split(titles_dataset, [int(split_ratio * len(titles_dataset)), len(titles_dataset) - int(split_ratio * len(titles_dataset))])
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [int(split_ratio * len(train_ds)), len(train_ds) - int(split_ratio * len(train_ds))])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = train_batch_size, shuffle = True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = val_batch_size,  shuffle = True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = test_batch_size, shuffle = True)
    return [train_dl, val_dl, test_dl, titles_dataset]
