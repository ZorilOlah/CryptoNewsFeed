import torch
from typing import List
from datasets import load_metric
from tqdm.auto import tqdm 
from transformers import BertTokenizer, BertForSequenceClassification
import itertools
import wandb

class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader,
                 eval_dataloader, 
                 parameters,
                 device):
        self.model = model
        self.parameter_dict = parameters
        self.epochs = parameters['num_epochs']
        self.optimizer = parameters['optimizer']
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = parameters['lr_scheduler']
        self.progress = tqdm(range(self.epochs*len(train_dataloader)))
        self.device = device
        self.training_outputs = {key: {} for key in [f'epoch_{n}' for n in range (self.epochs)]}
        self.eval_outputs = {key: {} for key in [f'epoch_{n}' for n in range (self.epochs)]}
        
    def train_model(self) -> None:
        wandb.init(project = "CNF Sentiment Analysis", config=self.parameter_dict)
        wandb.watch(self.model)
        for epoch in range(self.epochs):
            self.train_epoch_model(epoch = epoch)
            self.eval_model(epoch = epoch)
     
    def train_epoch_model(self, epoch) -> None:
        self.model.to(self.device)
        self.model.train()
        for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.train_dataloader):
            outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()
            wandb.log({"train_loss" : loss})
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress.update(1) 
            self.training_outputs[f'epoch_{epoch}'][f'batch_{batch_idx}'] = {'outputs' : outputs, "labels" : labels}

    def eval_model(self, epoch):
        self.model.eval()
        for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            wandb.log({"eval_loss" : loss})
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            self.eval_outputs[f'epoch_{epoch}'][f'batch_{batch_idx}'] = {'outputs' : outputs, 'labels' : labels}


    def test_model(self, model, dataloder, device):
        return
    
    def save_model(self, 
                   save_directory : str):
        self.model.save_pretrained(save_directory)
        
