import torch
from typing import List
from datasets import load_metric
from evaluation.utils import calculate_epoch_accuracies, calculate_epoch_f1
from tqdm.auto import tqdm 
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
import itertools
import wandb
from evaluation.evaluation import Results
import numpy as np
from sklearn.utils.class_weight import compute_class_weight 
import numpy as np


class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader,
                 eval_dataloader, 
                 total_dataset,
                 epochs,
                 learning_rate,
                 scheduler_form,
                 weight_decay,
                 device):
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.total_dataset = total_dataset
        
        
        self.class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(self.total_dataset.labels), y = self.total_dataset.labels)
        self.criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weight, dtype = torch.float32))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay= weight_decay)
        self.lr_scheduler = get_scheduler(name = scheduler_form, optimizer = self.optimizer, num_warmup_steps=0, num_training_steps = self.epochs*len(train_dataloader))

        
        self.progress = tqdm(range(self.epochs*len(train_dataloader)))
        self.device = device
        self.training_outputs = {key: {} for key in [f'epoch_{n}' for n in range (self.epochs)]}
        self.eval_outputs = {key: {} for key in [f'epoch_{n}' for n in range (self.epochs)]}
        
    def train_model(self) -> None:
        for epoch in range(self.epochs):
            self.train_epoch_model(epoch = epoch)
            self.eval_model(epoch = epoch)
            val_acc = calculate_epoch_accuracies(output = self.eval_outputs, epoch = epoch)
            train_acc = calculate_epoch_accuracies(output = self.training_outputs, epoch = epoch)
            val_f1 = calculate_epoch_f1(output = self.eval_outputs, epoch = epoch)
            wandb.log({'training_acc' : train_acc, 'eval_acc' : val_acc, 'eval_f1' : val_f1})
     
    def train_epoch_model(self, epoch) -> None:
        self.model.to(self.device)
        self.model.train()
        for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
            loss = self.criterion(input = outputs.logits, target =  labels)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.progress.update(1) 
            self.training_outputs[f'epoch_{epoch}'][f'batch_{batch_idx}'] = {'outputs' : outputs, "labels" : labels}
            wandb.log({"train_loss" : loss})
    def eval_model(self, epoch):
        self.model.eval()
        for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            self.eval_outputs[f'epoch_{epoch}'][f'batch_{batch_idx}'] = {'outputs' : outputs, 'labels' : labels}
            wandb.log({"eval_loss" : loss})
            

    def test_model(self, model, dataloder, device):
        return
    
    def save_model(self, 
                   save_directory : str):
        self.model.save_pretrained(save_directory)
        
    def _epoch_output(self, epoch):
        eval_preds = np.array([])
        eval_preds = [eval_preds.append(batch['labels']) for batch in self.eval_outputs[f'epoch_{epoch}']]
        
        