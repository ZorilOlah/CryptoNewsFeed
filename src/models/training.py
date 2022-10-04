import torch
from datasets import load_metric
from tqdm.auto import tqdm 


class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader,
                 eval_dataloader, 
                 num_epochs, 
                 optimizer, 
                 lr_scheduler,  
                 device):
        self.model = model
        self.epochs = num_epochs
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler
        self.progress = tqdm(range(self.epochs*len(train_dataloader)))
        self.device = device
        self.training_outputs = {key: {} for key in [f'epoch_{n}' for n in range (self.epochs)]}
        self.eval_outputs = {'epoch_1' : {}}
     
    def train_model(self) -> None:
        self.model.to(self.device)
        self.model.train()
        labels_list, preds_list = [], []
        for epoch in range(self.epochs):
            for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.train_dataloader):
                outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs.loss
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress.update(1)

                self.training_outputs[f'epoch_{epoch}'][f'batch_{batch_idx}'] = {'outputs' : outputs, "labels" : labels}

    def eval_model(self):
        self.model.eval()
        for batch_idx, [input_ids, token_type_ids, attention_mask, labels] in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            self.eval_outputs['epoch_1'][f'batch_{batch_idx}'] = {'outputs' : outputs, 'labels' : labels}


    def test_model(self, model, dataloder, device):
        return
    
    def save_model(self, 
                   save_directory : str):
        self.model.save_pretrained(save_directory)
