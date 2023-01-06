import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from sklearn.utils.class_weight import compute_class_weight 
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler

class LightBertFinance(pl.LightningModule):
    def __init__(self, 
                 model,
                 lr,
                 total_dataset,
                 tokenizer,
                 ):
        super().__init__()
        self.model = model
        self.metric = MulticlassF1Score(num_classes=3, average = None) #Check different average settings - determines reduction over labels
        self.lr = lr
        self.class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(total_dataset.labels), y = total_dataset.labels)
        self.criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weight, dtype = torch.float32))
        self.tz = tokenizer 
       
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        # print(f'\nTraining  Input ids : {input_ids}')
        # print(f'Training  at mask {attention_mask}')
        # print(f'Training  tti {token_type_ids}')
        # print(f'\nTraining text : {[self.tokens_to_string(ids) for ids in input_ids]}\n')
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        # print(f'\nlogits in train step : {outputs.logits}\n')
        # loss = outputs.loss
        # print(f'In training step : {outputs.logits}')
        loss = self.criterion(input = outputs.logits, target =  labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        # print(f'\nInput ids : {input_ids}')
        # print(f'at mask {attention_mask}')
        # print(f'tti {token_type_ids}')
        # print(f'\nVal text : {self.tokens_to_string(input_ids[0])}\n')
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        # print(f'\nlogits in val step : {outputs}\n')
        # print(outputs['logits'])
        # print(f'\n val loss : {outputs.loss}\n')
        loss = self.criterion(input = outputs.logits, target =  labels)
        self.log('val_loss', loss)
        return outputs.logits
    
    def validation_epoch_end(self, outputs) -> None:
        print(f'in end : {outputs}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def tokens_to_string(self, ids) -> str:
        tokens = self.tz.convert_ids_to_tokens(ids)
        strings = self.tz.convert_tokens_to_string(tokens)
        return strings