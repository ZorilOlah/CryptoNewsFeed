import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from sklearn.utils.class_weight import compute_class_weight 
import numpy as np

class LightBertFinance(pl.LightningModule):
    def __init__(self, 
                 model,
                 lr,
                 total_dataset,
                 ):
        super().__init__()
        self.model = model
        self.metric = MulticlassF1Score(num_classes=3, average = None) #Check different average settings - determines reduction over labels
        self.lr = lr
        self.class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(total_dataset.labels), y = total_dataset.labels)
        self.criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weight, dtype = torch.float32))

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        # loss = outputs.loss
        loss = self.criterion(input = outputs.logits, target =  labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        # print(outputs['logits'])
        # print(outputs.logits)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss
    
    # def validation_epoch_end(self, outputs) -> None:
    #     print(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]