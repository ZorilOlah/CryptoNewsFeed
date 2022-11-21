import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score


class LightBertFinance(pl.LightningModule):
    def __init__(self, 
                 model,
                 lr):
        super().__init__()
        self.model = model
        self.metric = MulticlassF1Score(num_classes=3, average = None) #Check different average settings - determines reduction over labels
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss
        self.log('val_loss', loss)
        
    def validation_epoch_end(self, outputs) -> None:
        self.metric(outputs['preds'], outputs['target'])
        self.log('f1_score', self.metric)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]