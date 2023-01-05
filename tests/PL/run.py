#%%
from typing import List, Callable
from pathlib import Path
import pandas as pd
import torch
import wandb 
from data.dataset import TitlesDataset, make_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm 
from torchmetrics.classification import MulticlassF1Score
from data.utils import three_column_label_transform, select_number_rows
import pytorch_lightning as pl
import traceback
from models.lit_model import LightBertFinance

titles_excel_file_path = str(Path(__file__).parent) + "/data/200_examples.csv"    
data_dir = str(Path(__file__).parent) + "/data"
tokenizer_yiyang = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone') 

model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_loss'},
    'parameters': 
    {
        'train_batch_size': {'values': [4]},
        'val_batch_size' : {'values' : [1]},
        'epochs': {'values': [1]},
        'lr': {'max': 0.1, 'min': 0.0001},
        'wd' : {'max': 0.1, 'min': 0.0001},
        'scheduler_form' : {'values': ['linear']}
     }
}

sweep_id = wandb.sweep(sweep = sweep_configuration, project = 'swipe_testing')

def main ():
    try:
        logger = pl.loggers.WandbLogger()

        trainer = pl.Trainer(
                            accelerator='cpu',
                            devices = "auto",
                            logger = logger,
                            log_every_n_steps=1,
                            max_epochs= wandb.config.epochs)
        
        train_dl, val_dl, test_dl, total_dataset = make_dataset(path = titles_excel_file_path, 
                                                                train_batch_size= wandb.config.train_batch_size, 
                                                                val_batch_size= wandb.config.val_batch_size, 
                                                                tokenizer = tokenizer_yiyang, 
                                                                data_dir=data_dir, 
                                                                target_transforms=three_column_label_transform)

        classifier = LightBertFinance(model = model, lr = wandb.config.lr, total_dataset=total_dataset )

        trainer.fit(classifier, train_dataloaders=train_dl, val_dataloaders = val_dl)
    except:
        traceback.print_exc()

wandb.agent(sweep_id, function = main, count = 1)
