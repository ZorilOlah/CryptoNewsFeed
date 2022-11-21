
# %%
from typing import List, Callable
from pathlib import Path
import pandas as pd
import torch
import wandb 
from data.dataset import TitlesDataset, make_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm 
from models.training_wandb_swipe import Trainer
from data.utils import three_column_label_transform, select_number_rows
from evaluation.evaluation import Results
import traceback

titles_excel_file_path = str(Path(__file__).parent) + "/data/200_examples.csv"    
data_dir = str(Path(__file__).parent) + "/data"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer_yiyang = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone') 

# %%

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'eval_f1'},
    'parameters': 
    {
        'batch_size': {'values': [4]},
        'epochs': {'values': [5]},
        'lr': {'max': 0.1, 'min': 0.0001},
        'wd' : {'max': 0.1, 'min': 0.0001},
        'scheduler_form' : {'values': ['linear']}
     }
}

sweep_id = wandb.sweep(sweep = sweep_configuration, project = 'swipe_testing')

# %%
def main():
    try: 
        run = wandb.init()
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)

        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        epochs = wandb.config.epochs
        scheduler = wandb.config.scheduler_form
        wd = wandb.config.wd
        
        train_dl, val_dl, test_dl, total_dataset = make_dataset(path = titles_excel_file_path, 
                                                                train_batch_size= batch_size, 
                                                                val_batch_size= batch_size, 
                                                                tokenizer = tokenizer_yiyang, 
                                                                data_dir=data_dir, 
                                                                target_transforms=three_column_label_transform)
            
        trainer = Trainer(model = model, 
                        train_dataloader = train_dl, 
                        eval_dataloader=val_dl, 
                        total_dataset = total_dataset, 
                        learning_rate = lr, 
                        epochs = epochs, 
                        scheduler_form = scheduler, 
                        weight_decay = wd, 
                        device = device)

        trainer.train_model()
        
        training_results = Results(output = trainer.training_outputs)
        eval_results = Results(output = trainer.eval_outputs)
        
        wandb.log({
            'total_training_acc' : training_results.total_acc,
            'total_eval_acc' : eval_results.total_acc,
        })
            # print(f'\nParameters : {configuration}\n')
        print(f'Evaluation Results : {eval_results.total_acc}')
    except:
        traceback.print_exc()

wandb.agent(sweep_id, function = main, count = 1)



# model_name = 'test_model'

# trainer.save_model(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
# eval_results.save_results(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
# %%
