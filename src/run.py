# %%
# %%
from typing import List, Callable
from pathlib import Path
import pandas as pd
import torch
import wandb 
from data.dataset import TitlesDataset, make_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm 
from models.training import Trainer
from data.utils import three_column_label_transform, select_number_rows
from datasets import load_metric
from evaluation.evaluation import Results
from collections import Counter
from models.utils import hyperparameters_from_configurations, configurations_from_dict

titles_excel_file_path = str(Path(__file__).parent) + "/data/50_examples.csv"    
data_dir = str(Path(__file__).parent) + "/data"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer_yiyang = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone') 

# %%

# search_space = {
#     "learning_rate" : [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
#     "weight_decay" : [0.001, 0.0001],
#     "batch_size" : [8, 16, 32],
#     "epochs" : [10],
# }

search_space = {
    "learning_rate" : [1e-5, 2e-5],
    "weight_decay" : [0.01],
    "batch_size" : [5],
    "epochs" : [10],
}

configurations = configurations_from_dict(search_space)

# %%
df_grid_res = pd.DataFrame()
for count, configuration in enumerate(configurations):
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)

    batch_size = configuration['batch_size']
    train_dl, val_dl, test_dl, total_dataset = make_dataset(path = titles_excel_file_path, train_batch_size= batch_size, tokenizer = tokenizer_yiyang, data_dir=data_dir, target_transforms=three_column_label_transform)
    
    params = hyperparameters_from_configurations(model = model, train_dataloader = train_dl, configurations= configuration)
    
    trainer = Trainer(model = model, train_dataloader = train_dl, eval_dataloader=val_dl,parameters = params, device = device)

    trainer.train_model()
    
    training_results = Results(output = trainer.training_outputs)
    eval_results = Results(output = trainer.eval_outputs)
    
    # print(f'\nParameters : {configuration}\n')
    # print(f'Evaluation Results : {eval_results.total_acc}')
    
    
    configs_res = configuration.copy()
    configs_res.update(eval_results.total_acc)
    df_grid_res = df_grid_res.append(configs_res, ignore_index= True)

df_grid_res.to_csv(str(data_dir) + '/results_7_oct.csv')

# model_name = 'test_model'

# trainer.save_model(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
# eval_results.save_results(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
# %%
