# %%
import itertools
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
from typing import List
import torch

def configurations_from_dict(hyperparamter_dict : dict) -> List:
    hyper_parameters = []
    for values in itertools.product(*hyperparamter_dict.values()):
        hyper_parameters.append(dict(zip(hyperparamter_dict.keys(), values)))
    return hyper_parameters

def hyperparameters_from_configurations(model : BertForSequenceClassification,
                    train_dataloader : torch.utils.data.DataLoader,
                    configurations : dict) -> dict:
    parameters = {}
    parameters['optimizer'] = torch.optim.AdamW(model.parameters(), lr = configurations['learning_rate'], weight_decay= configurations['weight_decay'])
    parameters['num_epochs'] = configurations['epochs']
    parameters['batch_size'] = configurations['batch_size']
    parameters['num_trainig_steps'] = parameters['num_epochs'] * len(train_dataloader)
    parameters['lr_scheduler'] = get_scheduler(name = 'linear', optimizer = parameters['optimizer'], num_warmup_steps=0, num_training_steps = parameters['num_trainig_steps'])
    return parameters
# %%
