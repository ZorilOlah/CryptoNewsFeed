# %%
from typing import List, Callable
from pathlib import Path
import torch
from data.dataset import TitlesDataset, make_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm 
from models.training import Trainer
from data.utils import three_column_label_transform
from datasets import load_metric
from evaluation.evaluation import Results
from collections import Counter

titles_excel_file_path = str(Path(__file__).parent) + "/data/50_examples.csv"    
data_dir = str(Path(__file__).parent) + "/data"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer_yiyang = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone') 
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
# %%

train_dl, val_dl, test_dl, total_dataset = make_dataset(path = titles_excel_file_path, train_batch_size= 32, tokenizer = tokenizer_yiyang, data_dir=data_dir, target_transforms=three_column_label_transform)
# %%

num_epochs = 1
num_training_steps = num_epochs * len(train_dl)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(name = 'linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps = num_training_steps)

trainer = Trainer(model = model, train_dataloader = train_dl, eval_dataloader=val_dl, num_epochs=num_epochs, optimizer = optimizer, lr_scheduler=lr_scheduler, device = device)


# %%

trainer.train_model()
trainer.eval_model()

# %% 

training_results = Results(output = trainer.training_outputs)

eval_results = Results(output = trainer.eval_outputs)
print(f'Evaluation Results : {eval_results.results["epoch_1"]["total"]}')

model_name = 'test_model'

trainer.save_model(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
eval_results.save_results(str(Path(__file__).parent) + '/models/saved_models/' + model_name)
# %%
