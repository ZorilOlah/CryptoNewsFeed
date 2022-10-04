# %%
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
from transformers import pipeline
from data.utils.loading_data import load_all_news_titles_as_list
import pandas as pd
from pathlib import Path

finbert_yiyang = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_yiyang = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

all_titles = load_all_news_titles_as_list()

hundred_titles = all_titles[0:100]

nlp = pipeline("sentiment-analysis", model=finbert_yiyang, tokenizer=tokenizer_yiyang)

results = nlp(hundred_titles)
print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

label_list = [item['label'] for item in results]
score_list = [item['score'] for item in results]

res_dict = {'titles' : hundred_titles, 'labels' : label_list, 'score' : score_list}

results_df = pd.DataFrame(res_dict)

results_df.to_csv(str(Path(__file__).parent) + '/data/all_news_titles_results.csv')

# finbert_prosus = AutoModel.from_pretrained('ProsusAI/finbert',num_labels=3)
# tokenizer_prosus = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# model_list = [finbert_yiyang, finbert_prosus]
# tokenizer_list = [tokenizer_yiyang, tokenizer_prosus]

# for model, tokenizer in zip(model_list, tokenizer_list):
#     nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#     sentences = ["there is a shortage of capital, and we need extra financing",  
#                 "growth is strong and we have plenty of liquidity", 
#                 "there are doubts about our finances", 
#                 "profits are flat"]
#     results = nlp(sentences)
#     print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
# %%
