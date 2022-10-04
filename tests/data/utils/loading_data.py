# %%
from typing import List
import pandas as pd
from pathlib import Path

def load_all_news_titles_as_list() -> List:
    data_directory = Path(__file__).parent.parent
    all_news_titles = pd.read_csv(str(data_directory) + "/all_news_titles.csv",on_bad_lines='skip', delimiter= ";", header=0)
    all_news_titles_list = list(all_news_titles['title'])
    return all_news_titles_list
# %%
