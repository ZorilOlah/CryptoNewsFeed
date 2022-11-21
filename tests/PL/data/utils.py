import pandas as pd
from pathlib import Path

def three_column_label_transform(labels_dict : dict):
    translator = {'Negative' : 0, 'Neutral' : 1, 'Positive' : 2}
    try:
        for label_type, value  in labels_dict.items():
            if type(value) == str:
                label = translator[label_type]
        return label
    except Exception as e: 
        print(f'finding label did not work. Here is the dict : {labels_dict} due to error : {e}')

def select_number_rows(excel_file_path : str,
                       nr_rows : int, 
                       save : bool) -> pd.DataFrame:
    parent_directory = Path(excel_file_path).parent
    dataset = pd.read_csv(excel_file_path)
    filtered_dataset = dataset[0:nr_rows]
    if save:
        filtered_dataset.to_csv(str(parent_directory) + f'/{nr_rows}_examples.csv')
    return filtered_dataset    

# %%