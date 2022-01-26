import os
import re
from typing import *
import datasets
import pandas as pd
from data_utils import split_idx
from utils import download


def load(positive_url: str = 'https://www.dropbox.com/s/fnpq3z4bcnoktiv/positive.csv?dl=1',
         negative_url: str = 'https://www.dropbox.com/s/r6u59ljhhjdg6j0/negative.csv?dl=1',
         splits: Optional[Dict[str, float]] = None, shuffle: bool = True,
         temp_path: str = 'dataset_cache/mokoron') -> datasets.DatasetDict:
    temp_path += "" if temp_path.endswith('/') else "/"
    positive_file = positive_url.split('/')[-1]
    if "?" in positive_file:
        name, ext = re.findall(r'(?P<name>\w+).(?P<ext>\w+)\?', positive_file)[0]
        positive_file = f"{name}.{ext}"
    positive_filepath = temp_path + positive_file
    negative_file = negative_url.split('/')[-1]
    if "?" in negative_file:
        name, ext = re.findall(r'(?P<name>\w+).(?P<ext>\w+)\?', negative_file)[0]
        negative_file = f"{name}.{ext}"
    negative_filepath = temp_path + negative_file
    if not os.path.isfile(positive_filepath):
        download(positive_url, temp_path[:-1])
    if not os.path.isfile(negative_filepath):
        download(negative_url, temp_path[:-1])
    pos_df = pd.read_csv(positive_filepath, sep=';', header=None)
    pos_df['label'] = 1
    pos_df.rename(columns={3: 'text'}, inplace=True)
    pos_df = pos_df[['text', 'label']]
    neg_df = pd.read_csv(negative_filepath, sep=';', header=None)
    neg_df['label'] = 0
    neg_df.rename(columns={3: 'text'}, inplace=True)
    neg_df = neg_df[['text', 'label']]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    sub_datasets = dict()
    for part, indices in split_idx(len(df), splits, shuffle).items():
        sub_datasets[part] = datasets.Dataset.from_pandas(df.iloc[indices])
    return datasets.DatasetDict(sub_datasets)
