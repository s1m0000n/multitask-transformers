"""
Loading mokoron dataset for sentiment analysis of russian tweets
Original post: https://study.mokoron.com/
"""

import os
from re import findall
from typing import Optional, Dict
import datasets
import pandas as pd
from utils import download, split_index


def load(positive_url: str = 'https://www.dropbox.com/s/fnpq3z4bcnoktiv/positive.csv?dl=1',
         negative_url: str = 'https://www.dropbox.com/s/r6u59ljhhjdg6j0/negative.csv?dl=1',
         splits: Optional[Dict[str, float]] = None, shuffle: bool = True,
         cache_path: str = 'dataset_cache/mokoron') -> datasets.DatasetDict:
    """
    Loading the mokoron dataset by url / from cache, preparing a splitted shuffled DatasetDict

    Note: positive_url / negative_url can contain ?... (for default dropbox for example)
    :param positive_url: Positive part of the dataset in .csv format
    :param negative_url: Negative part of the dataset in .csv format
    :param splits: Dictionary of names and corresponding parts summing up to 1 (ex: {"train": 0.7, "test": 0.3})
    :param shuffle: Shuffling before splitting
    :param cache_path: Path to save the downloaded dataset
    """
    cache_path += "" if cache_path.endswith('/') else "/"
    positive_file = positive_url.split('/')[-1]
    if "?" in positive_file:
        name, ext = findall(r'(?P<name>\w+).(?P<ext>\w+)\?', positive_file)[0]
        positive_file = f"{name}.{ext}"
    positive_filepath = cache_path + positive_file
    negative_file = negative_url.split('/')[-1]
    if "?" in negative_file:
        name, ext = findall(r'(?P<name>\w+).(?P<ext>\w+)\?', negative_file)[0]
        negative_file = f"{name}.{ext}"
    negative_filepath = cache_path + negative_file
    if not os.path.isfile(positive_filepath):
        download(positive_url, cache_path[:-1])
    if not os.path.isfile(negative_filepath):
        download(negative_url, cache_path[:-1])
    pos_df = pd.read_csv(positive_filepath, sep=';', header=None)
    pos_df['label'] = 1
    pos_df.rename(columns={3: 'text'}, inplace=True)
    pos_df = pos_df[['text', 'label']]
    neg_df = pd.read_csv(negative_filepath, sep=';', header=None)
    neg_df['label'] = 0
    neg_df.rename(columns={3: 'text'}, inplace=True)
    neg_df = neg_df[['text', 'label']]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    sub_datasets = {}
    for part, indices in split_index(len(df), splits, shuffle).items():
        sub_datasets[part] = datasets.Dataset.from_pandas(df.iloc[indices])
    return datasets.DatasetDict(sub_datasets)
