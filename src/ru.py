import os
import os.path
import random
from re import findall
from typing import Optional, Dict, Any, Union, Tuple

from datasets import Dataset, DatasetDict
import pandas as pd

from .utils import download, split_index


def load_mokoron(positive_url: str = 'https://www.dropbox.com/s/fnpq3z4bcnoktiv/positive.csv?dl=1',
                 negative_url: str = 'https://www.dropbox.com/s/r6u59ljhhjdg6j0/negative.csv?dl=1',
                 splits: Optional[Dict[str, float]] = None, shuffle: bool = True,
                 cache_path: str = 'dataset_cache/mokoron',
                 input_field: str = "input", label_field: str = "label") -> DatasetDict:
    """
    Loading the mokoron dataset by url / from cache, preparing a splitted shuffled DatasetDict

    Note: positive_url / negative_url can contain ?... (for default dropbox for example)
    :param label_field: Name of field representing label in the downloaded csv & later (usually "label")
    :param input_field: Name of field representing text input in the downloaded csv & later (usually "input")
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
    pos_df: pd.DataFrame = pd.read_csv(positive_filepath, sep=';', header=None)
    pos_df[label_field] = 1
    pos_df.rename(columns={3: input_field}, inplace=True)
    pos_df = pos_df[[input_field, label_field]]
    neg_df: pd.DataFrame = pd.read_csv(negative_filepath, sep=';', header=None)
    neg_df[label_field] = 0
    neg_df.rename(columns={3: input_field}, inplace=True)
    neg_df = neg_df[[input_field, label_field]]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    sub_datasets = {}
    for part, indices in split_index(len(df), splits, shuffle).items():
        sub_datasets[part] = Dataset.from_pandas(df.iloc[indices])
    return DatasetDict(sub_datasets)


def preprocess_danetqa(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Preprocess Russian SuperGLUE DaNetQA sample for binary classification
    :param sample: One dict-like sample
    :return: Format for sequence classification models ('text', 'label')
    """
    passage = sample["passage"]
    question = sample["question"]
    label = sample["label"]
    return {"input": f"{passage} {question}", "label": label}


def preprocess_parus(sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
    """
    Preprocesses Russian SuperGLUE PARus sample for binary classification
    :param sample: One dict-like sample of RSG's PARus in Hugging Face Datasets
    :return: Format for sequence classification models, in this case 0th or 1st option is correct
    """
    questions = {
        'cause': (
            'По какой причине?',
            'Почему?',
            'Причина этому',
            'Это произошло, потому что',
        ),
        'effect': (
            'Что произошло в результате?',
            'Из-за этого',
            'Вследствие чего',
            'В итоге получилось, следующее:'
        )
    }
    question_number = random.randint(0, len(questions) - 1)
    question = questions[sample["question"]][question_number]
    inputs = tuple(f'{sample["premise"]} {question} {sample[f"choice{j}"]}' for j in (1, 2))
    return {"label": sample["label"], "input": inputs}


def preprocess_headline_cause(sample: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    inputs = (sample["left_title"], sample["right_title"])
    return {"label": sample["label"], "input": inputs}
