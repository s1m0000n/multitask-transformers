"""
Utils to load data for tasks with Russian 🇷🇺 texts
"""
import os
import os.path
import random
from re import findall
from typing import Optional, Dict, Any, Union, Tuple
import corus
import datasets
import wget
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, Union, Set
import pandas as pd
from .utils import download, split_index


# TODO: get rid of not that necessary pandas (=> remove another dependency)
def load_mokoron(positive_url: str = 'https://www.dropbox.com/s/fnpq3z4bcnoktiv/positive.csv?dl=1',
                 negative_url: str = 'https://www.dropbox.com/s/r6u59ljhhjdg6j0/negative.csv?dl=1',
                 splits: Optional[Dict[str, float]] = None, shuffle: bool = True,
                 cache_path: str = 'dataset_cache/mokoron',
                 input_field: str = "input", label_field: str = "label") -> datasets.DatasetDict:
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
        # TODO: replace this downloader with wget.download
        # also, path wrangling can become much more simple
        download(positive_url, cache_path[:-1])
    if not os.path.isfile(negative_filepath):
        download(negative_url, cache_path[:-1])
    pos_df= pd.read_csv(positive_filepath, sep=';', header=None)
    pos_df[label_field] = 1
    pos_df.rename(columns={3: input_field}, inplace=True)
    pos_df = pos_df[[input_field, label_field]]
    neg_df = pd.read_csv(negative_filepath, sep=';', header=None)
    neg_df[label_field] = 0
    neg_df.rename(columns={3: input_field}, inplace=True)
    neg_df = neg_df[[input_field, label_field]]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    sub_datasets = {}
    for part, indices in split_index(len(df), splits, shuffle).items():
        sub_datasets[part] = datasets.Dataset.from_pandas(df.iloc[indices])
    return datasets.DatasetDict(sub_datasets)


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


def preprocess_danetqa_input_pair(sample: Dict[str, Any]) -> Dict[str, str]:
    passage = sample["passage"]
    question = sample["question"]
    label = sample["label"]
    return {"input": (passage, question), "label": label}


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

@dataclass(frozen=True)
class CorusDatasetInfo:
    name: str
    description: Optional[str] = None
    tags: Optional[Set[str]] = None
    num_texts: Optional[int] = None
    uncompressed_size: Optional[str] = None
    source_url: Optional[str] = None

@dataclass
class CorusDataset:
    corus_func: Callable[[str], Any]
    url: str
    filename: Optional[str] = None
    info: Optional[CorusDatasetInfo] = None
    
    actual_path: Optional[str] = field(init=False, default=None)
    
    def __post_init__(self) -> None:
        if self.filename is None:
            self.filename = self.url.split("/")[-1]
    
    def download(self, cache_path: str = "dataset_cache/corus", force: bool = False) -> None:
        try:
            os.makedirs(cache_path)
        except:
            pass
        check_path = os.path.join(cache_path, self.filename)
        if not force and os.path.exists(check_path):
            self.actual_path = check_path
        else:
            self.actual_path = wget.download(self.url, os.path.join(cache_path, self.filename))
    
    def load_corus(self, file_path: Optional[str] = None) -> Any:
        file_path = self.actual_path if file_path is None else file_path
        return self.corus_func(self.actual_path)
    
    def to_dataset(self, corus_data: Any) -> datasets.Dataset:
        # TODO: replace with memory friendly & safe stuff
        columns = {}
        for record in corus_data:
            for field, value in record.__dict__.items():
                if field in columns:
                    columns[field].append(value)
                else:
                    columns[field] = [value]
        return datasets.Dataset.from_dict(columns)
    
    def to_dataset_dict(
        self, 
        dataset: datasets.Dataset, 
        splits: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
    ) -> datasets.DatasetDict:
        splits = splits if splits else {"train": 0.7, "validation": 0.15, "test": 0.15}
        sub_datasets = {}
        for part, indices in split_index(len(dataset), splits, shuffle).items():
            sub_datasets[part] = dataset.select(indices)
        return datasets.DatasetDict(sub_datasets)
    
    def load(
        self,
        make_splits: bool = True,
        splits: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
        cache_path: str = "dataset_cache/corus", 
        force_redownload: bool = False,
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        self.download(cache_path, force_redownload)
        dataset = self.to_dataset(self.load_corus())
        if make_splits:
            return self.to_dataset_dict(dataset, splits, shuffle)
        return dataset

# TODO: move stuff like this to some yaml
LENTA_DESCRIPTION = "Corpus of news articles of Lenta.Ru"
LENTA_URL = "https://github.com/yutkin/Lenta.Ru-News-Dataset"
lenta_tags = {"news",}
corus_datasets = {
    "lenta": CorusDataset(
        corus_func = corus.load_lenta, 
        url = "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz",
        info = CorusDatasetInfo(
            name = "Lenta.ru v1.0",
            description = LENTA_DESCRIPTION,
            tags = lenta_tags,
            num_texts = 739351,
            uncompressed_size = "1.66 Gb",
            source_url = LENTA_URL,
        )
    ),
    "lenta2": CorusDataset(
        corus_func = corus.load_lenta2, 
        url = "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2",
        info = CorusDatasetInfo(
            name = "Lenta.ru v1.1+",
            description = LENTA_DESCRIPTION,
            tags = lenta_tags,
            num_texts = 800975,
            uncompressed_size = "1.94 Gb",
            source_url = LENTA_URL,
        )
    ),
}
