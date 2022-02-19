"""
Utilities for dowloading the data and other things
"""

import os
from operator import lt
from functools import partial
import multiprocessing as mp
from typing import Dict, Iterable, Tuple, Optional, Any, Union, Callable
import requests
import datasets
import numpy as np
from tqdm import tqdm
from .trf import Task


def download(url: str, destination_folder: str) -> str:
    """
    Downloads the file from url, saves to destination folder
    :param url: resource url, file extension extracted from it too
    :param destination_folder: path, where the file is saved
    :return: filename
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # create folder if it does not exist
    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    if "?" in filename:
        filename = filename.split('?')[0]
    file_path = os.path.join(destination_folder, filename)
    ref = requests.get(url, stream=True)
    if ref.ok:
        with open(file_path, 'wb') as file:
            for chunk in ref.iter_content(chunk_size=1024 * 8):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
        return filename
    # HTTP status code 4XX/5XX
    raise requests.HTTPError(f"Download failed: status code{ref.status_code}\n{ref.text}")


def make_features(tasks: Dict[str, Task], *map_args,
                  fields: Iterable[str] = ('input_ids', 'attention_mask', 'labels'),
                  **map_kwargs) -> Dict[str, Dict[str, datasets.Dataset]]:
    """
    Makes features to be used with Multitask Transformers which are similar to the regular ones,
    but a dictionary with data for each task
    :param tasks: The tasks with defined data, converter
    :param *map_args: Arguments for DatasetDict.map(...)
    :param fields: Columns in features (typically ('input_ids', 'attention_mask', 'labels') for transformers)
    :param **map_kwargs: Keyword arguments for DatasetDict.map(...) (example: batched=True)
    :return: return_description
    """

    features = {}
    for name, task in tasks.items():
        features[name] = {}
        for split_name, split_dataset in task.data.items():
            features[name][split_name] = split_dataset.map(task.converter, *map_args, **map_kwargs)
            features[name][split_name].set_format(type="torch", columns=fields)
    return features


def unpack_splits(features: Dict[str, Dict[str, datasets.Dataset]],
                  *split_names: Tuple[str]):
    """
    Separate multitask features into taskwise dict(task_name:dataset) or
    if unpackable tuple of this dicts for each split in *split_names
    :param features: Multitask features dict(for example: make_features(...))
    :param *split_names: Names of splits to aggregate
    :return: Single dict or multiple packed in a tuple for each split
    """
    assert len(split_names) >= 1, IndexError("Expected at least 1 split name (for example \"train\")")
    result = []
    for split_name in split_names:
        result.append({task_name: dataset[split_name] for task_name, dataset in features.items()})
    return tuple(result) if len(result) > 1 else result[0]


def split_index(length: int, splits: Optional[Dict[str, float]] = None, shuffle: bool = True):
    """
    Generates splitted and optionally shuffled index for length split into parts
    :param length: Length of generated index (ex: length=10 => indices 0, 1, ..., 9)
    :param splits; Dictionary of names and corresponding parts summing up to 1 (ex: {"train": 0.7, "test": 0.3})
    :param shuffle: Shuffling the index before splitting
    :return: Dictionary of parts with indices str -> np.array
    """
    if splits is None:
        splits = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    else:
        vals = splits.values()
        assert all(map(lambda x: isinstance(x, float), vals)), "expected float split ratios"
        assert sum(vals) == 1, "ratio values of parts must sum up to 1"
        assert all(map(partial(lt, 0), vals)), "part ratios must be > 0"
    if shuffle:
        seed = np.random.get_state()[1][0]
        _ = np.random.random()
        generator = np.random.default_rng(seed)
        index = generator.permutation(length)
    else:
        index = np.arange(0, length)
    result = {}
    prev = 0
    for key, value in splits.items():
        value = prev + int(value * length)
        result[key] = index[prev: value]
        prev = value
    return result


def dict_map(datadict: datasets.DatasetDict,
             func: Callable[[Dict[str, Any]], Optional[Union[Dict[str, Any], Iterable[Dict[str, Any]]]]],
             verbose: bool = True, progress_bars: bool = False, name: Optional[str] = None,
             parallel: bool = True, parallel_chunksize: Optional[int] = None,
             num_processes: Optional[int] = None) -> datasets.DatasetDict:
    """
    Apply function to each parts' records-dicts

    # FIX: add examples
    # -> https://github.com/s1m0000n/multitask-transformers/issues/15

    Notes about parallel mode:\n
    1. if verbose = True, parallel = True then multiprocessing.pool.imap used,
    where chunksize > 1 can improve performance dramatically, see more info:
    https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap \n

    2. if parallel = True, function must be defined before called completely,
    it can't be a lambda or closured inside another function
    (because of Python's multiprocessing internal implementation,
    where pickling is used, which can only encode predefined callables) \n

    :param verbose: print some messages
    :param datadict: dictionary of datasets.Dataset with names
    :param func: mapping dict with original {"field": values} to None(drop this record), \
                 single {"field": values} dictionary as a replacement or an iterable of \
                 new records made from the original one (for example augmenting texts / images)
    :param progress_bars: output progress bars (tqdm) may slow down parallel execution
    :param name: dataset name to use in verbose messages
    :param parallel: parallel computation using multiprocessing.Pool().{i}map ({i} depends on verbose)
    :param parallel_chunksize: for multiprocessing.pool.{i}map
    :param num_processes: number of processes to be used, defaults to number of processes found in your system
    :return: modified datasets.DatasetDict
    """
    data = dict(datadict) if isinstance(datadict, datasets.DatasetDict) else datadict
    result = {}
    if name is None:
        name = 'unnamed_dataset'
    if verbose:
        print(f'map({func.__name__}, {name}):')
    for i, (part, dataset) in enumerate(data.items()):
        if verbose:
            print(f'    map({func.__name__}, {name}[{part}])', end='')
            print(':' if progress_bars else '')
        cols = {}
        i = 0
        if parallel:
            if num_processes is None:
                num_processes = mp.cpu_count()
            with mp.Pool(processes=num_processes) as pool:
                if progress_bars:
                    if parallel_chunksize is None:
                        iterator = tqdm(pool.imap(func, dataset), total=len(dataset))
                    else:
                        iterator = tqdm(pool.imap(func, dataset, chunksize=parallel_chunksize), total=len(dataset))
                else:
                    iterator = pool.map(func, dataset)
            for sample in iterator:
                if isinstance(sample, dict):
                    for col, value in sample.items():
                        if col not in cols:
                            cols[col] = [None, ] * i
                        else:
                            cols[col].append(value)
                    i += 1
                elif isinstance(sample, Iterable):
                    for subsample in sample:
                        for col, value in subsample.items():
                            if col not in cols:
                                cols[col] = [None, ] * i
                            else:
                                cols[col].append(value)
                        i += 1
        else:
            iterator = tqdm(dataset) if progress_bars else dataset
            for sample in iterator:
                new_sample = func(sample)
                if isinstance(new_sample, dict):
                    for col, value in new_sample.items():
                        if col not in cols:
                            cols[col] = [None, ] * i
                        else:
                            cols[col].append(value)
                    i += 1
                elif isinstance(new_sample, Iterable):
                    for subsample in new_sample:
                        for col, value in subsample.items():
                            if col not in cols:
                                cols[col] = [None, ] * i
                            else:
                                cols[col].append(value)
                        i += 1
        result[part] = datasets.Dataset.from_dict(cols)
    return datasets.DatasetDict(result)
