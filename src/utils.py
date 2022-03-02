import os
import requests
import multiprocessing as mp
from functools import partial
from operator import lt
from typing import Callable, Dict, Any, Optional, Union, Iterable, Mapping, Iterator, TypeVar

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import datasets
from datasets import DatasetDict, Dataset

T = TypeVar('T')
T2 = TypeVar('T2')


def flatten(data: Iterable[Any],
            unpack_cond: Optional[Callable[[Any], bool]] = None,
            drop_cond: Optional[Callable[[Any], bool]] = None) -> Iterator[Any]:
    if unpack_cond is None:
        def unpack_cond(elem: Any) -> bool:
            return isinstance(elem, Iterable)
    if drop_cond is None:
        def drop_cond(_) -> bool:
            return False
    for item in data:
        if not drop_cond(item):
            if unpack_cond(item):
                for x in item:
                    yield x
            else:
                yield item


Mapper = Callable[[Dict[str, Any]], Optional[Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]]]


def dmap(func: Mapper, datadict: Union[DatasetDict, Mapping[str, Dataset]],
         parallel: bool = True, num_proc: Optional[int] = None,
         verbose: bool = False, verbose_name: str = "Unnamed Dataset") -> DatasetDict:
    def iftqdm(x, *args, **kwargs):
        return tqdm(x, *args, **kwargs) if verbose else x

    def ifprint(*args, **kwargs):
        return print(*args, **kwargs) if verbose else None

    data = dict(datadict) if isinstance(datadict, DatasetDict) else datadict
    parts = {}
    ifprint(f"Processing {verbose_name}")
    for part_name, dataset in iftqdm(data.items()):
        ifprint(f"=> Part {part_name}:")
        if parallel:
            with mp.Pool(mp.cpu_count() if num_proc is None else num_proc) as pool:
                processed = (pool.imap if verbose else pool.map)(func, dataset)
        else:
            processed = map(func, dataset)
        parts[part_name] = Dataset.from_pandas(pd.DataFrame.from_records(
            flatten(iftqdm(processed, total=len(dataset)),
                    lambda x: not isinstance(x, Dict),
                    lambda x: x is None)))
    return datasets.DatasetDict(parts)


def download(url: str, destination_folder: str) -> str:
    """
    Downloads the file from url, saves to destination folder
    :param url: resource url, file extension extracted from it too
    :param destination_folder: path, where the file is saved
    :return: filename | http error fot 4XX, 5XX codes
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
    raise requests.HTTPError(f"Download failed: status code{ref.status_code}\n{ref.text}")


def split_index(length: int,
                splits: Optional[Dict[str, float]] = None,
                shuffle: bool = True) -> Dict[str, np.ndarray]:
    """
    Generates splitted and optionally shuffled index for length split into parts
    :param length: Length of generated index (ex: length=10 => indices 0, 1, ..., 9)
    :param splits: Dictionary of names and corresponding parts summing up to 1 (ex: {"train": 0.7, "test": 0.3})
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


def itercat(*iterables: Iterable[T]) -> Iterable[T]:
    for iterable in iterables:
        if isinstance(iterable, Iterable):
            for item in iterable:
                yield item
        else:
            yield iterable
