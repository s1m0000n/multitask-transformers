import multiprocessing as mp
import datasets
from typing import *
from functools import partial
from operator import lt
from tqdm import tqdm
import numpy as np


def split_idx(length: int, splits: Optional[Dict[str, float]] = None, shuffle: bool = True):
    if splits is None:
        splits = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    else:
        vals = splits.values()
        assert all(map(lambda x: isinstance(x, float), vals)), "expected float split ratios"
        assert sum(vals) == 1, "ratio values of parts must sum up to 1"
        assert all(map(partial(lt, 0), vals)), "part ratios must be > 0"
    if shuffle:
        seed = np.random.get_state()[1][0]
        _ = np.random.random()  # do 1 step of rng, TODO: understand why
        generator = np.random.default_rng(seed)
        index = generator.permutation(length)
    else:
        index = np.arange(0, length)
    result = dict()
    prev = 0
    for k, v in splits.items():
        v = prev + int(v * length)
        result[k] = index[prev: v]
        prev = v
    return result


def dict_map(datadict: datasets.DatasetDict,
             func: Callable[[Dict[str, Any]], Optional[Union[Dict[str, Any], Iterable[Dict[str, Any]]]]],
             verbose: bool = True, progress_bars: bool = False, name: Optional[str] = None,
             parallel: bool = True, parallel_chunksize: Optional[int] = None,
             num_processes: Optional[int] = None) -> datasets.DatasetDict:
    """
    Apply function to each parts' records-dicts

    TODO Add use examples

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
    :param func: mapping dict with original {"field": values} to None(drop this record), single {"field": values} dictionary as a replacement or an iterable of new records made from the original one (for example augmenting texts / images)
    :param progress_bars: output progress bars (tqdm) may slow down parallel execution
    :param name: dataset name to use in verbose messages
    :param parallel: parallel computation using multiprocessing.Pool().{i}map ({i} depends on verbose)
    :param parallel_chunksize: for multiprocessing.pool.{i}map
    :param num_processes: number of processes to be used, defaults to number of processes found in your system
    :return: modified datasets.DatasetDict
    """
    data = dict(datadict) if isinstance(datadict, datasets.DatasetDict) else datadict
    result = dict()
    if name is None:
        name = 'unnamed_dataset'
    if verbose:
        print(f'map({func.__name__}, {name}):')
    for i, (part, dataset) in enumerate(data.items()):
        if verbose:
            print(f'    map({func.__name__}, {name}[{part}])', end='')
            print(':' if progress_bars else '')
        cols = dict()
        i = 0
        if parallel:
            if num_processes is None:
                num_processes = mp.cpu_count()
            with mp.Pool(processes=num_processes) as pool:
                if progress_bars:
                    if parallel_chunksize is None:
                        parallel_chunksize = 1  # TODO: do something smarter
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


def fmap(datadict, funcs) -> datasets.DatasetDict:
    # TODO: docstring
    print("Warning: Using data_utils.fmap is not parallel, but planned in near future")

    def column_functions(data):
        return {col: (data[col] if f is None else f(data)) for col, f in funcs.items()}

    return dict_map(datadict, column_functions, parallel=False)  # TODO: fix to be working in parallel mode
