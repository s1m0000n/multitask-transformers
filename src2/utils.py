import datasets
from datasets import DatasetDict, Dataset
from typing import Callable, Dict, Any, Optional, Union, Iterable, Mapping, Iterator, TypeVar
import multiprocessing as mp
import pandas as pd
from tqdm.auto import tqdm

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
