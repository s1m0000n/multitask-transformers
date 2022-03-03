import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Any, Optional, Mapping, Iterable, Dict, Union, Tuple

import pandas as pd
from tqdm.auto import tqdm
from datasets import DatasetDict, Dataset

from .utils import flatten

ActionT = Callable[[Dict[str, Any]], Optional[Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]]]


@dataclass(frozen=True)
class Preprocessor:
    actions: Iterable[ActionT]
    parallel: bool = True
    num_proc: Optional[int] = None
    verbose: bool = False

    def _print(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _tqdm(self, data: Iterable, *args, **kwargs):
        if self.verbose:
            return tqdm(data, *args, **kwargs)
        else:
            return data

    def _process_part(self, name: str, data: Iterable[Dict[str, Any]]) -> Dataset:
        self._print(f"=> Part {name}")
        for action in self.actions:
            if self.parallel:
                with mp.Pool(mp.cpu_count() if self.num_proc is None else self.num_proc) as pool:
                    data = (pool.imap if self.verbose else pool.map)(action, data)
            else:
                data = map(action, data)
            data = flatten(
                data=data,
                unpack_cond=lambda x: not isinstance(x, Dict),
                drop_cond=lambda x: x is None
            )
        return Dataset.from_pandas(pd.DataFrame.from_records(self._tqdm(data)))

    def __call__(self, data: Union[DatasetDict, Mapping[str, Dataset]], name: str = "Unnamed Dataset") -> DatasetDict:
        self._print(f"Preprocessing {name}:")
        parts = {name: self._process_part(name, dataset)
                 for name, dataset in dict(data).items()}
        return DatasetDict(parts)


@dataclass(frozen=True)
class NLIPreprocess:
    """
    Preprocessing for NLI _tasks with fields hypothesis, premise & label
    """
    hypothesis_field: str = "hypothesis"
    premise_field: str = "premise"
    label_field: str = "label"
    input_field: str = "input"
    label_field_in: str = "label"

    def __call__(self, sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
        hypothesis = sample[self.hypothesis_field]
        premise = sample[self.premise_field]
        label = sample[self.label_field_in]
        return {self.input_field: (hypothesis, premise), self.label_field: label}
