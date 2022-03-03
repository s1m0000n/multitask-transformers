# TODO: Docstrings for everything
import random
from dataclasses import dataclass, field
from datasets import DatasetDict, Dataset
from transformers import DataCollatorWithPadding, PreTrainedModel
from typing import Any, Optional, Set, Dict, Iterable, Callable, Union

from .preprocessing import Preprocessor
from .tokenizers import TokenizerT, ConfiguredTokenizer
from .utils import slice_into_chunks


@dataclass
class DataLoader:
    data: Dataset
    collate_fun: Callable
    batch_size: int
    shuffle: bool = True
    finite: bool = False

    def __iter__(self) -> 'DataLoader':
        data = self.data.shuffle() if self.shuffle else self.data
        self.batch_data = slice_into_chunks(self.batch_size, data)
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __next__(self):
        if self.batch_data:
            return self.collate_fun(next(self.batch_data))
        elif self.finite:
            raise StopIteration()
        else:
            self.__iter__()
            return self.collate_fun(next(self.batch_data))


@dataclass
class Data:
    data: DatasetDict
    configured_tokenizer: ConfiguredTokenizer
    preprocessor: Optional[Preprocessor] = None
    collator_class: Any = DataCollatorWithPadding
    collator_config: Optional[Dict[str, Any]] = None
    collator: Any = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, DatasetDict):
            raise TypeError("Wrong type for 'dataset_dict', must be an instance of 'DatasetDict'")
        if self.preprocessor is not None:
            self.data = self.preprocessor(self.data)
        collator_cfg = {} if self.collator_config is None else self.collator_config
        self.collator = self.collator_class(self.configured_tokenizer.tokenizer, **collator_cfg)

    @property
    def model_path(self) -> str:
        return self.configured_tokenizer.model_path

    @property
    def tokenizer(self) -> TokenizerT:
        return self.configured_tokenizer.tokenizer

    @property
    def tokenizer_init_config(self) -> Dict[str, Any]:
        return self.configured_tokenizer.init_config

    @property
    def tokenizer_call_config(self) -> Dict[str, Any]:
        call_config = self.__dict__
        del call_config["model_path"]
        del call_config["init_config"]
        del call_config["tokenizer"]
        return call_config

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def parts(self) -> Set[str]:
        return set(self.keys())

    def __getitem__(self, part: str):
        if part not in self.parts:
            raise KeyError(f"Expected 'part' to be in 'parts = {self.parts}', but got {part}")
        return self.data[part]

    def make_features(
            self,
            part: str,
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    ) -> Dataset:
        features = self[part].map(self.configured_tokenizer, batched=True)
        features.set_format(type="torch", columns=list(columns))
        return features

    def make_data_loader(
            self,
            part: str,
            batch_size: int,
            shuffle: bool = True,
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels"),
            finite: bool = False
    ) -> DataLoader:
        features = self.make_features(part, columns)
        return DataLoader(features, self.collator, batch_size, shuffle, finite)


@dataclass
class MultitaskDataLoader:
    task_datasets: Dict[str, Data]
    part: str
    batch_size: Optional[int] = None
    task_batch_sizes: Optional[Dict[str, int]] = None
    shuffle_task_data: Union[bool, Dict[str, bool]] = True
    shuffle_batches: bool = True
    columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    finite: bool = False

    def __post_init__(self):
        # Validation of batch size data & generalizing
        if self.task_batch_sizes is None:
            self.task_batch_sizes = {}
        for task in self.task_datasets:
            if task not in self.task_batch_sizes:
                if self.batch_size is None:
                    raise ValueError("Expected to have either batch size for all tasks in "
                                     "'task_batch_sizes' and 'batch_size': Optional[Any] or"
                                     "skipped some tasks in 'task_batch_sizes' with default in 'batch_size' or"
                                     "'task_batch_sizes' = None, 'batch_size': int "
                                     "for all tasks to have the same batch size")
                elif not isinstance(self.batch_size, int):
                    raise TypeError("Wrong type for 'batch_size', must be an instance of 'int'")
                else:
                    self.task_batch_sizes[task] = self.batch_size
            elif not isinstance(self.task_batch_sizes[task], int):
                raise TypeError("Wrong type for 'task_batch_sizes' keys, must be instances of 'int'")

        # Validating task shuffling & generalizing
        if isinstance(self.shuffle_task_data, bool):
            value = self.shuffle_task_data
            self.shuffle_task_data = {task: value for task in self.task_datasets}
        elif isinstance(self.shuffle_task_data, dict):
            if set(self.shuffle_task_data.keys()) != set(self.task_datasets.keys()):
                raise KeyError("'shuffle_task_data' must contain the same keys as 'task_datasets'")
        else:
            raise TypeError("Wrong type for 'shuffle_task_data', must be either bool for same shuffling rules "
                            "for all task datasets or Dict[str, bool] for per-task values")

        # Validating shuffle_shuffle batches
        if not isinstance(self.shuffle_batches, bool):
            raise TypeError("Wrong type for 'shuffle_batches', must be an instance of bool")

        # Making batches for all future epoch-iterations - __iter__ calls
        self._batches = []
        for task, data in self.task_datasets.items():
            batch_size = self.task_batch_sizes[task]
            shuffle = self.shuffle_task_data[task]
            for batch in data.make_data_loader(self.part, batch_size, shuffle, self.columns, finite=True):
                self._batches.append({task: batch})

    def __iter__(self) -> 'MultitaskDataLoader':
        self._current_loop_batches = self._batches[:]
        if self.shuffle_batches:
            random.shuffle(self._current_loop_batches)
        return self

    def __len__(self) -> int:
        return len(self._batches)

    def __next__(self):
        if self._current_loop_batches:
            return self._current_loop_batches.pop()
        elif self.finite:
            raise StopIteration()
        else:
            self.__iter__()
            return self._current_loop_batches.pop()


@dataclass
class TaskBatch:
    name: str
    data: Dict[str, Any]
    model_head: PreTrainedModel
    metrics: Optional[Callable] = None

    def data_on_device(self, device) -> Dict[str, Any]:
        data = {k: v.to(device) for k, v in self.data.items()}
        return data

    def __call__(self, *args, **kwargs):
        return self.model_head(*args, **kwargs)


@dataclass
class MultitaskBatchSampler:
    # Actually tasks: Tasks, but can not import it,
    # because import loop will happen if data_loaders imports tasks
    # and for tasks it's necessary to import data_loaders
    tasks: Any
    part: str
    batch_size: Optional[int] = None
    task_batch_sizes: Optional[Dict[str, int]] = None
    shuffle_task_data: Union[bool, Dict[str, bool]] = True
    shuffle_batches: bool = True
    columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    finite: bool = False

    def __post_init__(self) -> None:
        self.data_loader = self.tasks.make_data_loader(
            part=self.part,
            batch_size=self.batch_size,
            task_batch_sizes=self.task_batch_sizes,
            shuffle_task_data=self.shuffle_task_data,
            shuffle_batches=self.shuffle_batches,
            columns=self.columns,
            finite=self.finite
        )
        self.data_loader_iter = iter(self.data_loader)

    def __iter__(self) -> 'MultitaskBatchSampler':
        return self

    def __next__(self) -> TaskBatch:
        data_dict = next(self.data_loader_iter)
        name = tuple(data_dict.keys())[0]
        data = tuple(data_dict.values())[0]
        return TaskBatch(
            name=name,
            data=data,
            model_head=self.tasks.model[name],
            metrics=self.tasks[name].metrics
        )

    def __len__(self) -> int:
        return len(self.data_loader)