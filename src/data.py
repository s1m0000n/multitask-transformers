"""
Utilities & conventional data structures for single- and multitask data loading, preprocessing,
building & collating batches for single task / mixed tasks
"""
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Set, Dict, Iterable, Callable, Union, Tuple, List

from datasets import DatasetDict, Dataset
from transformers import DataCollatorWithPadding, PreTrainedModel

from .preprocessing import Preprocessor
from .tokenizers import TokenizerT, ConfiguredTokenizer
from .utils import slice_into_chunks


@dataclass
class DataLoader:
    """
    Just regular implementation of data loader with collation
    and finite / infinite Iteration

    :param data: Dataset with PyTorch feature dicts
    :param collate_fun: Data collator callable
    :param batch_size: Integer batch size
    :param shuffle: If true => data is shuffled before slicing into batches & collation
    :param finite: If false => just yields all the data a single time without repeats, else infinite iterator
    """
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
    """
    Data loader for multitask learning

    Note, that it yields only data and task name, so for training a multitask model
    you should probably use a 'MultitaskBatchSampler', which can be created
    using 'Tasks.make_batch_sampler()'

    :param task_datasets: Mapping task name -> data
    :param part: General part name for unfilled in 'parts'
    :param parts: Mapping task name -> part name
    :param batch_size: General batch size for unfilled in 'batch_sizes'
    :param batch_sizes: Mapping task name -> batch size
    :param shuffle_data: General shuffling data for unfilled in 'shuffle_task_data'
    :param shuffle_task_data: Mapping task name -> shuffle data?
    :param shuffle_batches: Determines whether batches are shuffled (batches of all tasks in set)
    :param columns: Columns to be prepared for training with pytorch (* => torch.Tensor)
    :param finite: If false => just yields all the data a single time without repeats, else infinite iterator
    """
    task_datasets: Dict[str, Data]
    part: Optional[str] = None
    parts: Optional[Dict[str, str]] = None
    batch_size: Optional[int] = None
    batch_sizes: Optional[Dict[str, int]] = None
    shuffle_data: bool = True
    shuffle_task_data: Optional[Dict[str, bool]] = None
    shuffle_batches: bool = True
    columns: Union[List[str], Tuple[str], Set[str]] = ("input_ids", "attention_mask", "labels")
    finite: bool = True

    def __post_init__(self):
        if not isinstance(self.finite, bool):
            raise TypeError("Wrong type for 'finite', must be an instance of bool")

        if not isinstance(self.columns, (list, tuple, set)):
            raise TypeError("Wrong type for 'columns', must be an instance of list | tuple | set")

        if not isinstance(self.task_datasets, dict):
            raise TypeError("Wrong type for 'task_dataset', must be an instance of dict[str, Data]")

        task_names = set(self.task_datasets.keys())

        # Checks for non configured task specific fields
        if self.batch_sizes is None:
            self.batch_sizes = {}
        if self.shuffle_task_data is None:
            self.shuffle_task_data = {}
        if self.parts is None:
            self.parts = {}

        # Validating & figuring out all params pairs
        # which are in format of <general> -> <specific for task>
        # in one loop
        for task in task_names:

            # -> Batch sizes
            if task not in self.batch_sizes:
                if self.batch_size is None:
                    raise ValueError("Expected to have either batch size for all tasks in "
                                     "'batch_sizes' and 'batch_size': Optional[Any] or"
                                     "skipped some tasks in 'batch_sizes' with default in 'batch_size' or"
                                     "'batch_sizes' = None, 'batch_size': int "
                                     "for all tasks to have the same batch size")
                elif not isinstance(self.batch_size, int):
                    raise TypeError("Wrong type for 'batch_size', must be an instance of 'int'")
                else:
                    self.batch_sizes[task] = self.batch_size
            elif not isinstance(self.batch_sizes[task], int):
                raise TypeError("Wrong type for 'batch_sizes' values, must be instances of 'int'")

            # -> Parts
            if task not in self.parts:
                if self.parts is None:
                    raise ValueError("TODO text")
                elif not isinstance(self.part, str):
                    raise TypeError("Wrong type for 'part', must be an instance of 'str'")
                else:
                    self.parts[task] = self.part
            elif not isinstance(self.parts[task], str):
                raise TypeError("Wrong type for 'parts' values, must be instances of 'str'")

            # -> Shuffling
            if task not in self.shuffle_task_data:
                if self.shuffle_data is None:
                    raise ValueError("TODO text")
                elif not isinstance(self.shuffle_data, bool):
                    raise TypeError("Wrong type for 'shuffle_data', must be an instance of 'bool'")
                else:
                    self.shuffle_task_data[task] = self.shuffle_data
            elif not isinstance(self.shuffle_task_data[task], bool):
                raise TypeError("Wrong type for 'shuffle_task_data' values, must be instances of 'bool'")

        # Validating shuffle_shuffle batches
        if not isinstance(self.shuffle_batches, bool):
            raise TypeError("Wrong type for 'shuffle_batches', must be an instance of bool")

        # Making batches for all future epoch-iterations - __iter__ calls
        self._batches = []
        for task, data in self.task_datasets.items():
            batch_size = self.batch_sizes[task]
            shuffle = self.shuffle_task_data[task]
            part = self.parts[task]
            for batch in data.make_data_loader(part, batch_size, shuffle, self.columns, finite=True):
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
    """
    Everything necessary for training on the batch of data of some task

    :param name: Task name for logging other stuff
    :param data: Collated batch of data
    :param model_head: Model for the task, which can be used as a regular 'PreTrainedModel' in train / eval loops
    :param metrics: Optional metrics func
    """
    name: str
    data: Dict[str, Any]
    model_head: PreTrainedModel
    metrics: Optional[Callable] = None

    def data_on_device(self, device) -> Dict[str, Any]:
        """
        Moves data to device
        :param device: PyTorch device like 'torch.device("cuda")'
        :return: Data on target device
        """
        data = {k: v.to(device) for k, v in self.data.items()}
        return data

    def __call__(self, *args, **kwargs):
        """
        Call forward pass on the encoder + model head
        :param args: As in default forward pass
        :param kwargs: As in default forward pass
        :return: Model outputs
        """
        return self.model_head(*args, **kwargs)

    def self_forward(self, device, *args, **kwargs):
        return self(self.data_on_device(device), *args, **kwargs)


@dataclass
class MultitaskBatchSampler:
    """
    Batch sampler for multitask learning

    :param tasks: 'Tasks' instance
    :param part: General part name for unfilled in 'parts'
    :param parts: Mapping task name -> part name
    :param batch_size: General batch size for unfilled in 'batch_sizes'
    :param batch_sizes: Mapping task name -> batch size
    :param shuffle_data: General shuffling data for unfilled in 'shuffle_task_data'
    :param shuffle_task_data: Mapping task name -> shuffle data?
    :param shuffle_batches: Determines whether batches are shuffled (batches of all tasks in set)
    :param columns: Columns to be prepared for training with pytorch (* => torch.Tensor)
    :param finite: If false => just yields all the data a single time without repeats, else infinite iterator
    """
    # Actually tasks: Tasks, but can not import it,
    # because import loop will happen if data_loaders imports tasks
    # and for tasks it's necessary to import data_loaders
    tasks: Any
    part: Optional[str] = None
    parts: Optional[Dict[str, str]] = None
    batch_size: Optional[int] = None
    batch_sizes: Optional[Dict[str, int]] = None
    shuffle_data: bool = True
    shuffle_task_data: Optional[Dict[str, bool]] = None
    shuffle_batches: bool = True
    columns: Union[List[str], Tuple[str], Set[str]] = ("input_ids", "attention_mask", "labels")
    finite: bool = True

    def __post_init__(self) -> None:
        config = self.__dict__.copy()
        del config["tasks"]
        self.data_loader = self.tasks.make_data_loader(**config)
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
