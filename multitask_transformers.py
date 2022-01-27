import datasets
import torch
import torch.nn as nn
import numpy as np
from typing import *
import multiprocessing as mp
from transformers import PreTrainedModel, PretrainedConfig, Trainer
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DefaultDataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from collections import UserDict


class Task:
    """
    Validates and stores tasks data, config, etc.
    """

    def __init__(self, cls: Any, config: Type[PretrainedConfig],
                 converter_to_features: Callable[[Iterable[Any]], Type[UserDict]],
                 data: datasets.DatasetDict, name: Optional[str] = None) -> None:
        """
        Checking task fields for compatibility with other components of multitask learner
        :param cls:
        :param config:
        :param converter_to_features:
        :param data:
        :param name:
        """
        attr = getattr(cls, "from_pretrained", None)
        if attr:
            assert callable(attr), "cls is expected to have \"from_pretrained\" method"
        self.cls = cls
        assert issubclass(
            PretrainedConfig), "config is expected to be a subclass of PretrainedConfig"
        self.config = config
        msg = "converter_to_features is expected to be a callable, with Iterable[Any] " \
              "arg representing batch and returning features: UserDict | " \
              "transformers.BatchEncoding to be used with forward method of transformers"
        assert callable(converter_to_features), msg
        self.converter = converter_to_features
        msg = "data is expected to be a dict of dataset with splits " \
              "(like train, val, test)"
        assert isinstance(data, datasets.DatasetDict), msg
        self.data = data
        self.name = name or "Untitled"


class MultitaskModel(PreTrainedModel):
    def __init__(self, encoder, task_models):
        super().__init__(PretrainedConfig())
        self.encoder = encoder
        self.task_models = nn.ModuleDict(task_models)

    @classmethod
    def create(cls, base_model_name: str, tasks: Dict[str, Task]):
        """
        Creating a MultitaskModel using the model class (Task.cls) and config (Task.config) from single-task models.
        Creating each single-task model, and having them share the same encoder transformer.
        :param base_model_name: name of encoder model, example: "bert-base-uncased"
        :param tasks: dictionary of tasks with names
        :return: MultitaskModel with initialized shared encoder and task_models
        """
        shared_encoder = None
        task_models = dict()
        for name, task in tasks.items():
            model = task.cls.from_pretrained(base_model_name, config=task.config)
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            task_models[name] = model
        return cls(encoder=shared_encoder, task_models=task_models)

    @classmethod
    def get_encoder_attr_name(cls, model):  # TODO: types, why class method? seems like static method
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        name = model.__class__.__name__
        if name.startswith("Bert"):
            return "bert"
        elif name.startswith("Roberta"):
            return "roberta"
        elif name.startswith("Albert"):
            return "albert"
        else:
            raise NotImplementedError(f"Add support for new model {name}")

    def forward(self, task_name: str, *args, **kwargs):
        """
        Make a forward pass through encoder and a task head

        :param task_name: name of the task
        :param args: *args passed to Model(encoder -> head)
        :param kwargs: **kargs passed to Model(encoder -> head)
        :return: typical results of forward for the model as usual
        """
        return self.task_models[task_name](*args, **kwargs)


def make_features(tasks: Dict[str, Task], num_proc: int = mp.cpu_count(),
                  fields: Iterable[str] = ('input_ids', 'attention_mask', 'labels'), *map_args, **map_kwargs) \
        -> Dict[str, Dict[str, datasets.Dataset]]:
    features = dict()
    for name, task in tasks.items():
        features[name] = dict()
        for split_name, split_dataset in task.data.items():
            features[name][split_name] = split_dataset.map(
                task.converter,
                *map_args,
                batched=True,
                load_from_cache_file=False,
                num_proc=num_proc,
                **map_kwargs
            )
            features[name][split_name].set_format(type="torch", columns=fields)
    return features


def unpack_splits(features: Dict[str, Dict[str, datasets.Dataset]],
                  *split_names: Tuple[str]) -> Tuple[Dict[str, datasets.Dataset]]:
    """
    Separate multitask features into taskwise dict(task_name:dataset) or
    if unpackable tuple of this dicts for each split in *split_names
    :param features: Multitask features dict(for example: make_features(...))
    :param *split_names: Names of splits to aggregate
    :return: Single dict or multiple packed in a tuple for each split
    """
    assert len(split_names), IndexError("Expected at least 1 split name (for example \"train\")")
    result = []
    for split_name in split_names:
        result.append({task_name: dataset[split_name] for task_name, dataset in features.items()})
    return tuple(result) if len(result) > 1 else result[0]


class NLPDataCollator(DefaultDataCollator):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                else:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskName:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name: str, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(Trainer):
    def get_single_train_dataloader(self, task_name: str, train_dataset) -> DataLoaderWithTaskName:
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )
        return DataLoaderWithTaskName(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            ),
        )

    def get_single_eval_dataloader(self, task_name: str, eval_dataset) -> DataLoaderWithTaskName:
        """
        Create a single-task data loader for evaluation that also yields task names
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")
        eval_sampler = (
            RandomSampler(eval_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        return DataLoaderWithTaskName(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=self.data_collator.collate_batch,
            ),
        )

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })

    def get_eval_dataloader(self, eval_dataset) -> MultitaskDataloader:
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        mt_dataloader = MultitaskDataloader({
            task_name: self.get_single_eval_dataloader(task_name, task_dataset)
            for task_name, task_dataset in eval_dataset.items()
        })
        # If this is not set explicitly
        # | AttributeError: 'MultitaskDataloader' object has no attribute 'batch_size'
        # | [issue] https://github.com/s1m0000n/multitask-transformers/issues/2
        mt_dataloader.batch_size = 32
        return mt_dataloader
