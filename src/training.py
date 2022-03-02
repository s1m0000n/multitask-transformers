import numpy as np
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from transformers import Trainer


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        """
        Just doing nothing, called by transformers.Trainer
        """
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
    """
    Multitask alternative of transformers.Trainer
    """

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
                collate_fn=self.data_collator,
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
                collate_fn=self.data_collator,
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
        mt_dataloader.batch_size = self.args.eval_batch_size
        return mt_dataloader
