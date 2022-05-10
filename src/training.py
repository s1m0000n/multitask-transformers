from typing import Optional, Dict, Any, Type

import torch
from torch.optim import Optimizer, AdamW

from src.metrics import MultitaskMetricsLog
from src.models import MultitaskModel
from src.dataloaders import MetabatchSampler
from src.utils import validate_isinstance
import logging


class MultitaskTrainer:
    def __init__(
            self,
            model: MultitaskModel,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params=None,
            validate: bool = True
    ) -> None:
        self.validate = validate
        self.model = validate_isinstance(model, MultitaskModel, "model", validate=self.validate)
        self.train_log = MultitaskMetricsLog()
        self.validation_log = MultitaskMetricsLog()
        if optimizer_params is None:
            optimizer_params = {"lr": 5e-5}
        else:
            validate_isinstance(optimizer_params, Dict, "optimizer_params", validate=self.validate)
        # TODO: add optimizer_class validation
        self.optimizer = AdamW(self.model.parameters(), **optimizer_params)
        logging.info("created multitask trainer")

    def train(
            self,
            train_sampler: MetabatchSampler,
            validation_sampler: Optional[MetabatchSampler] = None,
            loss_weights: Optional[Dict[str, Any]] = None,
            autodetect_device: bool = True,
            device: Optional[torch.device] = None,
            num_epochs: int = 1,
            num_warmup_steps: Optional[int] = None,
            warmup_steps_part: Optional[float] = 0.1,
    ) -> MultitaskModel:
        logging.info("starting training")
        validate_isinstance(train_sampler, MetabatchSampler, "train_sampler")
        num_training_steps = len(train_sampler) * validate_isinstance(num_epochs, int, "num_epochs", validate=self.validate)
        if num_warmup_steps is None:
            if warmup_steps_part is None:
                num_warmup_steps = 0
                logging.info("warmup steps not set, using 0 by default")
            else:
                num_warmup_steps = int(num_training_steps * warmup_steps_part)
        else:
            validate_isinstance(num_warmup_steps, int, "num_warmup_steps", validate=self.validate)
        if device is None:
            if autodetect_device:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                logging.info(f"auto detected device {device}")
            else:
                device = torch.device("cpu")
                logging.info(f"device auto detection is off and not set explicitly, so device set to CPU")
        else:
            validate_isinstance(device, torch.device, "device", validate=self.validate)
        task_names = set(train_sampler.datasets.keys())
        if self.validate:
            for name in validation_sampler.datasets.keys():
                if name not in task_names:
                    raise KeyError("Found an ")
        return self.model