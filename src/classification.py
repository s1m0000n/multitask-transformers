from dataclasses import dataclass
from typing import Optional, Callable, Dict

import numpy as np
import torch
from datasets import DatasetDict
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

from .data import Data
from .models import HFHead
from .preprocessing import Preprocessor
from .tasks import Task
from .tokenizers import TokenizerConfig
from .utils import validate_isinstance


def sequence_classification_compute_metrics(logits: np.ndarray, labels: np.ndarray, average_f1: str = "binary") -> Dict[str, float]:
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions, average=average_f1)
    }


class NLinearsHead(nn.Module):
    """
    Same implementation as in 'transformers', but using N linear layers
    """

    def __init__(
            self,
            num_layers: int = 1,
            dropout: float = 0.1,
            output_size: int = 2,
            encoder_hidden_size: int = 768,
            linear_hidden_size: Optional[int] = None,
            activation_fun: Optional[Callable] = F.relu
    ) -> None:
        super().__init__()

        self.num_layers = validate_isinstance(num_layers, int, "num_layers")
        if self.num_layers < 1:
            raise ValueError("'num_layers' must be >= 1")
        validate_isinstance(dropout, float, "dropout")
        validate_isinstance(output_size, int, "output_size")
        validate_isinstance(encoder_hidden_size, int, "encoder_hidden_size")
        if linear_hidden_size is None:
            linear_hidden_size = encoder_hidden_size
        else:
            validate_isinstance(linear_hidden_size, int, "linear_hidden_size")

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(nn.Linear(encoder_hidden_size, output_size))
        else:
            self.layers.append(nn.Linear(encoder_hidden_size, linear_hidden_size))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Linear(linear_hidden_size, linear_hidden_size))
            self.layers.append(nn.Linear(linear_hidden_size, output_size))

        self.activation_fun = validate_isinstance(activation_fun, Callable, "activation_fun")

    def forward(
            self,
            encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions,
            labels: Optional[torch.LongTensor] = None,
    ):
        pooled_output = encoder_outputs[1]
        x = self.dropout(pooled_output)

        for i in range(self.num_layers - 1):
            if self.activation_fun is None:
                x = self.layers[i].forward(x)
            else:
                x = self.activation_fun(self.layers[i].forward(x))
        logits = self.layers[self.num_layers - 1].forward(x)

        # Determining loss type & computing it
        loss = None
        if labels is not None:
            num_labels = len(torch.unique(labels))
            if num_labels == 1:
                # regression
                loss_fun = nn.MSELoss()
                loss = loss_fun(logits.squeeze(), labels.squeeze()) if num_labels == 1 else loss_fun(logits, labels)
            elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                # single label classification
                loss = nn.CrossEntropyLoss()(logits.view(-1, num_labels), labels.view(-1))
            else:
                # multi label classification
                loss = nn.BCEWithLogitsLoss()(logits, labels)

        # Creating an output like in default implementation
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@dataclass
class SequenceClassificationTask:
    """
    Default sequence classification implementation, based on transformers.AutoModelForSequenceClassification,
    which adds a linear layer <model_outputs> -> <num_classes> with dropout and softmax on top
    with cross-entropy loss automatic computation
    """
    name: str
    dataset_dict: DatasetDict
    num_labels: int = 2
    compute_metrics: Optional[Callable] = sequence_classification_compute_metrics
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None

    def to_task(self, model_path: str) -> Task:
        tokenizer_config = TokenizerConfig() if self.tokenizer_config is None else self.tokenizer_config
        return Task(
            name=self.name,
            head=HFHead(
                class_=AutoModelForSequenceClassification,
                config_params={"num_labels": self.num_labels},
            ),
            data=Data(
                dataset_dict=self.dataset_dict,
                configured_tokenizer=tokenizer_config.make_configured_tokenizer(model_path),
                preprocessor=self.preprocessor
            ),
            compute_metrics=self.compute_metrics
        )


@dataclass
class NLinearsSequenceClassificationTask:
    """
    Default sequence classification implementation, based on transformers.AutoModelForSequenceClassification,
    which adds a linear layer <model_outputs> -> <num_classes> with dropout and softmax on top
    with cross-entropy loss automatic computation
    """
    name: str
    dataset_dict: DatasetDict
    num_layers: int = 3
    num_labels: int = 2
    compute_metrics: Optional[Callable] = sequence_classification_compute_metrics
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None
    dropout: float = 0.1
    linear_hidden_size: Optional[int] = None
    activation_fun: Optional[Callable] = F.relu

    def to_task(self, model_path: str) -> Task:
        tokenizer_config = TokenizerConfig() if self.tokenizer_config is None else self.tokenizer_config
        return Task(
            name=self.name,
            head=NLinearsHead(
                num_layers=self.num_layers,
                dropout=self.dropout,
                output_size=self.num_labels,
                # TODO: Replace this somehow, so that it does not require doing full model loading
                encoder_hidden_size=AutoModel.from_pretrained(model_path).config.hidden_size,
                linear_hidden_size=self.linear_hidden_size,
                activation_fun=self.activation_fun
            ),
            data=Data(
                dataset_dict=self.dataset_dict,
                configured_tokenizer=tokenizer_config.make_configured_tokenizer(model_path),
                preprocessor=self.preprocessor
            ),
            compute_metrics=self.compute_metrics
        )
