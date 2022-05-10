"""
Tools for solving sequence clasification tasks
"""

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
from .utils import validate_isinstance, validate_isinstance_multi


def classification_metrics(
        logits: np.ndarray,
        labels: np.ndarray,
        average: Optional[str] = None,
        validate: bool = True
) -> Dict[str, float]:
    """
    Computing a set of simple classic metrics for classification tasks

    :param logits: Model ouputs - (not normalized) predictions (detached from PyTorch)
    :param labels: Corresponding class labels for each sample
    :param average: Averaging technique for N-class problems, N > 2
    """
    validate_isinstance(logits, np.ndarray, "logits", validate=validate)
    validate_isinstance(labels, np.ndarraym, "labels", validate=validate)
    validate_isinstance(
        average, str, "average", validate=validate,
        value_set={"micro", "macro", "weighted", "samples"}
    )
    predictions = np.argmax(logits, axis=-1)
    if average is None:
        num_labels = len(np.unique(labels))
        if num_labels == 2:
            average = "binary"
        elif num_labels > 2:
            average = "micro"
        else:
            raise ValueError("Expected 'labels' to contain binary / n-class labels")
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average=average),
        "recall": recall_score(labels, predictions, average=average),
        "f1": f1_score(labels, predictions, average=average)
    }


class NFeedForwardsHead(nn.Module):
    """
    Same implementation of classifier as in 'transformers', but using N linear layers
    """

    def __init__(
            self,
            num_layers: int = 1,
            dropout_in: float = 0.1,
            dropout_between: float = 0,
            output_size: int = 2,
            dim_emb: int = 768,
            linear_hidden_size: Optional[int] = None,
            activation_fun: Optional[Callable] = F.relu,
            validate: bool = True
    ) -> None:
        """
        :param num_layers: Number of linear layers used, including input & output (in case of 1 - basic softmax classifier)
        :param dropout_in: Dropout rate applied to incomming embeddings
        :param dropout_between: Dropout rate between linear layers
        :param output_size: Number of dimensions of ouputs (usually = number of classes)
        :param dim_emb: Embeddings shape (dimensions size)
        :param linear_hidden_size: Number of dimensions in linear layers
        :param activation_fun: Callable activation between linear layers
        """
        super().__init__()

        self.output_size = validate_isinstance(output_size, int, "output_size", validate=validate)
        self.num_layers = validate_isinstance(num_layers, int, "num_layers", validate=validate)
        if self.num_layers < 1:
            raise ValueError("'num_layers' must be >= 1")
        validate_isinstance(dropout_in, float, "dropout")
        validate_isinstance(dim_emb, int, "encoder_hidden_size")
        if linear_hidden_size is None:
            linear_hidden_size = dim_emb
        else:
            validate_isinstance(linear_hidden_size, int, "hidden_size")

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_between = nn.Dropout(validate_isinstance(dropout_between, float, "dropout_between"))
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(nn.Linear(dim_emb, self.output_size))
        else:
            self.layers.append(nn.Linear(dim_emb, linear_hidden_size))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Linear(linear_hidden_size, linear_hidden_size))
            self.layers.append(nn.Linear(linear_hidden_size, self.output_size))

        self.activation_fun = validate_isinstance(activation_fun, Callable, "activation_fun")

    def forward(
            self,
            encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions,
            labels: Optional[torch.LongTensor] = None,
    ):
        """
        :param encoder_outputs: Transormer outputs, including [CLS] token (1st index)
        :param labels: Corresponding class labels for each sample
        """
        pooled_output = encoder_outputs[1]
        x = self.dropout_in(pooled_output)

        for i in range(self.num_layers - 1):
            x = self.layers[i].forward(x)
            if self.activation_fun is not None:
                x = self.activation_fun(x)
            x = self.dropout_between.forward(x)
        logits = self.layers[self.num_layers - 1].forward(x)

        # Determining loss type & computing it
        loss = None
        if labels is not None:
            if self.output_size == 1:
                # regression
                loss_fun = nn.MSELoss()
                loss = loss_fun(logits.squeeze(), labels.squeeze()) if self.output_size == 1 else loss_fun(logits,
                                                                                                           labels)
            elif self.output_size > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                # single label classification
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.output_size), labels.view(-1))
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
class ClassificationTask:
    """
    Default sequence classification implementation, based on transformers.AutoModelForSequenceClassification,
    which adds a linear layer <model_outputs> -> <num_classes> with dropout and softmax on top
    with cross-entropy loss automatic computation
    """
    name: str
    dataset_dict: DatasetDict
    num_labels: int = 2
    metrics: Optional[Callable] = classification_metrics
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
            compute_metrics=self.metrics
        )


@dataclass
class NFFClassificationTask:
    """
    Default sequence classification implementation, based on transformers.AutoModelForSequenceClassification,
    which adds a linear layer <model_outputs> -> <num_classes> with dropout and softmax on top
    with cross-entropy loss automatic computation
    """
    name: str
    dataset_dict: DatasetDict
    num_layers: int = 3
    num_labels: int = 2
    metrics: Optional[Callable] = classification_metrics
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None
    dropout_in: float = 0.1
    dropout_between: float = 0.0
    hidden_size: Optional[int] = None
    activation_fun: Optional[Callable] = F.relu
    validate: bool = True

    def __post_init__(self) -> None:
        validate_isinstance_multi([
            (self.name, str, "name"),
            (self.dataset_dict, DatasetDict, "dataset_dict"),
            (self.num_layers, int, "num_layers"),
            (self.num_labels, int, "num_labels"),
            (self.metrics, Callable, "metrics", True),
            (self.preprocessor, Preprocessor, "preprocessor", True),
            (self.tokenizer_config, TokenizerConfig, "tokenizer_config", True),
            (self.dropout_in, (int, float), "dropout_in"),
            (self.dropout_between, (int, float), "dropout_between"),
            (self.hidden_size, int, "hidden_size", True),
            (self.activation_fun, Callable, "activation_fun", True),
        ], validate=self.validate)

    def to_task(self, model_path: str) -> Task:
        """
        Create a 'Task' instance, using model name / path to initialize
        :param model_path: Model name or path for Hugging face Transformers initialization
        """
        tokenizer_config = TokenizerConfig() if self.tokenizer_config is None else self.tokenizer_config
        return Task(
            name=self.name,
            head=NFeedForwardsHead(
                num_layers=self.num_layers,
                dropout_in=self.dropout_in,
                dropout_between=self.dropout_between,
                output_size=self.num_labels,
                # TODO: Replace this somehow, so that it does not require doing full model loading
                dim_emb=AutoModel.from_pretrained(model_path).config.hidden_size,
                linear_hidden_size=self.hidden_size,
                activation_fun=self.activation_fun
            ),
            data=Data(
                dataset_dict=self.dataset_dict,
                configured_tokenizer=tokenizer_config.make_configured_tokenizer(model_path),
                preprocessor=self.preprocessor
            ),
            compute_metrics=self.metrics
        )
