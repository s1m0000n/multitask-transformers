from dataclasses import dataclass
from typing import Optional, Dict, Callable

import numpy as np
import torch.nn.functional as F
from datasets import DatasetDict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.classification import ClassificationTask, NFFClassificationTask
from src.preprocessing import Preprocessor
from src.tasks import Task
from src.tokenizers import TokenizerConfig
from src.utils import validate_isinstance


def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: Optional[int] = None,
        validate: bool = True
) -> Dict[str, float]:
    validate_isinstance(y_true, np.ndarray, "y_true", validate=validate)
    validate_isinstance(y_pred, np.ndarray, "y_pred", validate=validate)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    result = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
        "rmsle": np.log(rmse),
        "r2": r2
    }
    if k is not None:
        validate_isinstance(k, int, "k", validate=validate, check=lambda x: x > 0)
        result["adj_r2"] = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return result


@dataclass(frozen=True)
class RegressionTask:
    """
    Same as `ClassificationTask(..., num_labels = 1,...)` and default
    `metrics = sequence_regression_compute_metrics`
    """
    name: str
    dataset_dict: DatasetDict
    metrics: Optional[Callable] = regression_metrics
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None

    def to_task(self, model_path: str) -> Task:
        return ClassificationTask(
            self.name, self.dataset_dict,
            num_labels=1,
            metrics=self.metrics,
            preprocessor=self.preprocessor,
            tokenizer_config=self.tokenizer_config
        ).to_task(model_path)

@dataclass
class NFFRegressionTask:
    name: str
    dataset_dict: DatasetDict
    num_layers: int = 5
    metrics: Optional[Callable] = regression_metrics
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None
    dropout_in: float = 0.1
    dropout_between: float = 0.0
    hidden_size: Optional[int] = None
    activation: Optional[Callable] = F.relu

    def to_cls_task(self) -> NFFClassificationTask:
        return NFFClassificationTask(
            self.name, self.dataset_dict,
            num_layers=self.num_layers,
            num_labels=1,
            metrics=self.metrics,
            preprocessor=self.preprocessor
        )
    def to_task(self, encoder_path: str) -> Task:
        return self.to_cls_task().to_task(encoder_path)