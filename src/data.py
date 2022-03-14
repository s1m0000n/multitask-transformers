"""
Storing and processing task data
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict, Iterable

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding

from .preprocessing import Preprocessor
from .tokenizers import ConfiguredTokenizer, TokenizerT, TokenizerInputT
from .utils import validate_isinstance


@dataclass
class Data:
    dataset_dict: DatasetDict
    configured_tokenizer: ConfiguredTokenizer
    columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    preprocessor: Optional[Preprocessor] = None
    collator_class: Any = DataCollatorWithPadding
    collator_config: Optional[Dict[str, Any]] = None

    collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        validate_isinstance(self.dataset_dict, DatasetDict, "data")
        if self.preprocessor is not None:
            self.dataset_dict = self.preprocessor(self.dataset_dict)
        collator_cfg = {} if self.collator_config is None else self.collator_config
        self.collator = self.collator_class(self.tokenizer, **collator_cfg)
        if isinstance(self.columns, str):
            raise TypeError("Wrong type for 'columns', must be an iterable of 'str', "
                            "but not 'str' itself, for example 'Tuple[str]' or 'List[str]'")
        self.columns = list(self.columns)

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
        return self.configured_tokenizer.call_config

    def encode(self, batch: TokenizerInputT) -> BatchEncoding:
        return self.configured_tokenizer.encode(batch)

    def keys(self):
        return self.dataset_dict.keys()

    def values(self):
        return self.dataset_dict.values()

    def items(self):
        return self.dataset_dict.items()

    def __len__(self):
        return len(self.dataset_dict)

    def __iter__(self):
        return iter(self.dataset_dict)

    def __getitem__(self, part: str):
        if part not in self.keys():
            raise KeyError(f"Expected 'part' to be in {self.keys()}, but got {part}")
        return self.dataset_dict[part]

    def features(self, part: str) -> Dataset:
        features = self[part].map(self.encode, batched=True)
        features.set_format(type="torch", columns=self.columns)
        return features
