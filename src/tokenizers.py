from dataclasses import dataclass, field
from transformers.file_utils import PaddingStrategy, TensorType
from typing import Any, Union, Optional, Dict, List, Callable
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, \
    TruncationStrategy, TextInput, TextInputPair, PreTokenizedInput, \
    PreTokenizedInputPair, EncodedInput, EncodedInputPair, BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer


TokenizerInputT = Dict[str, Union[
    List[TextInput], List[TextInputPair],
    List[PreTokenizedInput], List[PreTokenizedInputPair],
    List[EncodedInput], List[EncodedInputPair]
]]
TokenizerT = Union[PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class ConfiguredTokenizer:
    model_path: str
    tokenizer: TokenizerT = field(init=False)
    _tokenizer: Callable[[TokenizerInputT], BatchEncoding] = field(init=False)
    init_config: Optional[Dict[str, Any]] = None
    add_special_tokens: bool = True
    padding: Union[bool, str, PaddingStrategy] = False
    truncation: Union[bool, str, TruncationStrategy] = True
    max_length: Optional[int] = None
    stride: int = 0
    is_split_into_words: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: Optional[Union[str, TensorType]] = None
    return_token_type_ids: Optional[bool] = None
    return_attention_mask: Optional[bool] = None
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.init_config is None:
            self.init_config = {}
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, **self.init_config)
        self.tokenizer = tokenizer

        def configured_tokenizer(batch: TokenizerInputT) -> BatchEncoding:
            # TODO: Provide customization for input label
            features = self.tokenizer.batch_encode_plus(batch["input"], **self.call_config)
            features["labels"] = batch["label"]
            return features

        self._tokenizer = configured_tokenizer

    @property
    def call_config(self) -> Dict[str, Any]:
        call_config = self.__dict__.copy()
        del call_config["model_path"]
        del call_config["init_config"]
        del call_config["tokenizer"]
        del call_config["_tokenizer"]
        return call_config

    def __call__(self, batch_text_or_text_pairs: TokenizerInputT) -> BatchEncoding:
        return self._tokenizer(batch_text_or_text_pairs)

    def encode(self, batch_text_or_text_pairs: TokenizerInputT) -> BatchEncoding:
        return self(batch_text_or_text_pairs)


@dataclass
class TokenizerConfig:
    init_config: Optional[Dict[str, Any]] = None
    add_special_tokens: bool = True
    padding: Union[bool, str, PaddingStrategy] = False
    truncation: Union[bool, str, TruncationStrategy] = True
    max_length: Optional[int] = None
    stride: int = 0
    is_split_into_words: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: Optional[Union[str, TensorType]] = None
    return_token_type_ids: Optional[bool] = None
    return_attention_mask: Optional[bool] = None
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    verbose: bool = True

    def make_configured_tokenizer(self, model_path: str) -> ConfiguredTokenizer:
        return ConfiguredTokenizer(model_path=model_path, **self.__dict__.copy())

