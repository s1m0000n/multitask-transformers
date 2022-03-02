from typing import Dict, Union, Tuple
from dataclasses import dataclass


@dataclass
class NLIPreprocessor:
    """
    Preprocessing for NLI tasks with fields hypothesis, premise & label
    """
    hypothesis_field: str = "hypothesis"
    premise_field: str = "premise"
    label_field: str = "label"
    input_field: str = "input"
    label_field_in: str = "label"

    def __call__(self, sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
        hypothesis = sample[self.hypothesis_field]
        premise = sample[self.premise_field]
        label = sample[self.label_field_in]
        return {self.input_field: (hypothesis, premise), self.label_field: label}
