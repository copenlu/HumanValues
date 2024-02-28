from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import Dataset


class ValueDataset(Dataset, ABC):
    """
    Dataset for storing value-annotated text data.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_splits(self):
        """
        Return the indices of the train, validation, and test splits.
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        """
        Should output with keys:
        - text: the text of the example
        - orig_label: the label of the example
        - value: the value we are focusing on
        """
        raise NotImplementedError()