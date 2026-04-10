from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
