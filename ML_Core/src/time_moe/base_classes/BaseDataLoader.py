from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseDataset(ABC):

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> Any: ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class BaseDataLoader(ABC):

    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def __len__(self) -> int: ...
