from abc import ABC, abstractmethod
from typing import Iterator


class BaseDataLoader(ABC):

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
