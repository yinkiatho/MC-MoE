from abc import ABC, abstractmethod
from typing import Any, List
from .BaseConfig import ModelOutput


class BaseModel(ABC):
    """Abstract base class for HuggingFace model wrappers.

    Provides a common interface for loading, training, and running inference
    with a model. Concrete subclasses (e.g. TimeMoeRunner, TimeMoeSDK)
    implement the abstract methods.
    """

    def __init__(self, model_path: str = None, output_path: str = None, seed: int = 9899):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    @abstractmethod
    def load_model(self, *args, **kwargs) -> Any:
        """Load model weights from a local path or HuggingFace Hub."""
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs) -> Any:
        """Train or fine-tune the model."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> ModelOutput:
        """Run inference on a single input batch."""
        pass

    def batch_predict(self, input_data: List[Any]) -> List[ModelOutput]:
        """Run predict() over a list of inputs. Override for efficiency."""
        return [self.predict(x) for x in input_data]
