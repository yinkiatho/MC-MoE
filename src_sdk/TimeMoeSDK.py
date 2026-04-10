
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .BaseModel import BaseModel
from .BaseConfig import ModelOutput
from utils.utils import load_config, config_to_args  

if TYPE_CHECKING:
    from time_moe.runner import TimeMoeRunner 


class TimeMoeSDK(BaseModel):
    """Thin facade that delegates real work to TimeMoeRunner.

    Lazy-imports TimeMoeRunner inside _get_runner() to break the circular
    import that would occur if we imported runner.py at module level
    (runner.py itself imports from src_sdk at load time).
    
    # Load from HuggingFace Hub and run inference
    sdk = TimeMoeSDK.from_pretrained("kiatkock/MV-Time-MOE")
    output = sdk.predict(input_batch, max_horizon_length=1)
    print(output.predictions)  # inverse-scaled forecasts

    # Fine-tune from a YAML config
    sdk = TimeMoeSDK(model_path="kiatkock/MV-Time-MOE", output_path="./runs")
    sdk.train_model(config_path="config/train.yaml")

    # Fine-tune with inline kwargs
    sdk.train_model(
        args=config_namespace,
        from_scratch=False,
        micro_batch_size=8,
        num_train_epochs=3,
        learning_rate=1e-4,
    )
    """

    def __init__(
        self,
        model_path: str = None,
        output_path: str = "logs/time_moe",
        seed: int = 9899,
    ):
        super().__init__(model_path, output_path, seed)
        self._runner = None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_runner(self):
        """Lazily import and instantiate TimeMoeRunner."""
        if self._runner is None:
            # Deferred to break the circular import: runner.py imports src_sdk
            # at load time, so we must not import runner at this module's level.
            
            self._runner = TimeMoeRunner(
                model_path=self.model_path,
                output_path=self.output_path,
                seed=self.seed,
            )
        return self._runner

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        output_path: str = "logs/time_moe",
        seed: int = 9899,
        **kwargs,
    ) -> "TimeMoeSDK":
        """Load a model from a HuggingFace Hub ID or local directory.

        Args:
            model_path:  HF Hub ID (e.g. "kiatkock/MV-Time-MOE") or local path.
            output_path: Directory for training artefacts / checkpoints.
            **kwargs:    Forwarded to load_model() — e.g. ``attn_implementation``,
                         ``input_size``, ``channel_configs``.
        """
        instance = cls(model_path=model_path, output_path=output_path, seed=seed)
        instance.load_model(**kwargs)
        return instance

    # ------------------------------------------------------------------ #
    # BaseModel implementation                                             #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs) -> Any:
        """Load model weights and attach them to the runner.

        Keyword args are forwarded directly to TimeMoeRunner.load_model(),
        e.g. ``attn_implementation='eager'``, ``input_size=7``,
        ``channel_configs=[...]``, ``from_scratch=False``.
        """
        runner = self._get_runner()
        # Note: runner.load_model uses the misspelled kwarg 'from_scatch'
        from_scratch = kwargs.pop("from_scratch", False)
        model = runner.load_model(
            model_path=self.model_path,
            from_scatch=from_scratch,
            **kwargs,
        )
        runner.model = model
        return model

    def predict(
        self,
        input_batch: Dict[str, Any],
        max_horizon_length: int = 1,
        device=None,
    ) -> ModelOutput:
        """Run inference on a prepared batch.

        Args:
            input_batch:         Dict containing at minimum ``input_ids`` tensor
                                 plus optional ``channel_*`` tensors and
                                 ``mean`` / ``std`` tensors for inverse scaling.
            max_horizon_length:  Number of forecast steps to generate.
            device:              Optional device override (e.g. ``'cuda:0'``).

        Returns:
            :class:`ModelOutput` with ``predictions`` holding the inverse-scaled
            forecast tensor stored in ``input_batch['model_prediction_sequence']``.
        """
        runner = self._get_runner()
        return runner.predict(
            input_batch=input_batch,
            model=runner.model,
            device=device,
            model_path=self.model_path,
            max_horizon_length=max_horizon_length,
        )

    def train_model(
        self,
        config_path: Optional[str] = None,
        from_scratch: bool = False,
        optuna_search: bool = False,
        **kwargs,
    ) -> Any:
        """Fine-tune or pretrain the model.

        Args:
            config_path:    Optional path to a YAML training config file.
                            Loaded and converted to a Namespace that is passed
                            as ``args`` to the runner. Explicit ``**kwargs``
                            take precedence over file values.
            from_scratch:   Randomly initialise weights from the architecture
                            config at ``model_path`` instead of loading
                            pretrained weights.
            optuna_search:  Enable Optuna hyperparameter search.
            **kwargs:       Training config keys forwarded to
                            TimeMoeRunner.train_model() — e.g.
                            ``micro_batch_size``, ``num_train_epochs``,
                            ``learning_rate``, ``args`` (Namespace).
        """
        if config_path is not None:
            
            cfg = load_config(config_path)
            args = config_to_args(cfg)
            kwargs.setdefault("args", args)

        runner = self._get_runner()
        return runner.train_model(
            from_scratch=from_scratch,
            optuna_search=optuna_search,
            **kwargs,
        )

    def batch_predict(
        self,
        batches: List[Dict[str, Any]],
        max_horizon_length: int = 1,
        device=None,
    ) -> List[ModelOutput]:
        """Run predict() over a list of input_batch dicts.

        Args:
            batches:             List of input_batch dicts (same format as predict()).
            max_horizon_length:  Applied uniformly to all batches.
            device:              Optional device override.
        """
        return [
            self.predict(batch, max_horizon_length=max_horizon_length, device=device)
            for batch in batches
        ]
