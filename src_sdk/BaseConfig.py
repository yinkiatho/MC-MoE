from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from transformers import PretrainedConfig


@dataclass
class ModelConfig(PretrainedConfig):
    """Configuration for model initialization"""
    
    def __init__(self, 
                 name_or_path: str = '',
                 output_hidden_states: bool = False,
                 output_attentions: bool = False,
                 return_dict: bool = False,
                 **kwargs):
        
        super().__init__(
            name_or_path,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs
        )
        
            
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**data)
    
    

@dataclass
class ModelOutput:
    """Standardized model output"""
    predictions: Union[np.ndarray, List, Dict]
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None
    
