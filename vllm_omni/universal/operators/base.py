import abc
from typing import Any, Dict, List
import torch.nn as nn

class Operator(nn.Module, abc.ABC):
    """Base class for all operators in a UniversalStage.
    
    Mimics the architecture of diffusion models by inheriting from nn.Module
    and supporting weight source tracking.
    """
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.weight_sources: List[Any] = []

    @abc.abstractmethod
    def forward(self, inputs: Any, sampling_params: Any = None) -> Any:
        """Execute the operator logic. 
        
        Aligned with nn.Module's forward method.
        """
        pass

    def run(self, inputs: Any, sampling_params: Any = None) -> Any:
        """Alias for forward to maintain compatibility with UniversalEngine."""
        return self.forward(inputs, sampling_params=sampling_params)
