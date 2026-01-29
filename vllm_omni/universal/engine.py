import time
import torch
from typing import Any, List, Dict, Optional, Union
from vllm_omni.logger import init_logger
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.inputs.data import OmniPromptType
from .config import UniversalStageConfig
from .operators.base import Operator

logger = init_logger(__name__)

class UniversalEngine(torch.nn.Module):
    """Engine that executes a sequence of operators.
    
    The engine takes standardized OmniPromptType inputs and returns 
    OmniEngineCoreOutput, ensuring compatibility with the vLLM-Omni 
    inter-stage transfer protocol.
    
    Inherits from nn.Module to align with the architecture of diffusion pipelines.
    """
    
    def __init__(self, config: UniversalStageConfig, operators: List[Operator]):
        super().__init__()
        self.config = config
        self.operators = torch.nn.ModuleList(operators)

    @staticmethod
    def make_engine(config: UniversalStageConfig) -> "UniversalEngine":
        """Factory method to create a UniversalEngine instance, matching DiffusionEngine pattern.
        
        Args:
            config: The configuration for the universal stage.
            
        Returns:
            UniversalEngine: An initialized engine with all operators.
        """
        operators = []
        for i, op_cfg in enumerate(config.operators):
            # Use indexed prefix for weight mapping consistency (e.g., operator_0.)
            prefix = f"operator_{i}."
            operators.append(create_operator(op_cfg, prefix=prefix))
        
        return UniversalEngine(config, operators)

    def forward(self, inputs: OmniPromptType, sampling_params: Any = None, request_id: str = "") -> OmniEngineCoreOutput:
        """Execute the chain of operators."""
        worker_id = getattr(self.config, 'worker_id', 0)
        current_data = inputs
        for op in self.operators:
            try:
                start_time = time.time()
                current_data = op(current_data, sampling_params=sampling_params)
                end_time = time.time()
                logger.debug(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Operator {op.__class__.__name__} took "
                             f"{(end_time - start_time)*1000:.2f}ms")
            except Exception as e:
                logger.error(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Error in operator {op.__class__.__name__}: {e}")
                raise e
        return self._normalize_output(current_data, request_id=request_id)

    def step(self, inputs: OmniPromptType, sampling_params: Any = None, request_id: str = "") -> OmniEngineCoreOutput:
        """Execute a single step, matching the interface of DiffusionEngine."""
        return self.forward(inputs, sampling_params=sampling_params, request_id=request_id)

    def execute(self, inputs: OmniPromptType, sampling_params: Any = None, request_id: str = "") -> OmniEngineCoreOutput:
        """Alias for forward."""
        return self.forward(inputs, sampling_params=sampling_params, request_id=request_id)

    def load_weights(self):
        """Placeholder for loading weights, matching the model loading patterns."""
        logger.info(f"Loading weights for {len(self.operators)} operators")
        pass

    def _normalize_output(self, data: Any, request_id: str = "") -> OmniEngineCoreOutput:
        """Convert operator result into a standardized OmniEngineCoreOutput."""
        # Always provide required args: request_id, new_token_ids (empty for non-text)
        if isinstance(data, OmniEngineCoreOutput):
            if not hasattr(data, "request_id") or not getattr(data, "request_id", None):
                data.request_id = request_id or "dummy-request-id"
            if not hasattr(data, "new_token_ids"):
                data.new_token_ids = []
            return data
        res = OmniEngineCoreOutput(request_id=request_id or "dummy-request-id", new_token_ids=[])
        # Always ensure multimodal_outputs exists
        res.multimodal_outputs = {}
        if isinstance(data, torch.Tensor):
            res.pooling_output = data
        elif isinstance(data, dict):
            is_tensor_dict = all(isinstance(v, torch.Tensor) for v in data.values())
            res.multimodal_outputs = data
        else:
            res.multimodal_outputs = {"result": data}
        return res

def create_operator(operator_config: Dict[str, Any], prefix: str = "") -> Operator:
    """Factory to create an operator from config."""
    import importlib
    cls_path = operator_config.get("class")
    if not cls_path:
        raise ValueError("Operator config must contain 'class' field")
    
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(operator_config.get("config", {}), prefix=prefix)
