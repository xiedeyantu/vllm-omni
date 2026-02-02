import time
import torch
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
from vllm_omni.logger import init_logger
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.inputs.data import OmniPromptType
from .config import UniversalStageConfig
from .operators.base import Operator

logger = init_logger(__name__)


@dataclass
class UniversalEngineRequest:
    """Internal request object for UniversalEngine.
    
    Encapsulates batch processing parameters, similar to OmniDiffusionRequest.
    
    Attributes:
        prompts: List of prompt data dicts to process
        sampling_params: Sampling/processing parameters
        request_ids: List of request IDs corresponding to each prompt
    """
    prompts: List[OmniPromptType]
    sampling_params: Any
    request_ids: List[str]

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
        """Execute the chain of operators.
        
        Each operator receives a request object and processes it, returning OmniRequestOutput.
        The final operator output is converted back to OmniEngineCoreOutput for stage compatibility.
        """
        from vllm_omni.outputs import UniversalStageRequest, OmniRequestOutput
        
        worker_id = getattr(self.config, "worker_id", 0)
        
        # Create initial request object
        current_request = UniversalStageRequest.from_api_request(
            request_id=request_id,
            stage_id=self.config.stage_id,
            data=inputs
        )
        
        us_output = {}  # Store outputs from each operator
        
        for op in self.operators:
            try:
                start_time = time.time()
                
                # Call operator with request object, get OmniRequestOutput
                op_output = op(current_request)
                
                end_time = time.time()
                
                op_name = op.__class__.__name__
                logger.debug(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Operator {op_name} took "
                             f"{(end_time - start_time)*1000:.2f}ms")
                
                # Store operator output
                us_output[op_name] = op_output
                
                # Convert OmniRequestOutput data back to UniversalStageRequest for next operator
                # Extract the data from multimodal_output
                if isinstance(op_output, OmniRequestOutput):
                    # Use multimodal_output as data for next operator
                    next_data = op_output.multimodal_output or op_output.data
                else:
                    next_data = op_output
                
                # Create request for next operator
                current_request = UniversalStageRequest.from_api_request(
                    request_id=request_id,
                    stage_id=self.config.stage_id,
                    data=next_data
                )
                
            except Exception as e:
                logger.error(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Error in operator {op.__class__.__name__}: {e}")
                raise e
        
        # Convert final request back to OmniEngineCoreOutput
        return self._normalize_output(current_request.data, request_id=request_id, us_output=us_output)

    def step(self, request: Union[UniversalEngineRequest, OmniPromptType] = None, 
             sampling_params: Any = None, request_id: str = "") -> Union[List[OmniEngineCoreOutput], OmniEngineCoreOutput]:
        """Execute step with batch request object or single prompt.
        
        Matches DiffusionEngine.step() interface for batch processing.
        Accepts either:
        - request: UniversalEngineRequest (batch processing)
        - request (as single prompt) + sampling_params + request_id (backward compatibility)
        
        Args:
            request: Either UniversalEngineRequest for batch or single prompt for backward compatibility
            sampling_params: Only used in backward compatibility mode
            request_id: Only used in backward compatibility mode
            
        Returns:
            List[OmniEngineCoreOutput] for batch requests, or OmniEngineCoreOutput for single requests
        """
        # Batch processing mode: request is UniversalEngineRequest
        if isinstance(request, UniversalEngineRequest):
            # Process entire batch through operator chain
            return self._process_batch(request)
        
        # Backward compatibility mode: request is single prompt
        else:
            return self.forward(request, sampling_params=sampling_params, request_id=request_id)
    
    def _process_batch(self, request: UniversalEngineRequest) -> List[OmniEngineCoreOutput]:
        """Process entire batch through operator chain.
        
        Operators receive the full UniversalEngineRequest and return OmniRequestOutput.
        """
        from vllm_omni.outputs import OmniRequestOutput
        
        worker_id = getattr(self.config, "worker_id", 0)
        
        # Start with the batch request
        current_request = request
        
        # Collect all operator outputs to distribute per-request later
        all_op_outputs = {} # op_name -> OmniRequestOutput
        
        for op in self.operators:
            try:
                start_time = time.time()
                
                # Call operator with batch request, get OmniRequestOutput
                op_output = op(current_request)
                
                end_time = time.time()
                
                op_name = op.__class__.__name__
                logger.debug(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Operator {op_name} took "
                             f"{(end_time - start_time)*1000:.2f}ms")
                
                # Store operator output
                all_op_outputs[op_name] = op_output
                
                # Convert OmniRequestOutput to UniversalEngineRequest for next operator
                if isinstance(op_output, OmniRequestOutput):
                    # Extract results from multimodal_output
                    multimodal_output = op_output.multimodal_output or {}
                    
                    # Build new prompts from results for next operator
                    # Each operator result becomes input to next operator
                    # We look for "results" list which contains data for each request in batch
                    results = multimodal_output.get("results", [multimodal_output])
                    
                    current_request = UniversalEngineRequest(
                        prompts=results,
                        sampling_params=request.sampling_params,
                        request_ids=request.request_ids
                    )
                
            except Exception as e:
                logger.error(f"[Stage-{self.config.stage_id}-Worker-{worker_id}] Error in operator {op.__class__.__name__}: {e}")
                raise e
        
        # Convert final outputs to OmniEngineCoreOutput
        results = []
        final_request = current_request
        
        for i, req_id in enumerate(request.request_ids):
            prompt_data = final_request.prompts[i] if i < len(final_request.prompts) else {}
            
            # Extract this specific request's results from all operator outputs
            req_us_output = {}
            for op_name, op_out in all_op_outputs.items():
                if isinstance(op_out, OmniRequestOutput) and op_out.multimodal_output:
                    # Operators store batch results in op_out.multimodal_output["results"]
                    op_data = op_out.multimodal_output
                    batch_results = op_data.get("results", [])
                    
                    if batch_results and i < len(batch_results):
                        req_us_output[op_name] = batch_results[i]
                    else:
                        # Fallback: if no results list or index out of bounds, maybe it's just a single result
                        req_us_output[op_name] = op_data
            
            output = self._normalize_output(prompt_data, request_id=req_id, us_output=req_us_output)
            results.append(output)
        
        return results

    def execute(self, inputs: OmniPromptType, sampling_params: Any = None, request_id: str = "") -> OmniEngineCoreOutput:
        """Alias for forward."""
        return self.forward(inputs, sampling_params=sampling_params, request_id=request_id)

    def load_weights(self):
        """Placeholder for loading weights, matching the model loading patterns."""
        logger.info(f"Loading weights for {len(self.operators)} operators")
        pass

    def _normalize_output(self, data: Any, request_id: str = "", us_output: dict = None) -> OmniEngineCoreOutput:
        """Convert operator result into a standardized OmniEngineCoreOutput.
        
        Args:
            data: The final output from the operator chain
            request_id: Request ID for tracking
            us_output: Dictionary of operator outputs keyed by operator class name
        """
        if us_output is None:
            us_output = {}
        
        # Always provide required args: request_id, new_token_ids (empty for non-text)
        if isinstance(data, OmniEngineCoreOutput):
            if not hasattr(data, "request_id") or not getattr(data, "request_id", None):
                data.request_id = request_id or "dummy-request-id"
            if not hasattr(data, "new_token_ids"):
                data.new_token_ids = []
            # Store universal stage operator outputs
            data.us_output = us_output
            return data
        
        res = OmniEngineCoreOutput(request_id=request_id or "dummy-request-id", new_token_ids=[])
        # Always ensure multimodal_outputs and us_output exist
        res.multimodal_outputs = {}
        res.us_output = us_output
        
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
