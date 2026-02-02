import abc
from typing import Any, Dict, List, TYPE_CHECKING
import torch.nn as nn

from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.universal.engine import UniversalEngineRequest

class Operator(nn.Module, abc.ABC):
    """Base class for all operators in a UniversalStage.
    
    All operators receive UniversalEngineRequest (batch) and return OmniRequestOutput,
    enabling efficient batch processing throughout the pipeline.
    
    Data Flow:
    1. Input: UniversalEngineRequest with prompts (list), request_ids (list), params
    2. Processing: forward() batch-processes all prompts
    3. Output: OmniRequestOutput with combined results for all requests
    
    Example for video operator:
        Input: UniversalEngineRequest(
            prompts=[{"video_paths": [...]}, {"video_paths": [...]}],
            request_ids=["req-1", "req-2"],
            params={...}
        )
        Output: OmniRequestOutput with results for all requests
    """
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.weight_sources: List[Any] = []

    @abc.abstractmethod
    def forward(self, req: "UniversalEngineRequest") -> OmniRequestOutput:
        """Execute the operator logic on batch of requests. 
        
        Args:
            req: UniversalEngineRequest with:
                - prompts: List[Dict] with operator-specific input for each request
                  (e.g., [{"video_paths": [...]}, {"video_paths": [...]}])
                - request_ids: List[str] with unique identifiers for each request
                - sampling_params: sampling parameters (if applicable)
            
        Returns:
            OmniRequestOutput with:
            - request_id: ID for tracking (use first request_id if needed)
            - finished: whether processing completed successfully
            - extra_json: dict containing results for all requests
                        (preserve prompts structure, add operator-specific results)
        
        Implementation Notes:
        - Process all prompts in batch for efficiency
        - Extract data from each prompt in req.prompts
        - Combine results for all requests in the output
        - Preserve all input data in extra_json
        - Include operator-specific results keyed by operator class name
        - Handle errors gracefully (set finished=True with error in extra_json)
        """
        pass

    def run(self, req: "UniversalEngineRequest") -> OmniRequestOutput:
        """Alias for forward to maintain compatibility."""
        return self.forward(req)
