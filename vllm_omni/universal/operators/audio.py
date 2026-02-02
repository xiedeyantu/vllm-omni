from typing import Any, Dict, TYPE_CHECKING
import time
import random
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.logger import init_logger
from .base import Operator

if TYPE_CHECKING:
    from vllm_omni.universal.engine import UniversalEngineRequest

logger = init_logger(__name__)

class AudioOperator(Operator):
    """Placeholder operator for audio processing steps."""
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__(config, prefix=prefix)
        self.name = config.get("name", "AudioOperator")

    def forward(self, req: "UniversalEngineRequest") -> OmniRequestOutput:
        """Process audio from batch request."""
        try:
            # Simulate processing time with random sleep 1-3 seconds
            sleep_time = random.uniform(1.0, 3.0)
            logger.info(f"{self.name} sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
            # Process all prompts in batch
            logger.info(f"Running {self.name} on {len(req.request_ids)} requests")
            
            # Build combined output
            output_dict = {}
            if req.prompts:
                # Use first prompt as base
                output_dict = req.prompts[0].copy() if isinstance(req.prompts[0], dict) else {}
            
            # Add operator-specific results
            output_dict["AudioOperator"] = {
                "processed": True,
                "count": len(req.request_ids),
            }
            
            return OmniRequestOutput(
                request_id=req.request_ids[0] if req.request_ids else "unknown",
                finished=True,
                final_output_type="json",
                extra_json=output_dict,
            )
            
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return OmniRequestOutput(
                request_id=req.request_ids[0] if req.request_ids else "unknown",
                finished=True,
                final_output_type="json",
                extra_json={"error": str(e)},
            )
