from typing import Any, Dict
import time
import random
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.logger import init_logger
from .base import Operator

logger = init_logger(__name__)

class VisionOperator(Operator):
    """Placeholder operator for vision processing steps."""
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__(config, prefix=prefix)
        self.name = config.get("name", "VisionOperator")

    def forward(self, inputs: OmniPromptType, sampling_params: Any = None) -> Any:
        # Simulate processing time with random sleep 1-3 seconds
        sleep_time = random.uniform(1.0, 3.0)
        logger.info(f"{self.name} sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
        
        # Access inputs['prompt_embeds'] or inputs['additional_information']
        logger.info(f"Running {self.name} on input type: {type(inputs)}")
        return inputs
