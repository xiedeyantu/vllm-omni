
import uuid
import time
from typing import List, Sequence, Union, Any
from vllm_omni.outputs import OmniRequestOutput, UniversalStageRequest, UniversalStageOutput
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.logger import init_logger
from .config import UniversalStageConfig
from .engine import UniversalEngine, UniversalEngineRequest

logger = init_logger(__name__)

class UniversalStage:
    """Stage implementation for universal operators.
    
    This stage acts as a generic processor in the multi-modal pipeline, 
    capable of routing data through a sequence of operators.
    
    Runs as a single-worker (no multiprocessing) for simplicity; 
    the engine directly processes requests in the current process.
    """

    def __init__(self, config: UniversalStageConfig):
        self.config = config
        self.stage_id = config.stage_id
        self.engine_input_source = config.engine_input_source
        
        # Initialize engine directly in this process
        logger.info(f"[Stage-{self.stage_id}] Initializing UniversalEngine")
        self.engine = UniversalEngine.make_engine(config)
        logger.info(f"[Stage-{self.stage_id}] UniversalEngine initialized")

    def generate(
        self,
        prompts: Union[OmniPromptType, Sequence[OmniPromptType]],
        sampling_params: Any = None,
        request_ids: List[str] = None,
        **kwargs
    ) -> List[OmniRequestOutput]:
        """Generate/process outputs for the given prompts.
        
        Similar to OmniDiffusion.generate(), this encapsulates parameters
        into an internal request object for processing.
        
        Supports two input formats:
        1. Direct dict format: {"video_paths": [...], ...}
        2. OpenAI message format: {"messages": [...]}
        
        Args:
            prompts: Single prompt or sequence of prompts
            sampling_params: Sampling/processing parameters
            request_ids: List of request IDs (auto-generated if not provided)
            
        Returns:
            List of OmniRequestOutput objects
        """
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        else:
            prompts = list(prompts)

        logger.info(f"[Stage-{self.stage_id}] Raw prompts received ({len(prompts)} items): {[type(p).__name__ + ': ' + str(list(p.keys()) if isinstance(p, dict) else p)[:100] for p in prompts]}")
        
        # Extract multimodal data from OpenAI message format if present
        processed_prompts = []
        for i, p in enumerate(prompts):
            processed_p = self._extract_multimodal_from_messages(p)
            logger.info(f"[Stage-{self.stage_id}] Prompt {i}: {type(p).__name__} -> {type(processed_p).__name__}, keys: {list(processed_p.keys()) if isinstance(processed_p, dict) else 'N/A'}")
            processed_prompts.append(processed_p)
        
        prompts = processed_prompts
        logger.info(f"[Stage-{self.stage_id}] Processed prompts: {[list(p.keys()) if isinstance(p, dict) else type(p) for p in prompts]}")

        # Auto-generate request_ids if needed (similar to OmniDiffusion)
        if request_ids is None:
            request_ids = []
        request_ids = list(request_ids) if request_ids else []
        if len(request_ids) < len(prompts):
            # Use similar format as OmniDiffusion: index_uuid
            new_ids = [f"{i + len(request_ids)}_{uuid.uuid4()}" for i in range(len(prompts) - len(request_ids))]
            request_ids.extend(new_ids)

        # Create internal request object (similar to OmniDiffusionRequest)
        request = UniversalEngineRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=request_ids
        )
        
        # Process through engine
        return self._run_engine(request)

    def _extract_multimodal_from_messages(self, prompt: Any) -> dict:
        """Extract multimodal data from OpenAI message format if present.
        
        Converts OpenAI chat message format to direct multimodal format.
        
        Input format (OpenAI):
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "..."},
                        {"type": "video_paths", "video_paths": [...]}
                    ]
                }
            ]
        }
        
        Output format (direct):
        {
            "video_paths": [...],
            "text": "..."
        }
        
        Args:
            prompt: The input prompt (dict or other type)
            
        Returns:
            Processed prompt with extracted multimodal data
        """
        if not isinstance(prompt, dict):
            logger.debug(f"[Stage-{self.stage_id}] Prompt is not dict: {type(prompt)}")
            return prompt
        
        # If already in direct format, return as-is
        if "video_paths" in prompt or "image_paths" in prompt or "audio_paths" in prompt:
            logger.debug(f"[Stage-{self.stage_id}] Prompt already has multimodal data: {list(prompt.keys())}")
            return prompt
        
        # Check if this is OpenAI message format
        if "messages" not in prompt:
            logger.debug(f"[Stage-{self.stage_id}] Prompt has no messages: {list(prompt.keys())}")
            return prompt
        
        messages = prompt.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            return prompt
        
        # Extract multimodal data from message content
        extracted = {}
        text_parts = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            content = msg.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            
            if not isinstance(content, list):
                continue
            
            for part in content:
                if not isinstance(part, dict):
                    continue
                
                part_type = part.get("type")
                
                if part_type == "text":
                    text = part.get("text")
                    if text:
                        text_parts.append(text)
                
                elif part_type == "video_paths":
                    video_paths = part.get("video_paths", [])
                    if video_paths:
                        extracted["video_paths"] = video_paths
                        logger.debug(f"[Stage-{self.stage_id}] Extracted {len(video_paths)} video paths from messages")
                
                elif part_type == "image_paths":
                    image_paths = part.get("image_paths", [])
                    if image_paths:
                        extracted["image_paths"] = image_paths
                        logger.debug(f"[Stage-{self.stage_id}] Extracted {len(image_paths)} image paths from messages")
                
                elif part_type == "audio_paths":
                    audio_paths = part.get("audio_paths", [])
                    if audio_paths:
                        extracted["audio_paths"] = audio_paths
                        logger.debug(f"[Stage-{self.stage_id}] Extracted {len(audio_paths)} audio paths from messages")
        
        # If we extracted multimodal data, add the text and return
        if extracted:
            if text_parts:
                extracted["text"] = " ".join(text_parts)
            return extracted
        
        # Otherwise return original prompt
        return prompt


    def _run_engine(self, request: UniversalEngineRequest) -> List[OmniRequestOutput]:
        """Execute the engine and wrap results in OmniRequestOutput.
        
        Similar to OmniDiffusion._run_engine(), takes an internal request object
        and returns a list of OmniRequestOutput objects. Uses engine.step() with
        batch request for efficient batch processing.
        
        Args:
            request: UniversalEngineRequest with prompts, sampling_params, and request_ids
            
        Returns:
            List of OmniRequestOutput objects
        """
        try:
            logger.info(f"[Stage-{self.stage_id}] Processing batch with {len(request.prompts)} requests")
            start_time = time.time()
            
            # Execute engine with batch request (directly use the request object)
            engine_outputs = self.engine.step(request)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Convert each engine output to OmniRequestOutput
            results = []
            for rid, engine_output in zip(request.request_ids, engine_outputs):
                # Convert engine output to UniversalStageOutput
                us_output = UniversalStageOutput.from_engine_output(
                    request_id=rid,
                    stage_id=self.stage_id,
                    engine_output=engine_output,
                    execution_time_ms=execution_time_ms / len(request.request_ids),  # Distribute time evenly
                )
                
                logger.info(f"[Stage-{self.stage_id}] Request {rid} processed")
                
                # Convert to OmniRequestOutput for compatibility with existing code
                ro = us_output.to_omni_request_output()
                results.append(ro)
                
            logger.info(f"[Stage-{self.stage_id}] Batch processing completed in {execution_time_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"[Stage-{self.stage_id}] Error processing batch: {e}", exc_info=True)
            # Return error results for all requests
            results = []
            for rid in request.request_ids:
                error_us_output = UniversalStageOutput(
                    request_id=rid,
                    stage_id=self.stage_id,
                    finished=False,
                )
                error_us_output.add_error(str(e))
                error_ro = error_us_output.to_omni_request_output()
                results.append(error_ro)
        
        return results

    def close(self) -> None:
        """Cleanup resources."""
        pass

    def shutdown(self) -> None:
        """Shutdown the stage (alias for close)."""
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

