from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from PIL import Image
from vllm.outputs import RequestOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.inputs.data import OmniPromptType

if TYPE_CHECKING:
    from vllm_omni.engine import OmniEngineCoreOutput


class OmniModelRunnerOutput(ModelRunnerOutput):
    """Model runner output for omni models.

    Extends the base ModelRunnerOutput with support for multimodal outputs
    that may be produced by non-autoregressive stages.

    Attributes:
        multimodal_outputs: Optional dictionary mapping modality names to
            output tensors (e.g., {"image": tensor, "audio": tensor})
    """

    multimodal_outputs: dict[str, torch.Tensor] | None = None
    # IDs of requests whose KV cache has been extracted from GPU/NPU to CPU.
    # The Scheduler can safely free the block tables for these requests.
    kv_extracted_req_ids: list[str] | None = None


@dataclass
class OmniRequestOutput:
    """Unified request output for both pipeline stages and diffusion models.

    This class handles outputs from:
    1. Multi-stage LLM pipelines (with stage_id, final_output_type, request_output)
    2. Diffusion models (with images, prompt, metrics)

    Attributes:
        request_id: Unique identifier for this request
        finished: Whether generation is complete
        stage_id: Identifier of the stage that produced this output (pipeline mode)
        final_output_type: Type of output ("text", "image", "audio", "latents")
        request_output: The underlying RequestOutput from the stage (pipeline mode)
        images: List of generated PIL images (diffusion mode)
        prompt: The prompt used for generation (diffusion mode)
        latents: Optional tensor of latent representations (diffusion mode)
        metrics: Optional dictionary of generation metrics
    """

    request_id: str = ""
    finished: bool = True

    # Pipeline stage fields
    stage_id: int | None = None
    final_output_type: str = "text"
    request_output: RequestOutput | None = None

    # Diffusion model fields
    images: list[Image.Image] = field(default_factory=list)
    prompt: OmniPromptType | None = None
    latents: torch.Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    multimodal_output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pipeline(
        cls,
        stage_id: int,
        final_output_type: str,
        request_output: RequestOutput,
    ) -> "OmniRequestOutput":
        """Create output from pipeline stage.

        Args:
            stage_id: Stage identifier
            final_output_type: Type of output
            request_output: The stage's output

        Returns:
            OmniRequestOutput configured for pipeline mode
        """
        return cls(
            request_id=getattr(request_output, "request_id", ""),
            stage_id=stage_id,
            final_output_type=final_output_type,
            request_output=request_output,
            finished=True,
        )

    @classmethod
    def from_diffusion(
        cls,
        request_id: str,
        images: list[Image.Image],
        prompt: OmniPromptType | None = None,
        metrics: dict[str, Any] | None = None,
        latents: torch.Tensor | None = None,
        multimodal_output: dict[str, Any] | None = None,
        final_output_type: str = "image",
    ) -> "OmniRequestOutput":
        """Create output from diffusion model.

        Args:
            request_id: Request identifier
            images: Generated images
            prompt: The prompt used
            metrics: Generation metrics
            latents: Optional latent tensors

        Returns:
            OmniRequestOutput configured for diffusion mode
        """
        return cls(
            request_id=request_id,
            final_output_type=final_output_type,
            images=images,
            prompt=prompt,
            latents=latents,
            metrics=metrics or {},
            multimodal_output=multimodal_output or {},
            finished=True,
        )

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    # Pass-through properties keep vLLM serving codepaths compatible with
    # OmniRequestOutput for pipeline outputs (Issue #345).
    @property
    def prompt_token_ids(self) -> list[int] | None:
        """Return prompt token IDs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        chat completion generator which checks res.prompt_token_ids.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_token_ids", None)
        return None

    @property
    def outputs(self) -> list[Any]:
        """Return outputs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        and non-streaming chat completion generators.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "outputs", [])
        return []

    @property
    def encoder_prompt_token_ids(self) -> list[int] | None:
        """Return encoder prompt token IDs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "encoder_prompt_token_ids", None)
        return None

    @property
    def prompt_logprobs(self) -> Any:
        """Return prompt logprobs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_logprobs", None)
        return None

    @property
    def num_cached_tokens(self) -> int | None:
        """Return number of cached tokens from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "num_cached_tokens", None)
        return None

    @property
    def kv_transfer_params(self) -> Any:
        """Return KV transfer params from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "kv_transfer_params", None)
        return None

    @property
    def is_diffusion_output(self) -> bool:
        """Check if this is a diffusion model output."""
        return len(self.images) > 0 or self.final_output_type == "image"

    @property
    def is_pipeline_output(self) -> bool:
        """Check if this is a pipeline stage output."""
        return self.stage_id is not None and self.request_output is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "finished": self.finished,
            "final_output_type": self.final_output_type,
        }

        if self.is_diffusion_output:
            result.update(
                {
                    "num_images": self.num_images,
                    "prompt": self.prompt,
                    "metrics": self.metrics,
                }
            )

        if self.is_pipeline_output:
            result.update(
                {
                    "stage_id": self.stage_id,
                }
            )

        return result

    def __repr__(self) -> str:
        """Custom repr to properly show image count instead of image objects."""
        # For images, show count instead of full list
        images_repr = f"[{len(self.images)} PIL Images]" if self.images else "[]"

        # Build repr string
        parts = [
            f"request_id={self.request_id!r}",
            f"finished={self.finished}",
            f"stage_id={self.stage_id}",
            f"final_output_type={self.final_output_type!r}",
            f"request_output={self.request_output}",
            f"images={images_repr}",
            f"prompt={self.prompt!r}",
            f"latents={self.latents}",
            f"metrics={self.metrics}",
            f"multimodal_output={self.multimodal_output}",
        ]

        return f"OmniRequestOutput({', '.join(parts)})"


@dataclass
class UniversalStageRequest:
    """Request for universal stage operators.
    
    This class represents a standardized request that flows through 
    the universal stage pipeline. It can originate from:
    1. API requests (chat completion requests)
    2. Previous stage outputs (chained processing)
    
    The request data can be a dictionary (typical) or other structured formats
    that operators understand.
    
    Attributes:
        request_id: Unique identifier for this request
        stage_id: Identifier of the current/target stage
        data: The actual request data (typically dict with "video_paths", etc.)
        metadata: Optional metadata about the request
        source_stage_id: Which stage this request came from (for chaining)
        pass_through_fields: Additional fields to preserve through pipeline
    """
    
    request_id: str
    stage_id: int
    data: dict[str, Any]  # Main payload - operators expect this format
    metadata: dict[str, Any] = field(default_factory=dict)
    source_stage_id: int | None = None  # For stage chaining
    pass_through_fields: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_request(
        cls,
        request_id: str,
        stage_id: int,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "UniversalStageRequest":
        """Create request from API call.
        
        Args:
            request_id: Request identifier
            stage_id: Target stage ID
            data: Request data (dict with operator-specific fields)
            metadata: Optional metadata
            
        Returns:
            UniversalStageRequest configured for API mode
        """
        return cls(
            request_id=request_id,
            stage_id=stage_id,
            data=data,
            metadata=metadata or {},
            source_stage_id=None,
        )
    
    @classmethod
    def from_previous_stage(
        cls,
        request_id: str,
        source_stage_id: int,
        target_stage_id: int,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        pass_through: dict[str, Any] | None = None,
    ) -> "UniversalStageRequest":
        """Create request from previous stage output (stage chaining).
        
        Args:
            request_id: Request identifier (preserved from previous stage)
            source_stage_id: ID of previous stage
            target_stage_id: ID of current/target stage
            data: Processed data from previous stage
            metadata: Optional metadata
            pass_through: Fields to preserve through pipeline
            
        Returns:
            UniversalStageRequest configured for stage chaining
        """
        return cls(
            request_id=request_id,
            stage_id=target_stage_id,
            data=data,
            metadata=metadata or {},
            source_stage_id=source_stage_id,
            pass_through_fields=pass_through or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "stage_id": self.stage_id,
            "data": self.data,
            "metadata": self.metadata,
            "source_stage_id": self.source_stage_id,
        }


@dataclass
class UniversalStageOutput:
    """Unified output from universal stage operators.
    
    This class wraps the output from UniversalEngine and represents
    the processed result. It bridges between OmniEngineCoreOutput and
    OmniRequestOutput, providing:
    1. Operator-specific outputs (stored by operator name)
    2. Processing metadata and status
    3. Easy conversion to OmniRequestOutput for API responses
    4. Support for stage chaining to next stage
    
    Attributes:
        request_id: Request identifier
        stage_id: Stage that produced this output
        operator_outputs: Dict mapping operator class name to its output dict
            Each operator output typically contains:
            - "result": The main computation result
            - "status": "success" or "error"
            - "passed": Per-item pass/fail status
            - Other operator-specific fields
        metrics: Processing metrics (timing, throughput, etc.)
        errors: Any errors that occurred during processing
        finished: Whether processing completed successfully
        execution_time_ms: Total execution time in milliseconds
    """
    
    request_id: str
    stage_id: int
    operator_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    finished: bool = True
    execution_time_ms: float = 0.0
    
    @classmethod
    def from_engine_output(
        cls,
        request_id: str,
        stage_id: int,
        engine_output: "OmniEngineCoreOutput",
        execution_time_ms: float = 0.0,
    ) -> "UniversalStageOutput":
        """Create output from UniversalEngine output.
        
        Args:
            request_id: Request identifier
            stage_id: Stage ID
            engine_output: Output from UniversalEngine.execute()
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            UniversalStageOutput with all engine outputs captured
        """
        # Extract operator outputs from us_output dict
        operator_outputs = getattr(engine_output, "us_output", {})
        
        # Extract metrics if available
        metrics = {}
        if hasattr(engine_output, "multimodal_outputs"):
            metrics["multimodal_outputs"] = engine_output.multimodal_outputs
        if hasattr(engine_output, "pooling_output") and engine_output.pooling_output:
            metrics["pooling_output"] = engine_output.pooling_output
        
        return cls(
            request_id=request_id,
            stage_id=stage_id,
            operator_outputs=operator_outputs,
            metrics=metrics,
            finished=True,
            execution_time_ms=execution_time_ms,
        )
    
    def add_error(self, error_msg: str) -> None:
        """Add error message."""
        self.errors.append(error_msg)
        self.finished = False
    
    def to_omni_request_output(self) -> OmniRequestOutput:
        """Convert to OmniRequestOutput for API response.
        
        This bridges UniversalStageOutput to the standard OmniRequestOutput
        format that the serving layer expects.
        
        Returns:
            OmniRequestOutput with universal stage data packed into fields
        """
        ro = OmniRequestOutput(
            request_id=self.request_id,
            stage_id=self.stage_id,
            finished=self.finished,
            final_output_type="universal",  # Mark as universal stage output
        )
        
        # Pack operator outputs and metrics into multimodal_output
        ro.multimodal_output = {
            "operator_outputs": self.operator_outputs,
            "metrics": self.metrics,
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
        }
        
        # Add metrics to the metrics field for compatibility
        ro.metrics = {
            "execution_time_ms": self.execution_time_ms,
            "operators_count": len(self.operator_outputs),
            "has_errors": len(self.errors) > 0,
        }
        
        return ro
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "stage_id": self.stage_id,
            "operator_outputs": self.operator_outputs,
            "metrics": self.metrics,
            "errors": self.errors,
            "finished": self.finished,
            "execution_time_ms": self.execution_time_ms,
        }
    
    def __repr__(self) -> str:
        """Custom repr."""
        return (
            f"UniversalStageOutput("
            f"request_id={self.request_id!r}, "
            f"stage_id={self.stage_id}, "
            f"operators={len(self.operator_outputs)}, "
            f"finished={self.finished}, "
            f"time={self.execution_time_ms:.2f}ms"
            f")"
        )
