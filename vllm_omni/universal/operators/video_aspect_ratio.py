from typing import Any, Dict, List, TYPE_CHECKING
from fractions import Fraction
import av

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.logger import init_logger
from .base import Operator

if TYPE_CHECKING:
    from vllm_omni.universal.engine import UniversalEngineRequest

logger = init_logger(__name__)


class VideoAspectRatioPipeline(Operator):
    """Operator for filtering videos based on aspect ratio constraints.
    
    This operator:
    - Reads video metadata (width, height) from video files
    - Computes aspect ratios
    - Filters videos based on min/max ratio thresholds
    - Supports both 'any' (at least one passes) and 'all' (all pass) strategies
    """
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__(config, prefix=prefix)
        
        # Extract config parameters with defaults
        min_ratio_str = config.get("min_ratio", "9/21")
        max_ratio_str = config.get("max_ratio", "21/9")
        any_or_all = config.get("any_or_all", "any")
        
        # Parse aspect ratio fractions (support both '9/21' and '9:21' formats)
        self.min_ratio = Fraction(str(min_ratio_str).replace(":", "/"))
        self.max_ratio = Fraction(str(max_ratio_str).replace(":", "/"))
        self.any = (any_or_all == "any")
        
        logger.info(
            f"{self.prefix}VideoAspectRatioPipeline initialized: "
            f"min_ratio={self.min_ratio}, max_ratio={self.max_ratio}, "
            f"strategy={any_or_all}"
        )

    def _compute_aspect_ratio(self, video_path: str) -> float:
        """Compute aspect ratio for a single video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Aspect ratio (width / height) or 0.0 on error
        """
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            
            width = stream.codec_context.width
            height = stream.codec_context.height
            
            container.close()
            
            if height == 0:
                logger.warning(
                    f"{self.prefix}Video {video_path} has height=0, returning 0.0"
                )
                return 0.0
            
            aspect_ratio = width / height
            logger.debug(
                f"{self.prefix}Video {video_path}: {width}x{height}, "
                f"aspect_ratio={aspect_ratio:.4f}"
            )
            return aspect_ratio
            
        except Exception as e:
            logger.error(
                f"{self.prefix}Failed to compute aspect ratio for {video_path}: {e}"
            )
            return 0.0

    def _check_threshold(self, ratios: List[float]) -> bool:
        """Check if ratios satisfy the threshold constraints.
        
        Args:
            ratios: List of aspect ratios to check
            
        Returns:
            True if threshold is satisfied, False otherwise
        """
        if not ratios:
            return True
        
        # Convert each ratio to fraction for comparison
        keep_bools = [
            self.min_ratio <= Fraction(ratio).limit_denominator(1000) <= self.max_ratio
            for ratio in ratios
        ]
        
        # Apply strategy: 'any' = at least one passes, 'all' = all pass
        if self.any:
            return any(keep_bools)
        else:
            return all(keep_bools)

    def forward(self, req: "UniversalEngineRequest") -> OmniRequestOutput:
        """Process video aspect ratios from batch request.
        
        Expected input (each prompt in req.prompts dict with):
        - "video_paths": List of paths to video files
        - Additional fields are preserved in output
        
        Optional sampling_params override (req.sampling_params dict with):
        - "min_ratio": Override min aspect ratio
        - "max_ratio": Override max aspect ratio
        - "any_or_all": Override strategy (any/all)
        
        Returns:
            OmniRequestOutput with multimodal_output containing:
            - "video_paths": Original list of video paths (for each request)
            - "aspect_ratios": Computed aspect ratios
            - "passed": Per-video pass/fail status
            - "overall_passed": Overall pass/fail status
            - "status": Per-video processing status
            - All other input fields preserved
        """
        try:
            # Store original parameters for restoration
            original_min = self.min_ratio
            original_max = self.max_ratio
            original_any = self.any

            try:
                # Process all prompts in batch
                all_results = []
                
                for prompt in req.prompts:
                    if isinstance(prompt, dict):
                        video_paths = prompt.get("video_paths", [])
                        extra_data = {k: v for k, v in prompt.items() if k != "video_paths"}
                    else:
                        logger.warning(
                            f"{self.prefix}Prompt is not a dict, treating as empty video list"
                        )
                        video_paths = []
                        extra_data = {}

                    if not video_paths:
                        logger.warning(
                            f"{self.prefix}No video_paths provided in prompt"
                        )
                        result = {
                            "video_paths": [],
                            "aspect_ratios": [],
                            "passed": [],
                            "overall_passed": True,
                            "status": [],
                        }
                    else:
                        # Compute aspect ratios for all videos
                        ratios = []
                        video_passed = []
                        statuses = []
                        
                        for video_path in video_paths:
                            try:
                                ratio = self._compute_aspect_ratio(video_path)
                                ratios.append(ratio)
                                passed = self.min_ratio <= ratio <= self.max_ratio
                                video_passed.append(passed)
                                statuses.append("success")
                            except Exception as e:
                                logger.error(
                                    f"{self.prefix}Error processing video {video_path}: {e}"
                                )
                                ratios.append(0.0)
                                video_passed.append(False)
                                statuses.append("error")
                        
                        # Determine overall pass/fail
                        overall_passed = self._check_threshold(ratios)
                        
                        logger.info(
                            f"{self.prefix}VideoAspectRatioPipeline processed "
                            f"{len(video_paths)} videos: "
                            f"ratios={ratios}, overall_passed={overall_passed}"
                        )
                        
                        result = {
                            "video_paths": video_paths,
                            "aspect_ratios": ratios,
                            "passed": video_passed,
                            "overall_passed": overall_passed,
                            "status": statuses,
                        }
                    
                    all_results.append(result)
                
                # Build combined output for all requests
                return OmniRequestOutput(
                    request_id=req.request_ids[0] if req.request_ids else "unknown",
                    finished=True,
                    final_output_type="json",
                    multimodal_output={
                        "results": all_results
                    },
                )
                
            finally:
                # Restore original parameters
                self.min_ratio = original_min
                self.max_ratio = original_max
                self.any = original_any
                
        except Exception as e:
            error_msg = f"{self.prefix}VideoAspectRatioPipeline forward failed: {e}"
            logger.exception(error_msg)
            
            return OmniRequestOutput(
                request_id=req.request_ids[0] if req.request_ids else "unknown",
                finished=True,
                final_output_type="json",
                multimodal_output={"error": error_msg},
            )
