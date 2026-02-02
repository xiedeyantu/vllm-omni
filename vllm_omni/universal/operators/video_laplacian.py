from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import av
import cv2

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.logger import init_logger
from .base import Operator

if TYPE_CHECKING:
    from vllm_omni.universal.engine import UniversalEngineRequest

logger = init_logger(__name__)


class VideoLaplacianPipeline(Operator):
    """Operator for computing Laplacian variance (sharpness) scores for videos.
    
    This operator:
    - Reads video frames at specified stride intervals
    - Computes Laplacian variance (sharpness score) for each frame
    - Averages scores across all sampled frames
    - Filters videos based on min/max score thresholds
    - Supports optional per-video cropping coordinates
    - Supports both 'any' (at least one passes) and 'all' (all pass) strategies
    """
    
    def __init__(self, config: Dict[str, Any], prefix: str = ""):
        super().__init__(config, prefix=prefix)
        
        # Extract config parameters with defaults
        self.stride = int(config.get("stride", 8))
        self.min_score = float(config.get("min_score", -1e10))
        self.max_score = float(config.get("max_score", 1e10))
        any_or_all = config.get("any_or_all", "any")
        self.any = (any_or_all == "any")
        
        # Optional per-video cropping coordinates
        self.ocr_crop_coords = config.get("ocr_crop_coords", {}) or {}
        
        logger.info(
            f"{self.prefix}VideoLaplacianPipeline initialized: stride={self.stride}, "
            f"min_score={self.min_score}, max_score={self.max_score}, "
            f"strategy={any_or_all}"
        )

    def _compute_laplacian_score(
        self,
        video_path: str,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """Compute average Laplacian variance (sharpness) score for a video.
        
        Args:
            video_path: Path to the video file
            crop_coords: Optional tuple (x1, x2, y1, y2) for cropping frames
            
        Returns:
            Average Laplacian variance score or 0.0 on error
        """
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            
            laplacian_sum = 0.0
            frame_count = 0
            
            for i, frame in enumerate(container.decode(video=0)):
                # Sample frames at specified stride
                if i % self.stride != 0:
                    continue
                
                # Convert frame to numpy array (RGB)
                img = frame.to_ndarray(format="rgb24")

                # Apply cropping if provided
                if crop_coords:
                    x1, x2, y1, y2 = crop_coords
                    img = img[y1:y2, x1:x2, :]
                
                # Compute Laplacian variance (sharpness metric)
                gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
                laplacian_sum += variance
                frame_count += 1
                
                # Free memory
                del img
                del gray_image
            
            container.close()
            
            # Compute average score
            score = float(laplacian_sum / max(frame_count, 1))
            
            logger.debug(
                f"{self.prefix}Video {video_path}: "
                f"sampled {frame_count} frames (stride={self.stride}), "
                f"laplacian_score={score:.4f}"
            )
            
            return score
            
        except Exception as e:
            logger.error(
                f"{self.prefix}Failed to compute laplacian score for {video_path}: {e}"
            )
            return 0.0

    def _check_threshold(self, scores: List[float]) -> bool:
        """Check if scores satisfy the threshold constraints.
        
        Args:
            scores: List of Laplacian scores to check
            
        Returns:
            True if threshold is satisfied, False otherwise
        """
        if not scores:
            return True
        
        # Check each score against min/max bounds
        keep_bools = [
            self.min_score <= score <= self.max_score
            for score in scores
        ]
        
        # Apply strategy: 'any' = at least one passes, 'all' = all pass
        if self.any:
            return any(keep_bools)
        else:
            return all(keep_bools)

    def forward(self, req: "UniversalEngineRequest") -> OmniRequestOutput:
        """Process video Laplacian scores from batch request.
        
        Expected input (each prompt in req.prompts dict with):
        - "video_paths": List of paths to video files
        - Additional fields are preserved in output
        
        Optional params override (req.params dict with):
        - "stride": Override frame sampling stride
        - "min_score": Override minimum score threshold
        - "max_score": Override maximum score threshold
        - "any_or_all": Override strategy (any/all)
        - "ocr_crop_coords": Dict mapping video_path to (x1, x2, y1, y2) crop coords
        
        Returns:
            OmniRequestOutput with extra_json containing:
            - "video_paths": Original list of video paths
            - "laplacian_scores": Computed Laplacian variance scores
            - "passed": Per-video pass/fail status
            - "overall_passed": Overall pass/fail status
            - "status": Per-video processing status
            - All other input fields preserved
        """
        try:
            # Store original parameters for restoration
            original_stride = self.stride
            original_min = self.min_score
            original_max = self.max_score
            original_any = self.any

            try:
                # Process all prompts in batch
                all_results = []
                params = req.params if hasattr(req, 'params') else {}
                
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
                            "laplacian_scores": [],
                            "passed": [],
                            "overall_passed": True,
                            "status": [],
                        }
                    else:
                        # Override parameters from req.params if provided
                        if "stride" in params:
                            self.stride = params["stride"]
                        if "min_score" in params:
                            self.min_score = params["min_score"]
                        if "max_score" in params:
                            self.max_score = params["max_score"]
                        if "any_or_all" in params:
                            self.any = (params["any_or_all"] == "any")
                        
                        # Compute Laplacian scores for all videos
                        scores = []
                        video_passed = []
                        statuses = []
                        
                        for video_path in video_paths:
                            try:
                                # Get crop coordinates for this video
                                crop_coords = None
                                if video_path in self.ocr_crop_coords:
                                    crop_coords = self.ocr_crop_coords[video_path]
                                elif "ocr_crop_coords" in params:
                                    crop_coords = params["ocr_crop_coords"].get(video_path)
                                
                                # Compute Laplacian score
                                score = self._compute_laplacian_score(video_path, crop_coords)
                                scores.append(score)
                                passed = self.min_score <= score <= self.max_score
                                video_passed.append(passed)
                                statuses.append("success")
                            except Exception as e:
                                logger.error(
                                    f"{self.prefix}Error processing video {video_path}: {e}"
                                )
                                scores.append(0.0)
                                video_passed.append(False)
                                statuses.append("error")
                        
                        # Determine overall pass/fail
                        overall_passed = self._check_threshold(scores)
                        
                        logger.info(
                            f"{self.prefix}VideoLaplacianPipeline processed "
                            f"{len(video_paths)} videos: "
                            f"scores={scores}, overall_passed={overall_passed}"
                        )
                        
                        result = {
                            "video_paths": video_paths,
                            "laplacian_scores": scores,
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
                self.stride = original_stride
                self.min_score = original_min
                self.max_score = original_max
                self.any = original_any
                
        except Exception as e:
            error_msg = f"{self.prefix}VideoLaplacianPipeline forward failed: {e}"
            logger.exception(error_msg)
            
            return OmniRequestOutput(
                request_id=req.request_ids[0] if req.request_ids else "unknown",
                finished=True,
                final_output_type="json",
                multimodal_output={"error": error_msg},
            )
