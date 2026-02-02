from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class UniversalStageConfig:
    """Configuration for a UniversalStage."""
    stage_id: int
    worker_id: int
    engine_input_source: List[int] = field(default_factory=list)
    operators: List[Dict[str, Any]] = field(default_factory=list)
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalStageConfig":
        return cls(
            stage_id=data.get("stage_id", 0),
            worker_id=data.get("worker_id", 0),
            engine_input_source=data.get("engine_input_source", []),
            operators=data.get("operators", []),
            runtime=data.get("runtime", {}),
        )
