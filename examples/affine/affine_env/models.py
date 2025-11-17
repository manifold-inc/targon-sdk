from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Challenge:
    env: str
    prompt: str
    extra: Dict[str, Any]
    timestamp: Optional[float]
