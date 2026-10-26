"""
src/agent/trace.py

Structured JSON tracing and formatting helpers for the Sentry inspection agent.
Ensures step-by-step auditability and converts NumPy datatypes to standard
JSON-serializable Python representations without embedding raw image arrays.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import numpy as np


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts NumPy scalars, arrays, collections, and custom types
    into standard JSON-safe Python primitives (int, float, bool, str, list, dict).

    Large arrays or masks are never embedded directly; instead, a lightweight
    summary dictionary is produced.

    Parameters
    ----------
    obj : Any
        Any Python or NumPy object to serialize.

    Returns
    -------
    Any
        JSON-safe primitive representation.
    """
    if obj is None:
        return None

    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return str(val)
        return val

    if isinstance(obj, str):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, np.ndarray):
        # Prevent serializing large images or full binary masks into trace logs
        if obj.ndim >= 2 and obj.size > 50:
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "non_zero_count": int(np.count_nonzero(obj)),
            }
        return [make_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    if hasattr(obj, "isoformat"):
        return obj.isoformat()

    return str(obj)


@dataclass
class TraceStep:
    """
    Represents an atomic, audited action in the agent's inspection loop.

    Attributes
    ----------
    step : int
        1-indexed sequence number of the step.
    tool : str
        Name of the tool, function, or internal check executed.
    inputs_and_metrics : dict
        Salient inputs, configuration, and computed metrics for this step.
    decision : str
        The conclusion or branch selected based on the metrics.
    reason : str
        Human-readable technical rationale explaining why the decision was taken.
    timestamp : str
        ISO 8601 UTC timestamp recording when the step completed.
    """
    step: int
    tool: str
    inputs_and_metrics: Dict[str, Any]
    decision: str
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts step to a JSON-safe dictionary."""
        return {
            "step": self.step,
            "tool": self.tool,
            "inputs_and_metrics": make_json_safe(self.inputs_and_metrics),
            "decision": self.decision,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class InspectionTrace:
    """
    Manages and serializes an ordered sequence of audited agent trace steps.
    Enforces a strict maximum step limit to prevent runaway execution.
    """

    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self.steps: List[TraceStep] = []

    def add_step(
        self,
        tool: str,
        inputs_and_metrics: Dict[str, Any],
        decision: str,
        reason: str,
    ) -> TraceStep:
        """
        Records a new audited step into the trace.

        Parameters
        ----------
        tool : str
            Tool or procedure called (e.g., 'check_quality', 'detect_cracks').
        inputs_and_metrics : dict
            Metrics, parameters, or signals used for this decision.
        decision : str
            Action or branch determined by this step.
        reason : str
            Detailed explanation for the decision.

        Returns
        -------
        TraceStep
            The recorded TraceStep object.

        Raises
        ------
        RuntimeError
            If adding a step exceeds the configured maximum step limit.
        """
        if len(self.steps) >= self.max_steps:
            raise RuntimeError(
                f"Agent step limit exceeded: maximum allowed steps is {self.max_steps}."
            )

        step_num = len(self.steps) + 1
        now_utc = datetime.now(timezone.utc).isoformat()

        step = TraceStep(
            step=step_num,
            tool=tool,
            inputs_and_metrics=make_json_safe(inputs_and_metrics),
            decision=decision,
            reason=reason,
            timestamp=now_utc,
        )
        self.steps.append(step)
        return step

    def to_list(self) -> List[Dict[str, Any]]:
        """Returns the full trace as a list of JSON-safe dictionaries."""
        return [s.to_dict() for s in self.steps]

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes the trace into a JSON string."""
        return json.dumps(self.to_list(), indent=indent)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> TraceStep:
        return self.steps[idx]
