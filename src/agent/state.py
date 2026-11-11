"""Agent state definitions for LangGraph multi-modal workflow."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to merge dictionary states."""
    merged = left.copy() if left else {}
    if right:
        merged.update(right)
    return merged


def append_list(left: List[Any], right: List[Any]) -> List[Any]:
    """Reducer to append items to a list in state."""
    left_list = left if left is not None else []
    right_list = right if right is not None else []
    return left_list + right_list


class ImageArtifact(TypedDict, total=False):
    """Represents an image passed into the multi-modal agent."""
    name: str
    media_type: str  # e.g., 'image/png', 'image/jpeg'
    base64_data: str
    description: Optional[str]
    dimensions: Optional[Dict[str, int]]


class IntermediateStep(TypedDict, total=False):
    """Step execution metadata for transparent reasoning traces."""
    node: str
    timestamp: str
    thought: str
    action: Optional[str]
    output: Optional[str]
    latency_ms: Optional[float]


class AgentState(TypedDict):
    """Central state object passed between nodes in the LangGraph."""

    # Chat history with LangGraph's add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]

    # Multimodal artifacts (images, screenshots, charts)
    images: List[Dict[str, Any]]

    # High-level task routing & tracking
    current_node: str
    target_node: Optional[str]

    # Execution trace for UI visualization & observability
    intermediate_steps: Annotated[List[Dict[str, Any]], append_list]

    # Tool calls & outputs
    tool_history: Annotated[List[Dict[str, Any]], append_list]

    # Extracted data from subagents (e.g. OCR text, code results, search facts)
    extracted_data: Annotated[Dict[str, Any], merge_dicts]

    # Human-in-the-loop verification
    requires_human_approval: bool
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # Final synthesized output and status
    final_response: Optional[str]
    error: Optional[str]

