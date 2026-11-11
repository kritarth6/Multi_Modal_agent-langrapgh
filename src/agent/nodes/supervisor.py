"""Supervisor Node: Classifies intent and routes state to specialized subagents."""

import time
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState
from src.llm_factory import get_chat_model


SUPERVISOR_PROMPT = """You are the Supervisory Router in a Multi-Modal LangGraph Agent system.
Your job is to analyze the user's latest request and multimodal context, then determine the next best specialized node.

Available nodes:
- vision_analyst: Use when an image, chart, UI screenshot, diagram, or visual artifact is provided or needs inspection.
- web_researcher: Use when current real-time information, market data, news, or external facts are needed.
- code_interpreter: Use when math calculations, data analysis, Python scripts, or metric simulations are needed.
- synthesizer: Use when sufficient information is gathered, or for general queries that can be answered directly.

Respond strictly in this format:
ROUTE:<node_name> | REASON:<concise explanation>
"""


def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Inspects state and decides routing."""
    start_time = time.time()
    messages = state.get("messages", [])
    images = state.get("images", [])
    extracted_data = state.get("extracted_data", {})

    # Extract user text
    latest_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            c = msg.content
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        latest_text += part.get("text", "")
            elif isinstance(c, str):
                latest_text = c
            break

    # Smart heuristic priority
    target = "synthesizer"
    reason = "Direct reasoning query"

    has_images = bool(images)
    already_analyzed_vision = "vision_analysis" in extracted_data
    already_researched = "research_results" in extracted_data
    already_coded = "code_results" in extracted_data

    # Check if vision is needed
    if has_images and not already_analyzed_vision:
        target = "vision_analyst"
        reason = "Unprocessed visual artifacts present in state"
    elif any(kw in latest_text.lower() for kw in ["search", "latest", "recent", "who is", "news", "price of"]) and not already_researched:
        target = "web_researcher"
        reason = "User requested external web search / live data"
    elif any(kw in latest_text.lower() for kw in ["calculate", "math", "code", "run python", "compute", "simulate"]) and not already_coded:
        target = "code_interpreter"
        reason = "Mathematical calculation or Python script execution required"
    else:
        # LLM classification
        try:
            llm = get_chat_model(is_vision=False)
            prompt = [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=f"User Query: {latest_text}\nImages Attached: {len(images)}\nExtracted Data Keys: {list(extracted_data.keys())}")
            ]
            resp = llm.invoke(prompt, system_prompt="supervisor")
            content = resp.content if hasattr(resp, "content") else str(resp)
            if "ROUTE:" in content:
                parts = content.split("ROUTE:")[1].split("|")
                candidate = parts[0].strip().lower()
                if candidate in ["vision_analyst", "web_researcher", "code_interpreter", "synthesizer"]:
                    target = candidate
                    if len(parts) > 1 and "REASON:" in parts[1]:
                        reason = parts[1].replace("REASON:", "").strip()
        except Exception as e:
            target = "synthesizer"
            reason = f"Fallback routing due to: {e}"

    latency = round((time.time() - start_time) * 1000, 2)
    step = {
        "node": "supervisor",
        "timestamp": time.strftime("%H:%M:%S"),
        "thought": f"Routed workflow to '{target}'. Reason: {reason}",
        "action": f"route_to_{target}",
        "latency_ms": latency
    }

    return {
        "current_node": "supervisor",
        "target_node": target,
        "intermediate_steps": [step]
    }
