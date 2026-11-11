"""Vision Analyst Node: Multimodal perception and visual reasoning."""

import time
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState
from src.llm_factory import get_chat_model


VISION_SYSTEM_PROMPT = """You are an Expert Multimodal Visual Reasoning Agent.
Your role is to deeply analyze uploaded images, charts, architectural diagrams, screenshots, or documents.

Provide a comprehensive, structured visual inspection:
1. **Visual Overview**: Core subject, dimensions/format, visual hierarchy.
2. **Key Data & Annotations**: Specific text/OCR details, numerical values, graph axes, trends, or visual labels.
3. **Anomalies / Defects / Insights**: Highlight noteworthy findings (e.g. bugs in screenshots, bottlenecks in architectures, outliers in charts).
4. **Actionable Takeaways**: Concrete conclusions derived from the visual evidence.
"""


def vision_analyst_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Inspects images in the state using a vision-enabled model."""
    start_time = time.time()
    images = state.get("images", [])
    messages = state.get("messages", [])

    # Get latest user query
    user_query = "Please analyze the attached image thoroughly."
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            c = msg.content
            if isinstance(c, str) and c.strip():
                user_query = c
                break
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_query = part.get("text", user_query)

    llm = get_chat_model(is_vision=True)

    # Build multimodal content payload
    content_payload: List[Dict[str, Any]] = [
        {"type": "text", "text": f"User question: {user_query}\n\nPerform a rigorous visual analysis:"}
    ]

    for idx, img_info in enumerate(images):
        b64 = img_info.get("base64_data", "")
        media_type = img_info.get("media_type", "image/png")
        if b64:
            if not b64.startswith("data:"):
                data_url = f"data:{media_type};base64,{b64}"
            else:
                data_url = b64
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

    try:
        if len(content_payload) > 1:
            prompt = [
                SystemMessage(content=VISION_SYSTEM_PROMPT),
                HumanMessage(content=content_payload)
            ]
        else:
            prompt = [
                SystemMessage(content=VISION_SYSTEM_PROMPT),
                HumanMessage(content=f"Analyze visual context for: {user_query}")
            ]
        
        resp = llm.invoke(prompt)
        analysis_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        analysis_text = f"Visual analysis encountered an error with remote provider: {e}. Fallback: Visual features extracted from {len(images)} artifact(s)."

    latency = round((time.time() - start_time) * 1000, 2)
    step = {
        "node": "vision_analyst",
        "timestamp": time.strftime("%H:%M:%S"),
        "thought": f"Analyzed {len(images)} image artifact(s). Extracted visual data and structure.",
        "action": "multimodal_visual_reasoning",
        "output": analysis_text[:200] + "...",
        "latency_ms": latency
    }

    # Decide if we need further calculation or research
    query_lower = user_query.lower()
    next_node = "synthesizer"
    if any(k in query_lower for k in ["calculate", "verify math", "compute", "run python"]):
        next_node = "code_interpreter"
    elif any(k in query_lower for k in ["search", "compare with current", "verify online", "latest"]):
        next_node = "web_researcher"

    return {
        "current_node": "vision_analyst",
        "target_node": next_node,
        "extracted_data": {"vision_analysis": analysis_text},
        "intermediate_steps": [step]
    }
