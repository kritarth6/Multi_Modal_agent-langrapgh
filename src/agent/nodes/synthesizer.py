"""Synthesizer Node: Aggregates multimodal findings into a cohesive final response."""

import time
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState
from src.llm_factory import get_chat_model


SYNTHESIZER_SYSTEM_PROMPT = """You are an Executive Multimodal Intelligence Synthesizer.
Your goal is to present a cohesive, highly articulate, and well-structured response to the user.

Synthesize all available context:
- Visual Observations (if any images were analyzed)
- Computational/Code outputs (if any scripts were executed)
- Live Web Research citations (if any search queries were run)

Formatting guidelines:
1. Use clear Markdown headers (`###`), bold highlights, and clean bullet points.
2. If code results or numbers were computed, present them with precision.
3. If web sources were referenced, mention citations transparently.
4. Conclude with actionable takeaways or recommendations.
"""


def synthesizer_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Combines all multimodal observations and tool outputs into a final answer."""
    start_time = time.time()
    messages = state.get("messages", [])
    extracted_data = state.get("extracted_data", {})
    images = state.get("images", [])

    # Extract user query
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            c = msg.content
            if isinstance(c, str):
                user_query = c
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        user_query = p.get("text", "")
            break

    # Construct context dossier
    context_blocks = [f"User Question: {user_query}"]

    if "vision_analysis" in extracted_data:
        context_blocks.append(f"### Visual Inspection Findings:\n{extracted_data['vision_analysis']}")

    if "research_results" in extracted_data:
        res = extracted_data["research_results"]
        formatted_res = "\n".join([f"- **{r.get('title')}**: {r.get('snippet')} (URL: {r.get('url')})" for r in res])
        context_blocks.append(f"### Web Research Data:\n{formatted_res}")

    if "flight_data" in extracted_data:
        f_list = extracted_data["flight_data"]
        formatted_flights = "\n".join([
            f"- Flight **{f.get('flight_number')}** ({f.get('airline')}): Status `{f.get('status')}`, Departure `{f.get('departure_airport')}` -> Arrival `{f.get('arrival_airport')}`, Delay: {f.get('delay_minutes')} min"
            for f in f_list
        ])
        context_blocks.append(f"### Live Flight Telemetry (Aviationstack):\n{formatted_flights}")

    if "code_results" in extracted_data:
        code_info = extracted_data["code_results"]
        context_blocks.append(
            f"### Computation Output:\nStdout: {code_info.get('stdout')}\nVariables: {code_info.get('variables')}"
        )

    context_str = "\n\n".join(context_blocks)

    try:
        llm = get_chat_model(is_vision=False)
        prompt = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=f"Synthesize the following investigation dossier into an executive answer:\n\n{context_str}")
        ]
        resp = llm.invoke(prompt)
        final_answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        final_answer = (
            f"### Multimodal Investigation Summary\n\n"
            f"**Query**: {user_query}\n\n"
            f"{context_str}\n\n"
            f"*(Generated via fallback pipeline due to: {e})*"
        )

    latency = round((time.time() - start_time) * 1000, 2)
    step = {
        "node": "synthesizer",
        "timestamp": time.strftime("%H:%M:%S"),
        "thought": "Synthesized visual, computational, and research findings into final response.",
        "action": "finalize_answer",
        "latency_ms": latency
    }

    ai_msg = AIMessage(content=final_answer)

    return {
        "current_node": "synthesizer",
        "target_node": "END",
        "messages": [ai_msg],
        "final_response": final_answer,
        "intermediate_steps": [step]
    }
