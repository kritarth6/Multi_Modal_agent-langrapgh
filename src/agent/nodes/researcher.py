"""Web Researcher Node: Live grounding and information retrieval."""

import time
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState
from src.agent.tools.web_search import execute_web_search
from src.agent.tools.flight_tracker import track_flights


def web_researcher_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Retrieves external facts to ground the agent's answer."""
    start_time = time.time()
    messages = state.get("messages", [])
    extracted_data = state.get("extracted_data", {})

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

    # Contextual search query
    vision_context = extracted_data.get("vision_analysis", "")
    query = user_query
    if len(query) > 120:
        query = query[:120]

    # Execute search
    results = execute_web_search(query=query, max_results=3)

    # Check if flight tracking is relevant
    flight_data = None
    query_lower = user_query.lower()
    if any(k in query_lower for k in ["flight", "airline", "airport", "boarding pass", "iata"]):
        # Extract potential flight code
        import re
        flight_match = re.search(r'\b([A-Z0-9]{2}\s?\d{2,4})\b', user_query.upper())
        flight_iata = flight_match.group(1).replace(" ", "") if flight_match else None
        flight_res = track_flights(flight_iata=flight_iata, limit=3)
        if flight_res.get("success") and flight_res.get("data"):
            flight_data = flight_res["data"]

    latency = round((time.time() - start_time) * 1000, 2)
    step = {
        "node": "web_researcher",
        "timestamp": time.strftime("%H:%M:%S"),
        "thought": f"Executed live search for: '{query}'. Retrieved {len(results)} authoritative sources." + (f" Retrieved {len(flight_data)} live flight records." if flight_data else ""),
        "action": "web_search_and_aviation_tracker",
        "output": f"Found {len(results)} source citations." + (f" Flight data attached." if flight_data else ""),
        "latency_ms": latency
    }

    tool_history_items = [{
        "tool": "web_search",
        "input": {"query": query},
        "output": results,
        "latency_ms": latency
    }]

    extracted_payload = {"research_results": results}
    if flight_data:
        tool_history_items.append({
            "tool": "flight_tracker",
            "input": {"query": user_query},
            "output": flight_data,
            "latency_ms": latency
        })
        extracted_payload["flight_data"] = flight_data

    return {
        "current_node": "web_researcher",
        "target_node": "synthesizer",
        "extracted_data": extracted_payload,
        "intermediate_steps": [step],
        "tool_history": tool_history_items
    }
