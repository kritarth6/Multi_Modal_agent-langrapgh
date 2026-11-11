"""Unit tests for LangGraph StateGraph execution and routing."""

from langchain_core.messages import HumanMessage
from src.agent.graph import build_multimodal_agent_graph, get_graph_mermaid_diagram


def test_graph_mermaid_diagram():
    diagram = get_graph_mermaid_diagram()
    assert "graph TD" in diagram
    assert "Supervisor" in diagram
    assert "Vision" in diagram


def test_graph_text_query_execution():
    graph = build_multimodal_agent_graph()
    config = {"configurable": {"thread_id": "test-text-thread"}}

    initial_state = {
        "messages": [HumanMessage(content="Hello! Explain how multi-agent systems work.")],
        "images": [],
        "current_node": "START",
        "target_node": None,
        "intermediate_steps": [],
        "tool_history": [],
        "extracted_data": {},
        "requires_human_approval": False,
        "human_approved": None,
        "human_feedback": None,
        "final_response": None,
        "error": None,
    }

    final_state = None
    for event in graph.stream(initial_state, config=config):
        for node, update in event.items():
            if "final_response" in update and update["final_response"]:
                final_state = update

    assert final_state is not None
    assert final_state["final_response"] is not None
    assert len(final_state["final_response"]) > 0


def test_graph_multimodal_execution():
    graph = build_multimodal_agent_graph()
    config = {"configurable": {"thread_id": "test-vision-thread"}}

    initial_state = {
        "messages": [HumanMessage(content="What does this diagram show?")],
        "images": [{
            "name": "test.png",
            "media_type": "image/png",
            "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        }],
        "current_node": "START",
        "target_node": None,
        "intermediate_steps": [],
        "tool_history": [],
        "extracted_data": {},
        "requires_human_approval": False,
        "human_approved": None,
        "human_feedback": None,
        "final_response": None,
        "error": None,
    }

    nodes_visited = []
    final_output = None
    for event in graph.stream(initial_state, config=config):
        for node, update in event.items():
            nodes_visited.append(node)
            if "final_response" in update and update["final_response"]:
                final_output = update["final_response"]

    assert "supervisor" in nodes_visited
    assert "vision_analyst" in nodes_visited
    assert "synthesizer" in nodes_visited
    assert final_output is not None

