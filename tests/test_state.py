"""Unit tests for agent state definitions and reducers."""

from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import merge_dicts, append_list, AgentState


def test_merge_dicts():
    left = {"a": 1, "b": 2}
    right = {"b": 99, "c": 3}
    merged = merge_dicts(left, right)
    assert merged == {"a": 1, "b": 99, "c": 3}


def test_append_list():
    left = [1, 2]
    right = [3, 4]
    appended = append_list(left, right)
    assert appended == [1, 2, 3, 4]


def test_state_structure():
    msg = HumanMessage(content="Hello")
    state: AgentState = {
        "messages": [msg],
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
    assert len(state["messages"]) == 1
    assert state["current_node"] == "START"

