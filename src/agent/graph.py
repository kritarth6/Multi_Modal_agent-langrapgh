"""LangGraph StateGraph builder for Multimodal Multi-Agent Workflow."""

import os
import sqlite3
import logging
from typing import Any, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None

from src.agent.state import AgentState
from src.agent.nodes.supervisor import supervisor_node
from src.agent.nodes.vision import vision_analyst_node
from src.agent.nodes.researcher import web_researcher_node
from src.agent.nodes.code_executor import code_interpreter_node
from src.agent.nodes.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def route_from_supervisor(state: AgentState) -> Literal["vision_analyst", "web_researcher", "code_interpreter", "synthesizer"]:
    """Conditional router based on supervisor's target_node decision."""
    target = state.get("target_node")
    if target in ["vision_analyst", "web_researcher", "code_interpreter", "synthesizer"]:
        return target
    return "synthesizer"


def route_from_vision(state: AgentState) -> Literal["web_researcher", "code_interpreter", "synthesizer"]:
    """Conditional router following vision analysis."""
    target = state.get("target_node")
    if target in ["web_researcher", "code_interpreter", "synthesizer"]:
        return target
    return "synthesizer"


def build_multimodal_agent_graph(
    checkpointer: Optional[Any] = None,
    use_sqlite: bool = False,
    sqlite_path: str = "data/checkpoints.db",
) -> Any:
    """
    Constructs and compiles the StateGraph workflow with nodes, conditional edges,
    and state checkpointer (MemorySaver or persistent SqliteSaver).
    """
    # 1. Initialize Graph with AgentState
    builder = StateGraph(AgentState)

    # 2. Add Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("vision_analyst", vision_analyst_node)
    builder.add_node("web_researcher", web_researcher_node)
    builder.add_node("code_interpreter", code_interpreter_node)
    builder.add_node("synthesizer", synthesizer_node)

    # 3. Add Edges
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "vision_analyst": "vision_analyst",
            "web_researcher": "web_researcher",
            "code_interpreter": "code_interpreter",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_conditional_edges(
        "vision_analyst",
        route_from_vision,
        {
            "web_researcher": "web_researcher",
            "code_interpreter": "code_interpreter",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_edge("web_researcher", "synthesizer")
    builder.add_edge("code_interpreter", "synthesizer")
    builder.add_edge("synthesizer", END)

    # 4. Checkpointer setup
    if checkpointer is not None:
        memory = checkpointer
    elif use_sqlite and SqliteSaver is not None:
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        memory = SqliteSaver(conn)
    else:
        memory = MemorySaver()

    app = builder.compile(checkpointer=memory)
    return app


def get_graph_mermaid_diagram() -> str:
    """Returns the visual Mermaid diagram representation of the workflow."""
    return """graph TD
    START([User Input & Images]) --> Supervisor[Supervisor Router]
    
    Supervisor -->|Images Present| Vision[Vision Analyst]
    Supervisor -->|External Grounding| Researcher[Web Researcher]
    Supervisor -->|Math / Scripting| Coder[Code Interpreter]
    Supervisor -->|Direct Query| Synthesizer[Synthesizer]
    
    Vision -->|Needs Live Verification| Researcher
    Vision -->|Needs Math Verification| Coder
    Vision -->|Analysis Complete| Synthesizer
    
    Researcher --> Synthesizer
    Coder --> Synthesizer
    
    Synthesizer --> END([Final Synthesized Response])

    classDef supervisor fill:#3b82f6,stroke:#1d4ed8,color:#ffffff,stroke-width:2px;
    classDef agent fill:#10b981,stroke:#047857,color:#ffffff,stroke-width:2px;
    classDef terminal fill:#6366f1,stroke:#4338ca,color:#ffffff,stroke-width:2px;

    class Supervisor supervisor;
    class Vision,Researcher,Coder agent;
    class Synthesizer terminal;
"""
