"""FastAPI REST API server for the Multi-Modal LangGraph Agent."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.graph import build_multimodal_agent_graph, get_graph_mermaid_diagram
from src.config import settings

app = FastAPI(
    title="Multi-Modal LangGraph Agent API",
    description="Production-grade asynchronous multi-agent orchestration for text, vision, code, and web retrieval.",
    version="1.0.0",
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory compiled graph with checkpointer
graph_engine = build_multimodal_agent_graph()


class ChatImagePayload(BaseModel):
    name: Optional[str] = "upload.png"
    media_type: str = "image/png"
    base64_data: str


class ChatRequest(BaseModel):
    message: str = Field(..., example="Analyze this financial chart and calculate the CAGR")
    thread_id: Optional[str] = Field(default=None, example="user-session-123")
    images: Optional[List[ChatImagePayload]] = Field(default_factory=list)
    provider: Optional[str] = Field(default=None, example="groq")
    api_key: Optional[str] = None


class StepTrace(BaseModel):
    node: str
    timestamp: str
    thought: str
    action: Optional[str] = None
    output: Optional[str] = None
    latency_ms: Optional[float] = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    intermediate_steps: List[Dict[str, Any]]
    tool_history: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "default_provider": settings.DEFAULT_LLM_PROVIDER,
        "groq_configured": bool(settings.GROQ_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "tavily_configured": bool(settings.TAVILY_API_KEY),
        "aviationstack_configured": bool(settings.AVIATIONSTACK_API_KEY),
    }


@app.get("/api/graph")
def get_graph():
    """Returns the visual Mermaid diagram of the agent StateGraph."""
    return {"mermaid": get_graph_mermaid_diagram()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """Executes the multimodal agent on user input and returns synthesized response and traces."""
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Format images
    img_list = []
    if payload.images:
        for img in payload.images:
            img_list.append({
                "name": img.name,
                "media_type": img.media_type,
                "base64_data": img.base64_data,
            })

    initial_state = {
        "messages": [HumanMessage(content=payload.message)],
        "images": img_list,
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

    steps = []
    tools = []
    extracted = {}
    final_text = ""

    try:
        for event in graph_engine.stream(initial_state, config=config):
            for node_name, update in event.items():
                if "intermediate_steps" in update:
                    steps.extend(update["intermediate_steps"])
                if "tool_history" in update:
                    tools.extend(update["tool_history"])
                if "extracted_data" in update:
                    extracted.update(update["extracted_data"])
                if "final_response" in update and update["final_response"]:
                    final_text = update["final_response"]

        if not final_text:
            final_text = "Analysis completed. Check intermediate steps for details."

        return ChatResponse(
            thread_id=thread_id,
            response=final_text,
            intermediate_steps=steps,
            tool_history=tools,
            extracted_data=extracted,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

