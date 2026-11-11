"""Interactive Multimodal Agent Frontend built with Streamlit & LangGraph."""

import os
import sys
import uuid
import json
import time
from io import BytesIO
from PIL import Image
import streamlit as st

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import build_multimodal_agent_graph, get_graph_mermaid_diagram
from src.agent.tools.image_utils import image_to_base64
from src.config import settings

# Page setup
st.set_page_config(
    page_title="OmniAgent: Multi-Modal Multi-Agent System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-ingest Streamlit Cloud secrets into environment if present
try:
    if hasattr(st, "secrets"):
        for sec_key in ["GROQ_API_KEY", "TAVILY_API_KEY", "AVIATIONSTACK_API_KEY", "OPENAI_API_KEY"]:
            if sec_key in st.secrets and not os.environ.get(sec_key):
                os.environ[sec_key] = str(st.secrets[sec_key])
except Exception:
    pass

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []

if "tool_history" not in st.session_state:
    st.session_state.tool_history = []

if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}

if "compiled_graph" not in st.session_state:
    st.session_state.compiled_graph = build_multimodal_agent_graph()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    provider = st.selectbox(
        "LLM Provider",
        options=["Groq (Recommended - Fast)", "OpenAI", "Mock / Offline Demo"],
        index=0,
        help="Select model provider. Mock demo works offline without API keys."
    )
    
    selected_provider_key = "groq" if "Groq" in provider else ("openai" if "OpenAI" in provider else "mock")
    
    api_key = None
    if selected_provider_key == "groq":
        default_groq = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key", value=default_groq, type="password", placeholder="gsk_...")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    elif selected_provider_key == "openai":
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_openai, type="password", placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 🔌 Connected Integrations")
    groq_ok = bool(os.getenv("GROQ_API_KEY", "") or settings.GROQ_API_KEY)
    tavily_ok = bool(os.getenv("TAVILY_API_KEY", "") or settings.TAVILY_API_KEY)
    aviation_ok = bool(os.getenv("AVIATIONSTACK_API_KEY", "") or settings.AVIATIONSTACK_API_KEY)
    
    st.markdown(f"• **Groq Cloud:** {'🟢 Connected' if groq_ok else '⚪ Not set'}")
    st.markdown(f"• **Tavily Search:** {'🟢 Connected' if tavily_ok else '🟡 Fallback to DuckDuckGo'}")
    st.markdown(f"• **Aviationstack:** {'🟢 Connected' if aviation_ok else '⚪ Not set'}")

    st.markdown("---")
    st.markdown("### 🧠 Agent Persona & Guardrails")
    persona = st.selectbox(
        "Role Specialization",
        options=[
            "Senior Quantitative & Systems Architect",
            "Financial Chart & Equity Analyst",
            "Full-Stack Code & Security Auditor",
            "General Multimodal Assistant"
        ]
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    st.markdown("---")
    st.markdown("### 🧵 Thread & Memory Session")
    st.caption(f"Active Thread ID: `{st.session_state.thread_id[:8]}...`")
    
    col_reset, col_clear = st.columns(2)
    with col_reset:
        if st.button("🔄 New Thread", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.intermediate_steps = []
            st.session_state.tool_history = []
            st.session_state.extracted_data = {}
            st.session_state.uploaded_image = None
            st.rerun()
            
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.intermediate_steps = []
            st.session_state.tool_history = []
            st.session_state.extracted_data = {}
            st.session_state.uploaded_image = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Export Conversation")
    if st.session_state.messages:
        chat_dump = []
        for m in st.session_state.messages:
            chat_dump.append({"role": m["role"], "content": m["content"]})
        
        st.download_button(
            "Download JSON History",
            data=json.dumps(chat_dump, indent=2),
            file_name=f"agent_history_{st.session_state.thread_id[:6]}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.75rem; color:#94a3b8; line-height:1.5;'>
        <b>Tech Stack:</b><br/>
        • LangGraph StateGraph<br/>
        • Multi-Modal Perception<br/>
        • Stateful MemorySaver<br/>
        • Sandboxed Python REPL<br/>
        • Tavily / DuckDuckGo Search<br/>
        • Streamlit & FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Header Section ---
st.markdown(
    """
    <div class="header-badge">
        <span>⚡ LangGraph v1.2</span> • <span>Multi-Modal Orchestration</span> • <span>Stateful Memory</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 OmniAgent: Multi-Modal Multi-Agent System")
st.markdown(
    "Autonomous visual reasoning, dynamic web grounding, and sandboxed computation orchestrator."
)

# Tabs
tab_chat, tab_graph, tab_trace, tab_tools, tab_state = st.tabs([
    "💬 Interactive Console",
    "🕸️ LangGraph Architecture",
    "🔍 Execution Trace & Reasoning",
    "🛠️ Tool History",
    "📊 Raw State Inspector"
])

# --- Tab 1: Interactive Console ---
with tab_chat:
    # 1-Click Sample Showcase
    st.markdown("##### 🚀 1-Click Demo Scenarios")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    financial_sample = os.path.join(sample_dir, "sample_financial_chart.png")
    arch_sample = os.path.join(sample_dir, "sample_architecture.png")
    flight_sample = os.path.join(sample_dir, "sample_flight_telemetry.png")
    bug_sample = os.path.join(sample_dir, "sample_bug_screenshot.png")

    auto_run_prompt = None
    auto_run_image_path = None

    with c1:
        if st.button("📈 Financial Chart & CAGR", use_container_width=True):
            auto_run_prompt = "Analyze this financial chart, identify the overall price trajectory, and calculate the estimated return and volatility."
            auto_run_image_path = financial_sample
    with c2:
        if st.button("🏗️ Architecture Audit", use_container_width=True):
            auto_run_prompt = "Review this system architecture diagram. Identify potential bottlenecks, single points of failure, and security risks."
            auto_run_image_path = arch_sample
    with c3:
        if st.button("✈️ Flight Telemetry (AA100)", use_container_width=True):
            auto_run_prompt = "Analyze this flight boarding pass and query real-time aviation telemetry for flight AA100, including departure/arrival schedules and delay status."
            auto_run_image_path = flight_sample
    with c4:
        if st.button("🐛 Bug Debugger", use_container_width=True):
            auto_run_prompt = "Inspect this console error screenshot. Diagnose the root cause of the error and provide a code fix."
            auto_run_image_path = bug_sample
    with c5:
        if st.button("🔍 Live Grounding", use_container_width=True):
            auto_run_prompt = "Search for the latest architectural design patterns for multi-agent systems with LangGraph in 2026."

    st.markdown("---")

    # Image upload section
    col_uploader, col_preview = st.columns([2, 1])
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Attach Image / Chart / Diagram (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image to trigger visual multimodal reasoning."
        )
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file)
        elif auto_run_image_path and os.path.exists(auto_run_image_path):
            st.session_state.uploaded_image = Image.open(auto_run_image_path)

    with col_preview:
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Current Multimodal Context", use_container_width=True)
            if st.button("❌ Remove Image", key="clear_img"):
                st.session_state.uploaded_image = None
                st.rerun()

    # Display past conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "image" in msg and msg["image"]:
                st.image(msg["image"], width=300)
            st.markdown(msg["content"])

    # Chat Input
    user_input = st.chat_input("Ask a question, upload an image, or request analysis...")
    if auto_run_prompt and not user_input:
        user_input = auto_run_prompt

    if user_input:
        # Prepare image payload
        images_payload = []
        user_img_for_display = None
        if st.session_state.uploaded_image:
            b64_data = image_to_base64(st.session_state.uploaded_image)
            images_payload.append({
                "name": "uploaded_context.png",
                "media_type": "image/png",
                "base64_data": b64_data
            })
            user_img_for_display = st.session_state.uploaded_image

        # Append human message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "image": user_img_for_display
        })

        with st.chat_message("user"):
            if user_img_for_display:
                st.image(user_img_for_display, width=300)
            st.markdown(user_input)

        # Run LangGraph Agent
        with st.chat_message("assistant"):
            status_box = st.status("🧠 Agent orchestrating multi-node workflow...", expanded=True)
            
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "images": images_payload,
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

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response_container = st.empty()
            final_text = ""

            try:
                for event in st.session_state.compiled_graph.stream(initial_state, config=config):
                    for node_name, update in event.items():
                        # Display node progress
                        status_box.write(f"⚡ **Active Node**: `{node_name}`")
                        
                        steps = update.get("intermediate_steps", [])
                        if steps:
                            st.session_state.intermediate_steps.extend(steps)
                            latest = steps[-1]
                            status_box.caption(f"💭 *{latest.get('thought', '')}* ({latest.get('latency_ms', 0)} ms)")

                        if "tool_history" in update:
                            st.session_state.tool_history.extend(update["tool_history"])

                        if "extracted_data" in update:
                            st.session_state.extracted_data.update(update["extracted_data"])

                        if "final_response" in update and update["final_response"]:
                            final_text = update["final_response"]

                status_box.update(label="✅ Workflow Execution Complete!", state="complete", expanded=False)
                
                if final_text:
                    response_container.markdown(final_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_text
                    })
                else:
                    response_container.markdown("Execution completed. Please inspect intermediate steps.")

            except Exception as e:
                status_box.update(label=f"⚠️ Error: {str(e)}", state="error", expanded=True)
                st.error(f"Execution Error: {e}")

# --- Tab 2: LangGraph Architecture ---
with tab_graph:
    st.markdown("### 🕸️ LangGraph Cyclical State Machine")
    st.markdown(
        """
        The agent operates as a compiled **LangGraph StateGraph** with conditional routing,
        multimodal perception, external grounding, and stateful checkpointing.
        """
    )
    
    col_mermaid, col_nodes_desc = st.columns([3, 2])
    with col_mermaid:
        st.markdown(f"```mermaid\n{get_graph_mermaid_diagram()}\n```")
    
    with col_nodes_desc:
        st.markdown("#### 🧩 Node Specifications")
        st.markdown(
            """
            - **Supervisor Router**: Classifies request semantics, inspects state payload, and selects optimal subagent.
            - **Vision Analyst**: Processes multimodal inputs (charts, diagrams, OCR, screenshots).
            - **Web Researcher**: Grounding via Tavily / DuckDuckGo for live facts & citation verification.
            - **Code Interpreter**: Sandboxed Python REPL for mathematical verification, data processing, and visualization.
            - **Synthesizer**: Consolidates all visual and retrieved artifacts into an executive response.
            """
        )

# --- Tab 3: Execution Trace & Reasoning ---
with tab_trace:
    st.markdown("### 🔍 Execution Trace & Intermediate Thoughts")
    if not st.session_state.intermediate_steps:
        st.info("No active steps in this session yet. Run a prompt in the console to inspect intermediate reasoning.")
    else:
        for idx, step in enumerate(st.session_state.intermediate_steps):
            node_name = step.get("node", "step")
            with st.expander(f"Step {idx + 1}: [{node_name.upper()}] - {step.get('timestamp', '')} ({step.get('latency_ms', 0)} ms)", expanded=True):
                st.markdown(f"**Thought:** {step.get('thought', '')}")
                if step.get("action"):
                    st.markdown(f"**Action:** `{step.get('action')}`")
                if step.get("output"):
                    st.code(step.get("output"), language="text")

# --- Tab 4: Tool History ---
with tab_tools:
    st.markdown("### 🛠️ Tool Execution & Payloads")
    if not st.session_state.tool_history:
        st.info("No external tools called yet.")
    else:
        for idx, tool in enumerate(st.session_state.tool_history):
            with st.expander(f"Tool {idx + 1}: `{tool.get('tool')}` ({tool.get('latency_ms', 0)} ms)"):
                st.markdown("**Tool Input:**")
                st.json(tool.get("input", {}))
                st.markdown("**Tool Output:**")
                st.json(tool.get("output", {}))

# --- Tab 5: Raw State Inspector ---
with tab_state:
    st.markdown("### 📊 State & Checkpoint Inspector")
    st.markdown("Current internal state variables stored inside LangGraph Memory:")
    st.json({
        "thread_id": st.session_state.thread_id,
        "total_messages": len(st.session_state.messages),
        "extracted_data_keys": list(st.session_state.extracted_data.keys()),
        "has_uploaded_image": st.session_state.uploaded_image is not None,
        "extracted_data": st.session_state.extracted_data,
    })

