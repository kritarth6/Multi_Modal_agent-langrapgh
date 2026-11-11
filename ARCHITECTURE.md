# OmniAgent: Technical Architecture & System Design

## 1. System Overview

**OmniAgent** is an enterprise-grade Multi-Modal Multi-Agent system orchestrated via **LangGraph**. It combines visual perception, algorithmic code execution, dynamic web grounding, and hierarchical intent routing into a deterministic, stateful workflow.

```
                      +-----------------------------+
                      |   User Input & Artifacts    |
                      |  (Text Prompt + Multi-Image)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Supervisor Node        |
                      |  (Intent & Priority Router) |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+       +-------------------+       +--------------------+
|  Vision Analyst  |       |  Web Researcher   |       |  Code Interpreter  |
| (Llama 3.2 Vision|       | (Tavily & DDGS    |       |  (Sandboxed REPL   |
|   / GPT-4o)      |       |  Live Retrieval)  |       |   & Math Engine)   |
+--------+---------+       +---------+---------+       +---------+----------+
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Synthesizer Node       |
                      | (Multi-Artifact Integration)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Executive Final Output   |
                      +-----------------------------+
```

---

## 2. LangGraph State Machine Design

### 2.1 State Schema (`AgentState`)

The agent's state is encapsulated in a typed schema with explicit functional reducers:

```python
class AgentState(TypedDict):
    # Chat history with append reducer
    messages: Annotated[List[BaseMessage], add_messages]

    # Multimodal artifacts (images, charts, screenshots)
    images: List[Dict[str, Any]]

    # Routing tracking
    current_node: str
    target_node: Optional[str]

    # Execution trace for UI & observability
    intermediate_steps: Annotated[List[Dict[str, Any]], append_list]

    # Tool invocation history
    tool_history: Annotated[List[Dict[str, Any]], append_list]

    # Extracted data (OCR text, code results, search facts)
    extracted_data: Annotated[Dict[str, Any], merge_dicts]

    # Human-in-the-loop validation
    requires_human_approval: bool
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # Output artifact
    final_response: Optional[str]
    error: Optional[str]
```

### 2.2 Reducers & Immutability
- `add_messages`: Automatically reconciles conversation history without manual list concatenation.
- `append_list`: Accumulates step traces and tool execution records chronologically.
- `merge_dicts`: Preserves state artifacts produced across separate nodes without overwriting prior extractions.

---

## 3. Specialized Subagent Nodes

### 3.1 Supervisor Router (`supervisor_node`)
- **Objective**: Evaluates input semantics, message history, and attached image metadata.
- **Routing Logic**:
  - Unprocessed images -> routes to `vision_analyst`.
  - Analytical questions requesting formulas or code execution -> routes to `code_interpreter`.
  - Queries requiring real-time facts or external validation -> routes to `web_researcher`.
  - General conversational queries -> routes directly to `synthesizer`.

### 3.2 Vision Analyst (`vision_analyst_node`)
- **Objective**: Deep visual reasoning over uploaded graphics.
- **Payload Formatting**: Constructs standard multimodal content blocks `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.
- **Capabilities**:
  - Financial charts (axes, trends, support/resistance, CAGR estimations).
  - Software architectures (component mapping, latency bottlenecks, single points of failure).
  - UI/UX error screenshots (traceback parsing, broken layout detection).
  - Dense document & diagram OCR.

### 3.3 Web Researcher (`web_researcher_node`)
- **Objective**: Grounds agent reasoning with live authoritative internet data.
- **Dual-Engine Strategy**: Primary retrieval via **Tavily Search API**, falling back gracefully to **DuckDuckGo Search** (zero API key dependency).

### 3.4 Code Interpreter (`code_interpreter_node`)
- **Objective**: Deterministic calculation, statistical verification, and data manipulation.
- **Execution Sandbox**: Restricted global namespace with built-in functions, `math`, `datetime`, `pandas`, `numpy`, and stdout interception.

### 3.5 Synthesizer (`synthesizer_node`)
- **Objective**: Collates observations from vision, code execution, and web research into a cohesive, executive markdown response with clear takeaways.

---

## 4. Multi-Provider LLM Engine

The system supports a unified provider architecture via `src/llm_factory.py`:
1. **Groq Cloud**: Ultra-low latency inference using `llama-3.2-11b-vision-preview` and `llama-3.3-70b-versatile`.
2. **OpenAI**: High-capacity multimodal reasoning with `gpt-4o` and `gpt-4o-mini`.
3. **Mock Offline Mode**: Built-in simulated engine generating realistic step-by-step traces for zero-setup local testing.

---

## 5. Security & Isolation

- **Sandboxed Execution**: Custom Python REPL isolates code execution environment from system commands.
- **Sanitized Multimodal Payloads**: Enforces maximum dimension checks (`1536px`) and formats images to standard base64 data URLs to prevent memory exhaustion.
- **Credential Masking**: API keys are injected via environment variables or encrypted session state and never committed or logged in plaintext.

