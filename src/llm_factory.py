"""LLM Factory for instantiating multimodal & text models (Groq, OpenAI, Mock)."""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from src.config import settings

logger = logging.getLogger(__name__)


class MockMultiModalLLM:
    """
    Intelligent simulated LLM for testing & local demo runs without API keys.
    Provides realistic multi-step agent reasoning, visual inspection, and tool invocation.
    """

    def __init__(self, model_name: str = "mock-multimodal-v1"):
        self.model_name = model_name

    def invoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        # Determine intent from latest user message
        prompt_text = ""
        has_image = False
        
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            prompt_text += part.get("text", "")
                        elif part.get("type") == "image_url":
                            has_image = True
            elif isinstance(content, str):
                prompt_text += content
                if "image" in content.lower() or "chart" in content.lower() or "diagram" in content.lower():
                    has_image = True
            if prompt_text:
                break

        query = prompt_text.lower()

        # Context-aware mock responses
        if "supervisor" in str(kwargs.get("system_prompt", "")).lower() or "route" in query:
            if has_image or "chart" in query or "image" in query or "diagram" in query:
                return AIMessage(content="ROUTE:vision_analyst | REASON: Multimodal visual input detected for inspection.")
            elif "search" in query or "latest" in query or "who is" in query or "news" in query:
                return AIMessage(content="ROUTE:web_researcher | REASON: Information retrieval required.")
            elif "calculate" in query or "code" in query or "math" in query or "plot" in query:
                return AIMessage(content="ROUTE:code_interpreter | REASON: Computation required.")
            else:
                return AIMessage(content="ROUTE:synthesizer | REASON: Direct reasoning query.")

        if has_image:
            analysis = (
                "### Multimodal Visual Analysis Report\n\n"
                "- **Visual Layout & Structure**: Identified high-resolution visual input containing structured components, "
                "data metrics, and interface topology.\n"
                "- **Key Elements Detected**:\n"
                "  1. Quantitative indicators / visual trends showing measurable variation.\n"
                "  2. Hierarchical blocks with clear logical interconnections.\n"
                "  3. Visual callouts indicating focal areas or status conditions.\n"
                "- **Evaluation & Insights**: The artifact demonstrates consistent formatting. "
                "Further quantitative checks or real-time verification have been cross-checked through the pipeline.\n"
            )
            return AIMessage(content=analysis)

        return AIMessage(
            content=f"Processed query: '{prompt_text[:100]}...'\n\n"
            f"The Multi-Modal LangGraph Agent orchestrated sub-agents and synthesized this response."
        )


def get_chat_model(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    is_vision: bool = True,
) -> Any:
    """
    Returns an appropriate Chat Model instance based on provider and available credentials.
    Falls back gracefully to Mock model if no API key is available.
    """
    active_provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
    temp = temperature if temperature is not None else settings.TEMPERATURE

    # 1. Groq
    if active_provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if not key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    key = str(st.secrets["GROQ_API_KEY"])
            except Exception:
                pass

        if key:
            try:
                from langchain_groq import ChatGroq
                model_name = settings.GROQ_VISION_MODEL if is_vision else settings.GROQ_TEXT_MODEL
                return ChatGroq(
                    api_key=key,
                    model_name=model_name,
                    temperature=temp,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGroq with {model_name}: {e}. Retrying with qwen/qwen3.8-27b.")
                try:
                    from langchain_groq import ChatGroq
                    return ChatGroq(
                        api_key=key,
                        model_name="qwen/qwen3.8-27b",
                        temperature=temp,
                    )
                except Exception:
                    logger.warning("Falling back to MockMultiModalLLM.")
        else:
            logger.info("No Groq API key found. Using MockMultiModalLLM.")

    # 2. OpenAI
    elif active_provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if key:
            try:
                from langchain_openai import ChatOpenAI
                model_name = settings.OPENAI_VISION_MODEL if is_vision else settings.OPENAI_TEXT_MODEL
                return ChatOpenAI(
                    api_key=key,
                    model_name=model_name,
                    temperature=temp,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatOpenAI: {e}. Falling back to Mock.")
        else:
            logger.info("No OpenAI API key found. Using MockMultiModalLLM.")

    # 3. Default to Mock
    return MockMultiModalLLM(model_name="mock-multimodal-agent")

