"""Configuration management using Pydantic Settings."""

import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Default LLM Provider: 'groq', 'openai', or 'mock'
    DEFAULT_LLM_PROVIDER: Literal["groq", "openai", "mock"] = Field(
        default="groq",
        description="Default LLM provider to use if not overridden in request",
    )

    # API Keys
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API Key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily API Key for web search")
    AVIATIONSTACK_API_KEY: Optional[str] = Field(default=None, description="Aviationstack API Key for flight tracking")
    DATABASE_URL: Optional[str] = Field(default=None, description="PostgreSQL / Database connection string")

    # Model Defaults
    GROQ_VISION_MODEL: str = "llama-3.2-11b-vision-preview"
    GROQ_TEXT_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_VISION_MODEL: str = "gpt-4o"
    OPENAI_TEXT_MODEL: str = "gpt-4o-mini"

    # Agent Settings
    MAX_ITERATIONS: int = 10
    TEMPERATURE: float = 0.2
    EXECUTION_TIMEOUT_SECONDS: int = 30
    SAFE_CODE_EXECUTION: bool = True

    # Server Settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 8501


settings = Settings()
