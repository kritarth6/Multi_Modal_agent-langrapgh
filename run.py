"""Unified execution entrypoint for OmniAgent."""

import argparse
import os
import subprocess
import sys


def run_ui():
    """Launch Streamlit frontend dashboard."""
    print("🚀 Launching OmniAgent Interactive Streamlit UI...")
    app_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def run_api():
    """Launch FastAPI backend server."""
    print("🚀 Launching OmniAgent FastAPI Server on http://127.0.0.1:8000 ...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "src.api.server:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ])


def run_cli():
    """Launch interactive terminal CLI."""
    from src.cli import main as cli_main
    cli_main()


def main():
    parser = argparse.ArgumentParser(description="OmniAgent: Multi-Modal LangGraph Multi-Agent System")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit web dashboard")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--cli", action="store_true", help="Launch interactive terminal CLI")

    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.cli:
        run_cli()
    else:
        # Default to interactive UI
        run_ui()


if __name__ == "__main__":
    main()

