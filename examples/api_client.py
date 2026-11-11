"""Example Python client interacting with the OmniAgent FastAPI REST API."""

import base64
import os
import requests


API_URL = "http://127.0.0.1:8000"


def test_health():
    """Verify backend health and active integrations."""
    resp = requests.get(f"{API_URL}/api/health")
    print("\n--- [GET /api/health] ---")
    print(resp.json())


def test_text_chat():
    """Send text query to multi-agent workflow."""
    payload = {
        "message": "What are the latest best practices for multi-agent architectures in LangGraph?",
        "thread_id": "demo-session-text-01",
    }
    print("\n--- [POST /api/chat (Text)] ---")
    resp = requests.post(f"{API_URL}/api/chat", json=payload)
    data = resp.json()
    print("Response:")
    print(data.get("response"))
    print("\nIntermediate Nodes Visited:")
    for step in data.get("intermediate_steps", []):
        print(f"  [{step.get('node')}] -> {step.get('thought')}")


def test_multimodal_chat(image_path: str):
    """Send multimodal image + query to agent."""
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "message": "Perform an architectural critique of this system design and identify vulnerabilities.",
        "thread_id": "demo-session-vision-01",
        "images": [
            {
                "name": os.path.basename(image_path),
                "media_type": "image/png",
                "base64_data": b64
            }
        ]
    }
    print("\n--- [POST /api/chat (Multimodal)] ---")
    resp = requests.post(f"{API_URL}/api/chat", json=payload)
    data = resp.json()
    print("Response:")
    print(data.get("response"))


if __name__ == "__main__":
    test_health()
    test_text_chat()
    sample_img = os.path.join(os.path.dirname(__file__), "..", "frontend", "sample_images", "sample_architecture.png")
    test_multimodal_chat(sample_img)

