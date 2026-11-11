"""Interactive Command Line Interface for Multi-Modal LangGraph Agent."""

import sys
import uuid
from langchain_core.messages import HumanMessage
from src.agent.graph import build_multimodal_agent_graph
from src.agent.tools.image_utils import image_to_base64
from PIL import Image


def main():
    print("=" * 65)
    print("   🌐 Multi-Modal LangGraph Agent CLI")
    print("=" * 65)
    print("Type your message, or 'exit' / 'quit' to quit.")
    print("To attach an image, prefix with: --image /path/to/img.png <prompt>")
    print("-" * 65)

    graph = build_multimodal_agent_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting. Goodbye!")
                break

            images = []
            if user_input.startswith("--image"):
                parts = user_input.split(" ", 2)
                if len(parts) >= 2:
                    img_path = parts[1]
                    try:
                        pil_img = Image.open(img_path)
                        b64 = image_to_base64(pil_img)
                        images.append({
                            "name": img_path,
                            "media_type": f"image/{pil_img.format.lower() if pil_img.format else 'png'}",
                            "base64_data": b64
                        })
                        print(f"[*] Attached image: {img_path}")
                        user_input = parts[2] if len(parts) > 2 else "Analyze this image"
                    except Exception as e:
                        print(f"[!] Could not load image: {e}")

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "images": images,
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

            print("\n🤖 [Agent Orchestrating...]")
            final_output = None
            for event in graph.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    steps = state_update.get("intermediate_steps", [])
                    if steps:
                        latest = steps[-1]
                        print(f"  ├─ [{node_name}] {latest.get('thought', '')}")
                    if "final_response" in state_update and state_update["final_response"]:
                        final_output = state_update["final_response"]

            print("\n" + "=" * 65)
            print("Response:")
            print(final_output)
            print("=" * 65)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()

