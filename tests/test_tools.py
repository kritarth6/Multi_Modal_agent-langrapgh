"""Unit tests for agent tools: python REPL, web search, image utils."""

from PIL import Image
from src.agent.tools.python_repl import execute_python_code
from src.agent.tools.web_search import execute_web_search
from src.agent.tools.flight_tracker import track_flights
from src.agent.tools.image_utils import (
    image_to_base64,
    base64_to_image,
    resize_image_if_needed,
)


def test_python_repl_basic_math():
    res = execute_python_code("x = 10 * 5; print(f'Result: {x}')")
    assert res["success"] is True
    assert "Result: 50" in res["stdout"]
    assert res["variables"].get("x") == "50"


def test_python_repl_error_handling():
    res = execute_python_code("y = 10 / 0")
    assert res["success"] is False
    assert "ZeroDivisionError" in res["error"]


def test_web_search_fallback():
    # Test search with query (either live DDG or fallback returns valid structure)
    res = execute_web_search("LangGraph Python agents", max_results=2)
    assert isinstance(res, list)
    assert len(res) > 0
    assert "title" in res[0]
    assert "snippet" in res[0]


def test_image_utils_roundtrip():
    # Create small test image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    b64 = image_to_base64(img)
    assert isinstance(b64, str)
    assert len(b64) > 0

    reconstructed = base64_to_image(b64)
    assert reconstructed.size == (100, 100)

    # Resize test
    large_img = Image.new("RGB", (2000, 1000), color=(0, 255, 0))
    resized, changed = resize_image_if_needed(large_img, max_dim=1000)
    assert changed is True
    assert max(resized.size) <= 1000


def test_flight_tracker_handling():
    res = track_flights(flight_iata="AA100", limit=1)
    assert isinstance(res, dict)
    assert "success" in res
    assert "data" in res

