"""Helper script to generate rich synthetic test images for 1-click demos."""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def generate_all_samples():
    os.makedirs("frontend/sample_images", exist_ok=True)

    # 1. Financial Chart
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    np.random.seed(42)
    days = np.arange(1, 31)
    prices = 150 + np.cumsum(np.random.randn(30) * 3)
    ax.plot(days, prices, color="#38bdf8", linewidth=2.5, label="TechIndex ETF (USD)")
    ax.fill_between(days, prices, 140, color="#38bdf8", alpha=0.15)
    ax.set_title("Q3 Performance Metrics - Revenue & Price Trajectory", color="#f8fafc", fontsize=12, pad=10)
    ax.set_xlabel("Trading Days", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Price ($)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    plt.tight_layout()
    chart_path = "frontend/sample_images/sample_financial_chart.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"Generated {chart_path}")

    # 2. System Architecture Diagram
    img = Image.new("RGB", (700, 380), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    # Client Box
    draw.rounded_rectangle([40, 140, 170, 240], radius=10, fill=(30, 41, 59), outline=(56, 189, 248), width=2)
    draw.text((55, 180), "Client Frontend\n(Streamlit/React)", fill=(248, 250, 252))
    # API Gateway Box
    draw.rounded_rectangle([230, 130, 380, 250], radius=10, fill=(30, 41, 59), outline=(129, 140, 248), width=2)
    draw.text((245, 175), "FastAPI Gateway\n& Auth Proxy", fill=(248, 250, 252))
    # LangGraph Core
    draw.rounded_rectangle([440, 90, 660, 300], radius=12, fill=(30, 41, 59), outline=(52, 211, 153), width=2)
    draw.text((465, 110), "LangGraph Multi-Agent Cluster", fill=(52, 211, 153))
    draw.rectangle([460, 150, 640, 195], fill=(51, 65, 85))
    draw.text((475, 165), "Vision Analyst Node", fill=(248, 250, 252))
    draw.rectangle([460, 215, 640, 260], fill=(51, 65, 85))
    draw.text((475, 230), "Web Researcher Node", fill=(248, 250, 252))
    # Connector arrows
    draw.line([(170, 190), (230, 190)], fill=(148, 163, 184), width=3)
    draw.line([(380, 190), (440, 190)], fill=(148, 163, 184), width=3)
    arch_path = "frontend/sample_images/sample_architecture.png"
    img.save(arch_path)
    print(f"Generated {arch_path}")

    # 3. Bug Screenshot Mockup
    img_bug = Image.new("RGB", (680, 360), color=(30, 41, 59))
    draw_b = ImageDraw.Draw(img_bug)
    draw_b.rectangle([0, 0, 680, 40], fill=(15, 23, 42))
    draw_b.ellipse([15, 12, 27, 24], fill=(239, 68, 68))
    draw_b.ellipse([35, 12, 47, 24], fill=(234, 179, 8))
    draw_b.ellipse([55, 12, 67, 24], fill=(34, 197, 94))
    draw_b.text((90, 12), "App Dashboard - Console Error Caught", fill=(148, 163, 184))
    draw_b.rounded_rectangle([40, 70, 640, 310], radius=8, fill=(15, 23, 42), outline=(239, 68, 68), width=2)
    draw_b.text((60, 95), "Error: Uncaught TypeError: Cannot read properties of undefined (reading 'chartData')", fill=(248, 113, 113))
    draw_b.text((60, 130), "   at AnalyticsDashboard.render (dashboard.tsx:142:19)", fill=(203, 213, 225))
    draw_b.text((60, 160), "   at react-dom.development.js:15420:23", fill=(148, 163, 184))
    draw_b.text((60, 200), "State Payload: { user: 'admin', metrics: null, status: 200 }", fill=(251, 191, 36))
    draw_b.text((60, 240), "Root Cause: metrics object returned null from upstream API before mount.", fill=(148, 163, 184))
    # 4. Aviation Boarding Pass / Flight Card
    img_flight = Image.new("RGB", (650, 320), color=(15, 23, 42))
    draw_f = ImageDraw.Draw(img_flight)
    draw_f.rounded_rectangle([30, 30, 620, 290], radius=12, fill=(30, 41, 59), outline=(56, 189, 248), width=2)
    draw_f.text((50, 50), "GLOBAL AIRWAYS - BOARDING PASS & FLIGHT TELEMETRY", fill=(56, 189, 248))
    draw_f.text((50, 90), "Flight: AA100 / DAL412", fill=(248, 250, 252))
    draw_f.text((50, 120), "Origin: JFK (New York) -> Destination: LHR (London Heathrow)", fill=(203, 213, 225))
    draw_f.text((50, 150), "Status: EN ROUTE (Altitude: 36,000 ft, Ground Speed: 540 kts)", fill=(52, 211, 153))
    draw_f.text((50, 190), "Aircraft: Boeing 787-9 Dreamliner", fill=(148, 163, 184))
    draw_f.text((50, 220), "Gate: B24 | Terminal: 4 | Estimated Arrival: On Time (19:45 UTC)", fill=(251, 191, 36))
    flight_path = "frontend/sample_images/sample_flight_telemetry.png"
    img_flight.save(flight_path)
    print(f"Generated {flight_path}")


if __name__ == "__main__":
    generate_all_samples()

