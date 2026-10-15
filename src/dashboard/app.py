"""Argus Quant — Unified Institutional Trading Terminal.

Launches the high-performance ASGI server (FastAPI / Starlette + Uvicorn)
serving the modern HTML5 / CSS / Vanilla JavaScript / Tailwind CSS trading platform.
"""
import sys
from pathlib import Path

import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(host: str = "0.0.0.0", port: int = 8501):
    """Starts the Uvicorn ASGI server hosting the terminal."""
    print("\n================ ARGUS QUANT TRADING TERMINAL ================")
    print(f"URL:      http://localhost:{port}")
    print("Stack:    HTML5 | CSS3 | Vanilla JS | Tailwind CSS | FastAPI")
    print("Press Ctrl+C to stop.")
    print("===============================================================\n")
    uvicorn.run("src.dashboard.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Argus Quant Trading Terminal Web Server")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind server (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args, _ = parser.parse_known_args()
    main(host=args.host, port=args.port)
