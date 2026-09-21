#!/usr/bin/env python3
"""One-command launcher for the Veritas visual dashboard.

This script:
  1. (Re)generates ``demo/graph_export.json`` from the sample dataset.
  2. Starts a local static HTTP server rooted at the ``demo/`` folder.
  3. Opens the dashboard in the default browser.

It uses only the Python standard library, so no ``pip install`` is required to
run the demo.

Usage
-----
    python3 demo/serve_dashboard.py
    python3 demo/serve_dashboard.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

import build_graph_export  # local module (same directory)

DEMO_DIR = Path(__file__).resolve().parent


def regenerate_export() -> None:
    """Rebuild the graph export so the dashboard always shows fresh data."""
    print("Generating knowledge-graph export...")
    build_graph_export.main()


def serve(port: int, open_browser: bool) -> None:
    """Serve the demo directory over HTTP until interrupted."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DEMO_DIR))

    # Allow quick restarts without "address already in use".
    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    with Server(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/dashboard.html"
        print("=" * 60)
        print("  Veritas dashboard is live at:")
        print(f"    {url}")
        print("  Press Ctrl+C to stop.")
        print("=" * 60)
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Veritas visual dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    parser.add_argument("--skip-export", action="store_true", help="Skip regenerating the export.")
    args = parser.parse_args()

    if not args.skip_export:
        regenerate_export()

    serve(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
