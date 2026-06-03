"""WebSocket broker and optional static file server for the beamforming dashboard."""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import logging
import os
import threading
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Broker")

# Keep track of all connected websockets
connected_clients = set()


async def register(websocket):
    """Registers a new client connection."""
    connected_clients.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address}. Total clients: {len(connected_clients)}")


async def unregister(websocket):
    """Unregisters a client connection."""
    connected_clients.remove(websocket)
    logger.info(f"Client disconnected: {websocket.remote_address}. Total clients: {len(connected_clients)}")


async def broker_handler(websocket):
    """Handles incoming messages and broadcasts them to all other clients."""
    await register(websocket)
    try:
        async for message in websocket:
            # Broadcast the received message to all OTHER connected clients
            if not connected_clients:
                continue
            
            # Create a list of send tasks for all other sockets
            other_clients = [client for client in connected_clients if client != websocket]
            if other_clients:
                # Use gather to send concurrently
                await asyncio.gather(
                    *[client.send(message) for client in other_clients],
                    return_exceptions=True
                )
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        await unregister(websocket)


def start_static_server(host: str, port: int, directory: str) -> http.server.ThreadingHTTPServer:
    """Serves the dashboard HTML/CSS/JS over HTTP."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=directory,
    )
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


async def main():
    parser = argparse.ArgumentParser(description="Launch WebSocket broker for the beamforming dashboard.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to run WebSocket server.")
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for the dashboard static file server (0 to disable).",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dashboard_dir = os.path.join(project_root, "dashboard")

    httpd = None
    if args.http_port > 0:
        httpd = start_static_server(args.host, args.http_port, dashboard_dir)
        logger.info(f"Dashboard UI at http://{args.host}:{args.http_port}/")

    logger.info(f"WebSocket broker at ws://{args.host}:{args.port}")

    try:
        async with websockets.serve(broker_handler, args.host, args.port):
            await asyncio.Future()
    finally:
        if httpd is not None:
            httpd.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Broker server stopped by user.")
