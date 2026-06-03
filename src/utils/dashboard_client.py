"""Thread-safe WebSocket client to push live data to the broker without blocking training."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
import numpy as np
from typing import Any
import websocket  # websocket-client

class DashboardClient:
    """Streams environment states to the WebSocket broker from a background thread."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        n_rows: int = 8,
        n_cols: int = 8,
        trajectory_length: int = 100,
    ):
        self.url = f"ws://{host}:{port}"
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self._trajectory: deque[list[float]] = deque(maxlen=trajectory_length)
        self._last_episode = -1

    def start(self) -> None:
        """Starts the background sender thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the background sender thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def send_state(
        self,
        step: int,
        episode: int,
        algorithm: str,
        target_pos: np.ndarray,
        target_angles: tuple[float, float],
        beam_angles: tuple[float, float],
        gain: float,
        error_deg: float,
        reward: float,
        phases: np.ndarray,
        beam_pattern: dict[str, Any] | None = None,
    ) -> None:
        """Pushes a new state update to the sending queue.

        Drops frames if the queue is full to prevent blocking the training thread.
        """
        if episode != self._last_episode:
            self._trajectory.clear()
            self._last_episode = int(episode)

        self._trajectory.append(
            [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
        )

        if hasattr(phases, "reshape"):
            phase_grid = np.angle(phases).reshape((self.n_rows, self.n_cols)).tolist()
        else:
            phase_grid = phases

        state_data = {
            "type": "state_update",
            "algorithm": algorithm,
            "episode": int(episode),
            "step": int(step),
            "metrics": {
                "gain": float(gain),
                "error_deg": float(error_deg),
                "reward": float(reward)
            },
            "phases": phase_grid,
            "beam_angles": [float(beam_angles[0]), float(beam_angles[1])],
            "target_angles": [float(target_angles[0]), float(target_angles[1])],
            "target": {
                "x": float(target_pos[0]),
                "y": float(target_pos[1]),
                "z": float(target_pos[2]),
                "trajectory": list(self._trajectory),
            }
        }
        
        if beam_pattern:
            state_data["beam_pattern"] = {
                "theta": beam_pattern["theta"].tolist() if isinstance(beam_pattern["theta"], np.ndarray) else beam_pattern["theta"],
                "phi": beam_pattern["phi"].tolist() if isinstance(beam_pattern["phi"], np.ndarray) else beam_pattern["phi"],
                "power": beam_pattern["power"].tolist() if isinstance(beam_pattern["power"], np.ndarray) else beam_pattern["power"]
            }

        try:
            self.queue.put_nowait(state_data)
        except queue.Full:
            # Drop frame to prioritize training throughput
            pass

    def _run(self) -> None:
        """Background thread main loop."""
        ws_conn = None
        while self.running:
            # Try to connect
            try:
                # We use simple websocket-client block interface
                ws_conn = websocket.create_connection(self.url, timeout=2.0)
            except Exception:
                # Connection failed, wait and retry
                time.sleep(2.0)
                continue

            # Connection succeeded, process queue
            while self.running:
                try:
                    data = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    ws_conn.send(json.dumps(data))
                except Exception:
                    # Send failed (broken pipe, etc.), break out to reconnect
                    try:
                        ws_conn.close()
                    except Exception:
                        pass
                    break
            
            if ws_conn:
                try:
                    ws_conn.close()
                except Exception:
                    pass
