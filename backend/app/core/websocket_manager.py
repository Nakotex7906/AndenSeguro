"""
Gestor de conexiones WebSocket — Andén Seguro.

Administra las conexiones activas del frontend y permite
emitir alertas en tiempo real mediante broadcast.
"""

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Registra clientes WebSocket y envía mensajes a todos simultáneamente."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        """Acepta y registra una nueva conexión WebSocket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self._loop = asyncio.get_event_loop()
        logger.info(
            f"WebSocket conectado. Total clientes: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remueve una conexión WebSocket del registro."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket desconectado. Total clientes: {len(self.active_connections)}"
        )

    async def _broadcast(self, message: dict):
        """Envía un mensaje JSON a todas las conexiones activas (interno)."""
        data = json.dumps(message)
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    def broadcast_sync(self, message: dict):
        """Emite un broadcast desde código síncrono (e.g., generate_frames)."""
        if self._loop and self.active_connections:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def broadcast(self, message: dict):
        """Emite un broadcast desde código asíncrono."""
        await self._broadcast(message)


# Instancia singleton global
ws_manager = ConnectionManager()
