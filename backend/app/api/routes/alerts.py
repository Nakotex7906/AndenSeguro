"""
Rutas de Websocket para Alertas y Métricas — Andén Seguro.

Transmite actualizaciones del dashboard y alertas en tiempo real al frontend.
"""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.core.websocket_manager import ws_manager
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws")
async def alerts_websocket_endpoint(
    websocket: WebSocket,
    # db: Session no se inyecta directamente de get_db de forma limpia en websockets 
    # si se mantiene abierto indefinidamente. Es mejor crear sesiones on-demand.
):
    """
    WebSocket dedicado para recibir incidentes activos y métricas globales
    del dashboard en tiempo real (reemplaza a los simuladores del frontend).
    """
    from app.db.session import engine
    
    await ws_manager.connect(websocket)
    try:
        while True:
            # Enviar métricas globales periódicamente (DashboardOverview)
            # Para esto llamamos a la misma lógica de dashboard.py
            from app.api.routes.dashboard import get_dashboard_overview
            
            # Usar una sesión fresca por cada iteración para obtener datos actualizados
            with Session(engine) as session:
                overview = get_dashboard_overview(db=session)
            
            payload = {
                "type": "dashboard_metrics",
                "data": overview.model_dump()
            }
            
            await websocket.send_json(payload)
            
            # Esperar 2 segundos antes de volver a enviar
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error en websocket de alertas: {e}")
        ws_manager.disconnect(websocket)
