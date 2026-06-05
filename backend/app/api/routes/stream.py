"""
Rutas de streaming de video y configuración de zonas — Andén Seguro.

Endpoints para el feed MJPEG, estadísticas en tiempo real,
configuración de polígonos de zona y canal WebSocket.
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.core.websocket_manager import ws_manager
from app.db.session import get_db
from app.schemas.stream import CameraStats, ZoneConfig
from app.services import station_service
from app.services.camera_manager import camera_manager

logger = logging.getLogger(__name__)

router = APIRouter()

def get_service_or_404(camera_id: int, db: Session) -> "CameraService":
    service = camera_manager.get_camera(camera_id, db)
    if not service:
        raise HTTPException(status_code=404, detail="Cámara no encontrada o no configurada.")
    return service


@router.get("/video_feed/{camera_id}")
def video_feed(camera_id: int, db: Session = Depends(get_db)):
    """Endpoint de streaming en tiempo real para una cámara. Consumible como imagen MJPEG."""
    service = get_service_or_404(camera_id, db)
    return StreamingResponse(
        service.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/config/{camera_id}")
def get_zone_config(
    camera_id: int,
    db: Session = Depends(get_db),
):
    """
    Retorna la configuración de zonas activa en memoria para la cámara solicitada.
    """
    service = get_service_or_404(camera_id, db)
    return {
        "yellow_points": service.yellow_points,
        "red_points": service.red_points,
    }


@router.get("/stats/{camera_id}", response_model=CameraStats)
def get_stats(camera_id: int, db: Session = Depends(get_db)):
    """Retorna las estadísticas en tiempo real del análisis de la cámara específica."""
    service = get_service_or_404(camera_id, db)
    return service.current_stats


@router.post("/config/{camera_id}")
def update_zone_config(
    camera_id: int,
    config: ZoneConfig,
    db: Session = Depends(get_db),
):
    """
    Actualiza la configuración de zonas y persiste en DB.
    """
    service = get_service_or_404(camera_id, db)
    
    # Actualizar en memoria del servicio activo
    service.yellow_points = config.yellow_points
    service.red_points = config.red_points

    # Persistir en DB
    station_service.update_camera_zones(
        db,
        camera_id,
        yellow_points=[list(p) for p in config.yellow_points],
        red_points=[list(p) for p in config.red_points],
    )
    logger.info(f"Zonas guardadas en DB para cámara #{camera_id}")

    return {"status": "success", "message": "Zonas actualizadas correctamente."}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para recibir alertas en tiempo real desde el pipeline de IA."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Mantener la conexión abierta; el servidor envía mensajes proactivamente
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)