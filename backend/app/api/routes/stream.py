"""
Rutas de streaming de video y configuración de zonas — Andén Seguro.

Endpoints para el feed MJPEG, estadísticas en tiempo real,
configuración de polígonos de zona y canal WebSocket.
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.core.config import get_settings
from app.core.websocket_manager import ws_manager
from app.db.session import get_db
from app.schemas.stream import CameraStats, ZoneConfig
from app.services import station_service
from app.services.camera_service import CameraService
from app.vision.detector import PersonDetector
from app.vision.pose import PoseAnalyzer
from app.vision.risk import RiskEvaluator

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@lru_cache()
def get_camera_service() -> CameraService:
    """Singleton: Construye el servicio de cámara con el pipeline de IA."""
    detector = PersonDetector(
        model_name=settings.YOLO_MODEL,
        imgsz=settings.YOLO_IMGSZ,
        conf_threshold=settings.YOLO_CONF_THRESHOLD,
    )
    pose_analyzer = PoseAnalyzer()
    risk_evaluator = RiskEvaluator(
        loitering_threshold=settings.LOITERING_THRESHOLD_SECONDS,
    )

    return CameraService(
        detector=detector,
        pose_analyzer=pose_analyzer,
        risk_evaluator=risk_evaluator,
        app_key=settings.APP_KEY,
        app_secret=settings.APP_SECRET,
        serial=settings.SERIAL,
        base_url=settings.BASE_URL,
    )


@router.get("/video_feed")
def video_feed(service: CameraService = Depends(get_camera_service)):
    """Endpoint de streaming en tiempo real. Consumible como imagen MJPEG."""
    return StreamingResponse(
        service.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/config")
def get_zone_config(
    service: CameraService = Depends(get_camera_service),
    camera_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Retorna la configuración de zonas.
    Consulta DB si se indica camera_id, sino retorna lo que hay en memoria.
    """
    if camera_id is not None:
        return station_service.get_camera_zones(db, camera_id)
    return {
        "yellow_points": service.yellow_points,
        "red_points": service.red_points,
    }


@router.get("/stats", response_model=CameraStats)
def get_stats(service: CameraService = Depends(get_camera_service)):
    """Retorna las estadísticas en tiempo real del análisis de la cámara."""
    return service.current_stats


@router.post("/config")
def update_zone_config(
    config: ZoneConfig,
    service: CameraService = Depends(get_camera_service),
    camera_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Actualiza la configuración de zonas.
    Persiste en DB si se indica camera_id.
    """
    # Actualizar en memoria del servicio activo
    service.yellow_points = config.yellow_points
    service.red_points = config.red_points

    # Persistir en DB si se indica cámara
    if camera_id is not None:
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