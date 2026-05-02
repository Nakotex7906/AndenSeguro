import os
from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple
from ultralytics import YOLO
from app.services.camera_service import CameraService
from functools import lru_cache

router = APIRouter()

class ZoneConfig(BaseModel):
    yellow_points: List[Tuple[float, float]]
    red_points: List[Tuple[float, float]]

@lru_cache()
def get_yolo_model():
    """Patrón Singleton: Carga el modelo una sola vez en memoria."""
    print("[INFO] Cargando modelo YOLOv8...")
    return YOLO("yolov8n.pt")

@lru_cache()
def get_camera_service() -> CameraService:
    """Proveedor de la instancia del servicio como Singleton."""
    model = get_yolo_model()
    return CameraService(
        model=model,
        app_key=os.getenv("APP_KEY"),
        app_secret=os.getenv("APP_SECRET"),
        serial=os.getenv("SERIAL"),
        base_url=os.getenv("BASE_URL", "https://isaopen.ezvizlife.com")
    )

@router.get("/video_feed")
def video_feed(service: CameraService = Depends(get_camera_service)):
    """
    Endpoint de streaming en tiempo real.
    Consumible desde el frontend como una imagen MJPEG.
    """
    return StreamingResponse(
        service.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/config")
def get_zone_config(service: CameraService = Depends(get_camera_service)):
    """Devuelve la configuración actual de polígonos."""
    return {"yellow_points": service.yellow_points, "red_points": service.red_points}

@router.post("/config")
def update_zone_config(config: ZoneConfig, service: CameraService = Depends(get_camera_service)):
    """Actualiza la configuración actual y la guarda."""
    import json
    # Guardar en memoria
    service.yellow_points = config.yellow_points
    service.red_points = config.red_points
    
    # Persistir en archivo JSON
    with open(service.config_file, "w") as f:
        json.dump({
            "yellow_points": service.yellow_points,
            "red_points": service.red_points
        }, f)
        
    return {"status": "success", "message": "Zonas actualizadas correctamente."}