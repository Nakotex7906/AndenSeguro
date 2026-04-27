import os
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from app.services.camera_service import CameraService
from functools import lru_cache

router = APIRouter()

@lru_cache()
def get_yolo_model():
    """Patrón Singleton: Carga el modelo una sola vez en memoria."""
    print("[INFO] Cargando modelo YOLOv8...")
    return YOLO("yolov8n.pt")

def get_camera_service(model: YOLO = Depends(get_yolo_model)) -> CameraService:
    """Proveedor de la instancia del servicio con todas sus dependencias."""
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