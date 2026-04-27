import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.camera_service import CameraStreamer

# 1. ESTA ES LA LÍNEA CLAVE QUE ESTÁ BUSCANDO MAIN.PY
router = APIRouter()

camera_service = CameraStreamer(
    app_key=os.getenv("APP_KEY"),
    app_secret=os.getenv("APP_SECRET"),
    serial=os.getenv("SERIAL"),
    base_url=os.getenv("BASE_URL", "https://isaopen.ezvizlife.com")
)

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        camera_service.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )