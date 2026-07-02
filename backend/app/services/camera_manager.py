import logging
from typing import Dict, Optional

from sqlmodel import Session

from app.core.config import get_settings
from app.services import station_service
from app.services.camera_service import CameraService
from app.vision.detector import PersonDetector
from app.vision.pose import PoseAnalyzer
from app.vision.risk import RiskEvaluator

logger = logging.getLogger(__name__)
settings = get_settings()

class CameraManager:
    """Gestiona múltiples instancias de CameraService y su persistencia de zonas."""
    
    def __init__(self):
        self._cameras: Dict[int, CameraService] = {}
        
        # PoseAnalyzer es stateless, puede compartirse
        self._pose_analyzer = PoseAnalyzer()

    def get_camera(self, camera_id: int, db: Session) -> Optional[CameraService]:
        """Devuelve o inicializa un CameraService para el camera_id solicitado."""
        if camera_id in self._cameras:
            return self._cameras[camera_id]

        # Verificar si tenemos URL configurada para esta cámara
        stream_url = ""
        if camera_id == 1:
            stream_url = settings.CAMERA_1_URL
        elif camera_id == 2:
            stream_url = settings.CAMERA_2_URL
        elif camera_id == 3:
            stream_url = settings.CAMERA_3_URL
            
        if not stream_url:
            logger.warning(f"No hay stream configurado para la cámara {camera_id}.")
            # Se permite crear la instancia aunque no tenga feed (devolverá error o mockup interno)

        # Cargar zonas guardadas en BD para esta cámara
        zones = station_service.get_camera_zones(db, camera_id)
        yellow_points = zones.get("yellow_points", []) if zones else []
        red_points = zones.get("red_points", []) if zones else []

        # Crear instancias independientes para mantener el estado (tracking y tiempos) separado por cámara
        detector = PersonDetector(
            model_name=settings.YOLO_MODEL,
            imgsz=settings.YOLO_IMGSZ,
            conf_threshold=settings.YOLO_CONF_THRESHOLD,
        )
        risk_evaluator = RiskEvaluator(
            loitering_threshold=settings.LOITERING_THRESHOLD_SECONDS,
        )

        service = CameraService(
            camera_id=camera_id,
            stream_url=stream_url,
            detector=detector,
            pose_analyzer=self._pose_analyzer,
            risk_evaluator=risk_evaluator,
            yellow_points=yellow_points,
            red_points=red_points
        )
        
        self._cameras[camera_id] = service
        logger.info(f"Instancia de cámara #{camera_id} inicializada. Zonas precargadas: Amarillas={len(yellow_points)}, Rojas={len(red_points)}")
        
        return service

camera_manager = CameraManager()
