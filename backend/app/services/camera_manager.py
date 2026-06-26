import logging
from typing import Dict, Optional

from sqlmodel import Session

from app.core.config import get_settings
from app.models.station import Camera
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

    def _get_camera_record(self, db: Session, camera_id: int) -> Optional[Camera]:
        """Obtiene el registro persistido de una cámara.

        Args:
            db (Session): Sesión activa de SQLModel.
            camera_id (int): Identificador de la cámara solicitada.

        Returns:
            Optional[Camera]: Cámara encontrada en base de datos o ``None`` si no existe.
        """
        return station_service.get_camera(db, camera_id)

    def _resolve_stream_url(self, camera: Camera) -> Optional[str]:
        """Valida y normaliza la ruta o URL de streaming persistida.

        Args:
            camera (Camera): Registro persistido de la cámara.

        Returns:
            Optional[str]: Stream válido o ``None`` si el campo está vacío.
        """
        stream_url = (camera.stream_url or "").strip()
        if not stream_url:
            logger.warning(
                "No hay stream configurado en camera.stream_url para la cámara %s.",
                camera.id,
            )
            return None
        return stream_url

    def _load_camera_zones(self, db: Session, camera_id: int) -> tuple[list, list]:
        """Carga desde BD los polígonos de zona de una cámara.

        Args:
            db (Session): Sesión activa de SQLModel.
            camera_id (int): Identificador de la cámara solicitada.

        Returns:
            tuple[list, list]: Puntos amarillos y rojos asociados a la cámara.
        """
        zones = station_service.get_camera_zones(db, camera_id)
        yellow_points = zones.get("yellow_points", []) if zones else []
        red_points = zones.get("red_points", []) if zones else []
        return yellow_points, red_points

    def _build_camera_service(
        self,
        camera_id: int,
        stream_url: str,
        yellow_points: list,
        red_points: list,
    ) -> CameraService:
        """Crea una instancia de CameraService con dependencias explícitas.

        Args:
            camera_id (int): Identificador de la cámara.
            stream_url (str): Ruta o URL del feed de video.
            yellow_points (list): Polígono amarillo serializado.
            red_points (list): Polígono rojo serializado.

        Returns:
            CameraService: Servicio inicializado y listo para operar.
        """
        detector = PersonDetector(
            model_name=settings.YOLO_MODEL,
            imgsz=settings.YOLO_IMGSZ,
            conf_threshold=settings.YOLO_CONF_THRESHOLD,
        )
        risk_evaluator = RiskEvaluator(
            loitering_threshold=settings.LOITERING_THRESHOLD_SECONDS,
        )

        return CameraService(
            camera_id=camera_id,
            stream_url=stream_url,
            detector=detector,
            pose_analyzer=self._pose_analyzer,
            risk_evaluator=risk_evaluator,
            yellow_points=yellow_points,
            red_points=red_points,
        )

    def get_camera(self, camera_id: int, db: Session) -> Optional[CameraService]:
        """Devuelve o inicializa un CameraService para el camera_id solicitado.

        Args:
            camera_id (int): Identificador de la cámara requerida.
            db (Session): Sesión activa de SQLModel.

        Returns:
            Optional[CameraService]: Servicio inicializado o ``None`` si la cámara no existe o no tiene stream.
        """
        if camera_id in self._cameras:
            return self._cameras[camera_id]

        camera = self._get_camera_record(db, camera_id)
        if camera is None:
            logger.error("No se encontró la cámara %s en la base de datos.", camera_id)
            return None

        stream_url = self._resolve_stream_url(camera)
        if stream_url is None:
            return None

        yellow_points, red_points = self._load_camera_zones(db, camera_id)
        service = self._build_camera_service(
            camera_id=camera_id,
            stream_url=stream_url,
            yellow_points=yellow_points,
            red_points=red_points,
        )
        
        self._cameras[camera_id] = service
        logger.info(f"Instancia de cámara #{camera_id} inicializada. Zonas precargadas: Amarillas={len(yellow_points)}, Rojas={len(red_points)}")
        
        return service

camera_manager = CameraManager()
