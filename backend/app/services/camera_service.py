"""
Servicio de cámara refactorizado — Andén Seguro.

Responsabilidad única: coordinar la lectura del feed de video,
orquestar el pipeline de IA (detector → pose → riesgo) y despachar
los incidentes críticos de manera asíncrona hacia servicios externos.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List, Optional, Union

import cv2
import numpy as np
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.websocket_manager import ws_manager
from app.db.session import engine
from app.models.user import User
from app.services.incident_service import create_incident
from app.services.notification_service import notification_service
from app.services.vision_service import vision_service
from app.vision.detector import PersonDetector
from app.vision.pose import PoseAnalyzer
from app.vision.risk import RiskEvaluator, RiskLevel

logger = logging.getLogger(__name__)
settings = get_settings()


class CameraService:
    """Coordina la captura de video en tiempo real y la ejecución del pipeline de IA."""

    # Propiedades para el control para no saturar la API de Groq con demasiadas solicitudes simultáneas
    _last_global_alert_time: float = 0.0
    _global_cooldown_seconds: float = 60.0 # Maximo una alerta por minuto 
    _global_cooldown_lock = threading.Lock()

    def __init__(
        self,
        camera_id: int,
        stream_url: str,
        detector: PersonDetector,
        pose_analyzer: PoseAnalyzer,
        risk_evaluator: RiskEvaluator,
        yellow_points: list = None,
        red_points: list = None,
    ):
        """
        Inicializa las dependencias de hardware virtual, modelos y estados de la cámara.
        """
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.detector = detector
        self.pose_analyzer = pose_analyzer
        self.risk_evaluator = risk_evaluator

        # Estado de zonas cargadas desde DB
        self.yellow_points = yellow_points or []
        self.red_points = red_points or []

        # Estadísticas en tiempo real
        self.current_stats = {
            "total_persons": 0,
            "risk_persons": 0,
            "danger_persons": 0,
        }

        # Cooldown de alertas para evitar duplicados (track_id -> last_alert_time)
        self._alert_cooldown: dict[int, float] = {}
        self._alert_cooldown_seconds = 10.0
        
        # Executor de hilos secundario para I/O intensivo sin bloquear el streaming
        self._background_executor = ThreadPoolExecutor(max_workers=3)

    def _resolve_video_source(self) -> Union[str, int]:
        """
        Determina la fuente de video analizando si corresponde a una URL externa o local.

        Returns:
            Union[str, int]: URL del stream, endpoint de video o índice de hardware.
        """
        source = self.stream_url

        if source:
            logger.info(f"Cámara #{self.camera_id} - Usando stream: {source}")
            if "youtube.com" in source or "youtu.be" in source:
                try:
                    import yt_dlp
                    logger.info("Extrayendo URL cruda de YouTube con yt-dlp...")
                    ydl_opts = {"format": "best[ext=mp4]", "quiet": True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(source, download=False)
                        source = info["url"]
                except ImportError:
                    logger.error("Falta instalar yt-dlp. Ejecuta: pip install yt-dlp")
            return source

        logger.warning(f"Cámara #{self.camera_id} - No hay stream configurado. Usando cámara local 0")
        return 0

    def _should_emit_alert(self, track_id: Optional[int]) -> bool:
        """
        Verifica si se debe emitir una alerta basándose en una política de cooldown anti-spam.

        Garantiza la protección de la API de Groq limitando el despacho a un máximo de
        una alerta por minuto a nivel global en todo el sistema. Adicionalmente, verifica
        que cada sujeto individual (track_id) haya cumplido su periodo de gracia particular.

        Args:
            track_id (Optional[int]): Identificador único del sujeto en riesgo detectado por el modelo.

        Returns:
            bool: True si la alerta supera con éxito los filtros de tiempo global e
                  individual y puede ser procesada; False en caso contrario.
        """
        current_time = time.time()

        with CameraService._global_cooldown_lock:
            
            # Protección para no saturar la api de Groq
            time_since_last_global_alert = current_time - CameraService._last_global_alert_time
            if time_since_last_global_alert < CameraService._global_cooldown_seconds:
                logger.debug(
                    f"[COOLDOWN GLOBAL] Alerta omitida de forma preventiva. Restan "
                    f"{CameraService._global_cooldown_seconds - time_since_last_global_alert:.1f}s "
                    f"para liberar el uso de la API."
                )
                return False

            # Validación de cooldown individual por track_id para evitar alertas duplicadas del mismo sujeto
            if track_id is not None:
                last_track_time = self._alert_cooldown.get(track_id, 0.0)
                if current_time - last_track_time < self._alert_cooldown_seconds:
                    return False

            # Actualización de los timestamps de cooldown global e individual
            CameraService._last_global_alert_time = current_time
            if track_id is not None:
                self._alert_cooldown[track_id] = current_time

            return True
    
    def _extract_focused_roi(self, frame: np.ndarray, bbox: List[int], padding_percentage: float = 0.45) -> np.ndarray:
        """
        Extrae la Región de Interés (ROI) aplicando un margen de contexto alrededor del 
        bounding box detectado para optimizar el análisis cognitivo de la IA.

        Args:
            frame (np.ndarray): Matriz completa de la imagen en memoria.
            bbox (List[int]): Coordenadas límites de la detección [x1, y1, x2, y2].
            padding_percentage (float): Porcentaje de margen extra de contexto.

        Returns:
            np.ndarray: Submatriz recortada de la zona crítica.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = int(box_w * padding_percentage)
        pad_h = int(box_h * padding_percentage)

        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(w, x2 + pad_w)
        y2_padded = min(h, y2 + pad_h)

        return frame[y1_padded:y2_padded, x1_padded:x2_padded]

    def _save_snapshot_to_disk(self, frame_roi: np.ndarray, track_id: int) -> str:
        """
        Escribe de forma física el recorte del incidente en la carpeta estática del servidor.

        Args:
            frame_roi (np.ndarray): Matriz de píxeles recortada.
            track_id (int): Identificador de la persona en riesgo.

        Returns:
            str: El nombre único del archivo guardado en el disco (.jpg).

        Raises:
            IOError: Si ocurre un fallo en el sistema de archivos al escribir la imagen.
        """
        timestamp = int(time.time())
        filename = f"incident_cam_{self.camera_id}_track_{track_id}_{timestamp}.jpg"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        target_dir = os.path.join(project_root, "static", "snapshots")
        
        os.makedirs(target_dir, exist_ok=True)
        full_path = os.path.join(target_dir, filename)
        
        success = cv2.imwrite(full_path, frame_roi)
        if not success:
            raise IOError(f"No se pudo escribir físicamente el snapshot en: {full_path}")
            
        logger.info(f"[IO] Captura guardada con éxito en el sistema local: {filename}")
        return filename

    def _async_vision_and_persistence_worker(self, frame_roi: np.ndarray, track_id: int, alert_level: str, zone: str):
        """
        Trabador asíncrono en segundo plano encargado del almacenamiento persistente,
        invocación del VisionService saneado y la distribución multiplataforma del incidente.
        """
        logger.info(f"Procesando incidente en background para track {track_id}")
        
        try:
            filename = self._save_snapshot_to_disk(frame_roi, track_id)
            
            base_url = getattr(settings, "API_BASE_URL", "http://localhost:8000")
            public_image_url = f"{base_url}/static/snapshots/{filename}"

            # Consumo de IA (Retorna texto estrictamente limpio de tags <think>)
            ia_description = vision_service.analyze_incident_zone(frame_roi, alert_level)

            with Session(engine) as session:
                incident = create_incident(
                    db=session,
                    camera_id=self.camera_id,
                    alert_level=alert_level,
                    description=ia_description,
                    image_url=public_image_url  
                )
                
                statement = select(User).where(User.expo_push_token != None)
                active_guards = session.exec(statement).all()

                if not active_guards:
                    logger.warning(
                        "No se encontraron guardias con tokens push registrados en la DB. Alerta no enviada por Push."
                    )
                else:
                    push_title = f"ALERTA CRÍTICA - Cámara #{self.camera_id}"
                    metadata_app = {
                        "incident_id": incident.id,
                        "camera_id": self.camera_id,
                        "level": alert_level,
                        "image_url": public_image_url
                    }

                    for guard in active_guards:
                        logger.info(f"Despachando push a guardia: {guard.username}")
                        notification_service.send_push_notification(
                            expo_token=guard.expo_push_token,
                            title=push_title,
                            body=ia_description,
                            extra_data=metadata_app,
                        )
                
                ws_manager.broadcast_sync({
                    "type": "incident_enriched",
                    "incident_id": incident.id,
                    "camera_id": self.camera_id,
                    "level": alert_level,
                    "description": ia_description,
                    "image_url": public_image_url
                })
                logger.info("[WS] Alerta enriquecida distribuida a los clientes conectados.")
                
        except Exception as e:
            logger.error(f"Error crítico en el flujo de fondo del incidente: {e}")

    def _process_and_dispatch_snapshot(
        self, 
        frame: np.ndarray, 
        bbox: list[int], 
        track_id: int, 
        alert_level: str, 
        zone: str
    ):
        """
        Extrae la Región de Interés (ROI) de forma síncrona y delega el procesamiento
        pesado de IA e I/O al pool de hilos en segundo plano.
        """
        try:
            frame_roi = self._extract_focused_roi(frame, bbox)
            
            self._background_executor.submit(
                self._async_vision_and_persistence_worker,
                frame_roi,
                track_id,
                alert_level,
                zone
            )
        except Exception as error:
            logger.error(f"Fallo crítico al empaquetar o despachar el snapshot para el track {track_id}: {error}")

    def generate_frames(self) -> Generator[bytes, None, None]:
        """
        Generador continuo de frames procesados para streaming visual MJPEG de baja latencia.
        """
        source = self._resolve_video_source()
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Cámara #{self.camera_id} - No se pudo abrir la fuente de video: {source}")
            return

        if source != 0 and "youtube" not in str(self.stream_url):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        debug_pose = settings.DEBUG_POSE
        consecutive_failures = 0
        max_consecutive_failures = 10  

        while True:
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Cámara #{self.camera_id} - Cierre preventivo por fallos continuos de red.")
                break

            try:
                success, frame = cap.read()
                if not success:
                    if source and isinstance(source, str) and source.endswith(".mp4"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures += 1
                        continue
                    
                    logger.warning(f"Cámara #{self.camera_id} - Error al leer frame. Reintentando...")
                    consecutive_failures += 1
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error crítico leyendo frame en el core de OpenCV: {e}")
                break

            consecutive_failures = 0
            h, w = frame.shape[:2]

            abs_yellow = [(int(p[0] * w), int(p[1] * h)) for p in self.yellow_points] if self.yellow_points else []
            abs_red = [(int(p[0] * w), int(p[1] * h)) for p in self.red_points] if self.red_points else []

            overlay = frame.copy()
            if abs_yellow:
                cv2.fillPoly(overlay, [np.array(abs_yellow, np.int32)], (0, 215, 255))
            if abs_red:
                cv2.fillPoly(overlay, [np.array(abs_red, np.int32)], (0, 0, 255))
                
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            if abs_yellow:
                cv2.polylines(frame, [np.array(abs_yellow, np.int32)], True, (0, 215, 255), 2)
            if abs_red:
                cv2.polylines(frame, [np.array(abs_red, np.int32)], True, (0, 0, 255), 2)

            yellow_polygon = np.array(abs_yellow, np.int32) if abs_yellow else None
            red_polygon = np.array(abs_red, np.int32) if abs_red else None

            detections = self.detector.detect(frame)
            total_personas = 0
            personas_riesgo = 0
            personas_peligro = 0

            for det in detections:
                total_personas += 1
                x1, y1, x2, y2 = det.box_xyxy
                feet = ((x1 + x2) // 2, y2)

                pose_result = self.pose_analyzer.analyze(det.keypoints_xy)
                assessment = self.risk_evaluator.evaluate(
                    track_id=det.track_id,
                    feet=feet,
                    pose_result=pose_result,
                    yellow_polygon=yellow_polygon,
                    red_polygon=red_polygon,
                )

                color_map = {
                    RiskLevel.SAFE: ((0, 255, 0), "SEGURO"),
                    RiskLevel.CAUTION: ((0, 215, 255), "PRECAUCION"),
                    RiskLevel.HIGH_RISK: ((0, 0, 255), "ALTO RIESGO"),
                    RiskLevel.DANGER: ((0, 0, 255), "PELIGRO"),
                }
                color, label = color_map.get(assessment.level, ((0, 255, 0), "SEGURO"))

                if assessment.level == RiskLevel.DANGER:
                    personas_peligro += 1
                elif assessment.level in (RiskLevel.CAUTION, RiskLevel.HIGH_RISK):
                    personas_riesgo += 1

                if assessment.level in (RiskLevel.DANGER, RiskLevel.HIGH_RISK):
                    if self._should_emit_alert(det.track_id):
                        alert_level = "red" if assessment.level == RiskLevel.DANGER else "orange"
                        
                        ws_manager.broadcast_sync({
                            "type": "alert",
                            "camera_id": self.camera_id,
                            "level": alert_level,
                            "track_id": det.track_id,
                            "zone": assessment.zone.value,
                            "time_in_zone": round(assessment.time_in_zone, 1),
                            "bad_posture": assessment.is_bad_posture,
                        })

                        self._process_and_dispatch_snapshot(
                            frame=frame.copy(),
                            bbox=det.box_xyxy,
                            track_id=det.track_id,
                            alert_level=alert_level,
                            zone=assessment.zone.value
                        )

                if debug_pose and det.keypoints_xy is not None:
                    for kp in det.keypoints_xy:
                        px, py = int(kp[0]), int(kp[1])
                        if px > 0 and py > 0:
                            cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)

                    if len(det.keypoints_xy) > 12:
                        head_x = int(det.keypoints_xy[0][0])
                        head_y_int = int(det.keypoints_xy[0][1])
                        hip_x = int((det.keypoints_xy[11][0] + det.keypoints_xy[12][0]) / 2)
                        hip_y_int = int((det.keypoints_xy[11][1] + det.keypoints_xy[12][1]) / 2)

                        if head_x > 0 and head_y_int > 0 and hip_y_int > 0:
                            color_eje = (0, 165, 255) if pose_result.is_bad_posture else (0, 255, 255)
                            cv2.line(frame, (head_x, head_y_int), (hip_x, hip_y_int), color_eje, 2)
                            if pose_result.is_bad_posture:
                                cv2.putText(
                                    frame, "POSTURA CRITICA", (head_x - 30, head_y_int - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1,
                                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_label = det.track_id if det.track_id is not None else "N/A"
                cv2.putText(
                    frame, f"ID:{id_label} {label}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

            self.current_stats["total_persons"] = total_personas
            self.current_stats["risk_persons"] = personas_riesgo
            self.current_stats["danger_persons"] = personas_peligro

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        cap.release()
        logger.info(f"Cámara #{self.camera_id} - Captura de video liberada de forma correcta.")