"""
Servicio de cámara refactorizado — Andén Seguro.

Responsabilidad única: coordinar la lectura del feed de video
y orquestar el pipeline de IA (detector → pose → riesgo).
"""

import logging
import time

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.websocket_manager import ws_manager
from app.vision.detector import PersonDetector
from app.vision.pose import PoseAnalyzer
from app.vision.risk import RiskEvaluator, RiskLevel

logger = logging.getLogger(__name__)
settings = get_settings()


class CameraService:
    """Coordina la captura de video y la ejecución del pipeline de IA."""

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

    def _resolve_video_source(self):
        """Determina la fuente de video."""
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

    def _should_emit_alert(self, track_id: int | None) -> bool:
        """Verifica si se debe emitir una alerta (cooldown anti-spam)."""
        if track_id is None:
            return True
        current = time.time()
        last = self._alert_cooldown.get(track_id, 0)
        if current - last > self._alert_cooldown_seconds:
            self._alert_cooldown[track_id] = current
            return True
        return False

    def generate_frames(self):
        """Generador de frames procesados para streaming MJPEG."""
        try:
            source = self._resolve_video_source()
            cap = cv2.VideoCapture(source)
            if source != 0 and "youtube" not in str(self.stream_url):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            logger.error(f"Fallo al iniciar captura: {e}")
            return

        debug_pose = settings.DEBUG_POSE

        while True:
            try:
                success, frame = cap.read()
                if not success:
                    # Si el stream es un archivo local y termina, reiniciarlo para hacer un loop infinito
                    if source and isinstance(source, str) and source.endswith(".mp4"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Error al leer frame. Reintentando en 1s...")
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error leyendo frame del video: {e}")
                break

            h, w = frame.shape[:2]

            # Convertir porcentajes a píxeles absolutos
            abs_yellow = (
                [(int(p[0] * w), int(p[1] * h)) for p in self.yellow_points]
                if self.yellow_points
                else []
            )
            abs_red = (
                [(int(p[0] * w), int(p[1] * h)) for p in self.red_points]
                if self.red_points
                else []
            )

            # Dibujar zonas de riesgo de forma independiente
            overlay = frame.copy()
            if abs_yellow:
                pts_y = np.array(abs_yellow, np.int32)
                cv2.fillPoly(overlay, [pts_y], (0, 215, 255))
            if abs_red:
                pts_r = np.array(abs_red, np.int32)
                cv2.fillPoly(overlay, [pts_r], (0, 0, 255))
                
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            if abs_yellow:
                cv2.polylines(frame, [np.array(abs_yellow, np.int32)], True, (0, 215, 255), 2)
            if abs_red:
                cv2.polylines(frame, [np.array(abs_red, np.int32)], True, (0, 0, 255), 2)

            # Pipeline de IA: Detección → Pose → Riesgo
            yellow_polygon = (
                np.array(abs_yellow, np.int32) if abs_yellow else None
            )
            red_polygon = (
                np.array(abs_red, np.int32) if abs_red else None
            )

            detections = self.detector.detect(frame)

            total_personas = 0
            personas_riesgo = 0
            personas_peligro = 0

            for det in detections:
                total_personas += 1
                x1, y1, x2, y2 = det.box_xyxy
                feet = ((x1 + x2) // 2, y2)

                # Análisis postural
                pose_result = self.pose_analyzer.analyze(det.keypoints_xy)

                # Evaluación de riesgo
                assessment = self.risk_evaluator.evaluate(
                    track_id=det.track_id,
                    feet=feet,
                    pose_result=pose_result,
                    yellow_polygon=yellow_polygon,
                    red_polygon=red_polygon,
                )

                # Determinar color y etiqueta para visualización
                color_map = {
                    RiskLevel.SAFE: ((0, 255, 0), "SEGURO"),
                    RiskLevel.CAUTION: ((0, 215, 255), "PRECAUCION"),
                    RiskLevel.HIGH_RISK: ((0, 0, 255), "ALTO RIESGO"),
                    RiskLevel.DANGER: ((0, 0, 255), "PELIGRO"),
                }
                color, label = color_map.get(
                    assessment.level, ((0, 255, 0), "SEGURO")
                )

                # Contabilizar
                if assessment.level == RiskLevel.DANGER:
                    personas_peligro += 1
                elif assessment.level in (
                    RiskLevel.CAUTION,
                    RiskLevel.HIGH_RISK,
                ):
                    personas_riesgo += 1

                # Emitir alerta WebSocket si es peligro
                if assessment.level in (
                    RiskLevel.DANGER,
                    RiskLevel.HIGH_RISK,
                ):
                    if self._should_emit_alert(det.track_id):
                        alert_level = (
                            "red"
                            if assessment.level == RiskLevel.DANGER
                            else "orange"
                        )
                        ws_manager.broadcast_sync(
                            {
                                "type": "alert",
                                "camera_id": self.camera_id,
                                "level": alert_level,
                                "track_id": det.track_id,
                                "zone": assessment.zone.value,
                                "time_in_zone": round(
                                    assessment.time_in_zone, 1
                                ),
                                "bad_posture": assessment.is_bad_posture,
                            }
                        )

                # Debug: Dibujar esqueleto
                if debug_pose and det.keypoints_xy is not None:
                    for kp in det.keypoints_xy:
                        px, py = int(kp[0]), int(kp[1])
                        if px > 0 and py > 0:
                            cv2.circle(
                                frame, (px, py), 3, (255, 0, 255), -1
                            )

                    if len(det.keypoints_xy) > 12:
                        head_x = int(det.keypoints_xy[0][0])
                        head_y_int = int(det.keypoints_xy[0][1])
                        hip_x = int(
                            (
                                det.keypoints_xy[11][0]
                                + det.keypoints_xy[12][0]
                            )
                            / 2
                        )
                        hip_y_int = int(
                            (
                                det.keypoints_xy[11][1]
                                + det.keypoints_xy[12][1]
                            )
                            / 2
                        )

                        if head_x > 0 and head_y_int > 0 and hip_y_int > 0:
                            color_eje = (
                                (0, 165, 255)
                                if pose_result.is_bad_posture
                                else (0, 255, 255)
                            )
                            cv2.line(
                                frame,
                                (head_x, head_y_int),
                                (hip_x, hip_y_int),
                                color_eje,
                                2,
                            )
                            if pose_result.is_bad_posture:
                                cv2.putText(
                                    frame,
                                    "POSTURA CRITICA",
                                    (head_x - 30, head_y_int - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (0, 165, 255),
                                    1,
                                )

                # Dibujar bounding box e ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_label = (
                    det.track_id if det.track_id is not None else "N/A"
                )
                cv2.putText(
                    frame,
                    f"ID:{id_label} {label}",
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Actualizar estadísticas
            self.current_stats["total_persons"] = total_personas
            self.current_stats["risk_persons"] = personas_riesgo
            self.current_stats["danger_persons"] = personas_peligro

            # Codificar frame para streaming MJPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

        cap.release()
        logger.info("Captura de video liberada.")