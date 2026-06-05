"""
Evaluador de riesgo — Pipeline de IA.

Combina la posición geográfica (polígonos), el tiempo de merodeo
y la postura corporal para determinar el nivel de riesgo de cada persona.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from app.vision.pose import PoseResult

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Niveles de riesgo del sistema."""

    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    DANGER = "danger"


class ZoneType(str, Enum):
    """Tipo de zona en la que se encuentra la persona."""

    NONE = "none"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class RiskAssessment:
    """Resultado completo de la evaluación de riesgo."""

    level: RiskLevel
    zone: ZoneType
    time_in_zone: float
    is_bad_posture: bool


class RiskEvaluator:
    """Evalúa el nivel de riesgo de cada persona detectada."""

    def __init__(self, loitering_threshold: float = 5.0):
        self.loitering_threshold = loitering_threshold
        self.track_history: dict[int, float] = {}

    def evaluate(
        self,
        track_id: Optional[int],
        feet: tuple[int, int],
        pose_result: PoseResult,
        yellow_polygon: Optional[np.ndarray],
        red_polygon: Optional[np.ndarray],
    ) -> RiskAssessment:
        """
        Evalúa el riesgo de una persona combinando tres factores:
        1. Posición en polígono de zona (roja o amarilla).
        2. Tiempo de permanencia en la zona (merodeo).
        3. Postura corporal (cabeza gacha o inclinada).
        """
        current_time = time.time()

        # Evaluación geográfica
        in_red = False
        in_yellow = False

        if red_polygon is not None and len(red_polygon) >= 3:
            in_red = (
                cv2.pointPolygonTest(red_polygon, feet, False) >= 0
            )
        if yellow_polygon is not None and len(yellow_polygon) >= 3:
            in_yellow = (
                cv2.pointPolygonTest(yellow_polygon, feet, False) >= 0
            )

        # Factor temporal: tracking de permanencia en zona
        time_in_zone = 0.0
        if in_yellow or in_red:
            if track_id is not None:
                if track_id not in self.track_history:
                    self.track_history[track_id] = current_time
                time_in_zone = current_time - self.track_history[track_id]
        else:
            if track_id is not None and track_id in self.track_history:
                del self.track_history[track_id]

        # Evaluación combinada de riesgo
        if in_red:
            return RiskAssessment(
                level=RiskLevel.DANGER,
                zone=ZoneType.RED,
                time_in_zone=time_in_zone,
                is_bad_posture=pose_result.is_bad_posture,
            )
        elif in_yellow:
            if (
                time_in_zone > self.loitering_threshold
                or pose_result.is_bad_posture
            ):
                return RiskAssessment(
                    level=RiskLevel.HIGH_RISK,
                    zone=ZoneType.YELLOW,
                    time_in_zone=time_in_zone,
                    is_bad_posture=pose_result.is_bad_posture,
                )
            else:
                return RiskAssessment(
                    level=RiskLevel.CAUTION,
                    zone=ZoneType.YELLOW,
                    time_in_zone=time_in_zone,
                    is_bad_posture=pose_result.is_bad_posture,
                )

        return RiskAssessment(
            level=RiskLevel.SAFE,
            zone=ZoneType.NONE,
            time_in_zone=0.0,
            is_bad_posture=pose_result.is_bad_posture,
        )
