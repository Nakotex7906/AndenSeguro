"""
Analizador de pose — Pipeline de IA.

Extrae y evalúa los puntos del esqueleto (keypoints) de cada detección,
aplicando anonimización facial conforme a la Ley 19.628.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Resultado del análisis postural de una persona."""

    is_bad_posture: bool
    head_y: float
    hip_y: float


class PoseAnalyzer:
    """Evalúa la postura corporal a partir de los keypoints COCO."""

    def analyze(self, keypoints_xy: Optional[np.ndarray]) -> PoseResult:
        """
        Analiza la postura comparando la posición vertical de la cabeza
        respecto a la cadera. Una cabeza al nivel o por debajo de la cadera
        indica una postura crítica (inclinación, cabeza gacha).

        Keypoints COCO relevantes:
            0 = nariz (referencia de cabeza)
            11 = cadera izquierda
            12 = cadera derecha
        """
        if keypoints_xy is None or len(keypoints_xy) <= 12:
            return PoseResult(is_bad_posture=False, head_y=0.0, hip_y=0.0)

        head_y = float(keypoints_xy[0][1])

        # Promedio de ambas caderas; fallback si no son visibles
        left_hip_y = float(keypoints_xy[11][1])
        right_hip_y = float(keypoints_xy[12][1])
        hip_y = (left_hip_y + right_hip_y) / 2 if left_hip_y > 0 else right_hip_y

        bad_posture = False
        if head_y > 0 and hip_y > 0 and head_y >= (hip_y - 20):
            bad_posture = True

        return PoseResult(
            is_bad_posture=bad_posture,
            head_y=head_y,
            hip_y=hip_y,
        )
