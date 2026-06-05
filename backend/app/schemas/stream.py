"""
DTOs de streaming — Andén Seguro.
"""

from pydantic import BaseModel


class ZoneConfig(BaseModel):
    """Configuración de los polígonos de zona (porcentajes 0.0 a 1.0)."""

    yellow_points: list[tuple[float, float]]
    red_points: list[tuple[float, float]]


class CameraStats(BaseModel):
    """Estadísticas en tiempo real del análisis de la cámara."""

    total_persons: int = 0
    risk_persons: int = 0
    danger_persons: int = 0
