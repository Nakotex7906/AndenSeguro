"""
Modelo de incidente — Andén Seguro.

Registra permanentemente las alertas validadas por el sistema de IA,
incluyendo nivel, duración, estado de atención y resolución.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Incident(SQLModel, table=True):
    """Tabla de incidentes de seguridad detectados por el sistema."""

    __tablename__ = "incidents"

    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: int = Field(foreign_key="cameras.id", index=True)
    alert_level: str = Field(
        description="Nivel de alerta: orange | red",
    )
    description: str = Field(default="")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
    )
    duration_seconds: Optional[float] = Field(default=None)
    status: str = Field(
        default="pending",
        index=True,
        description="Estado: pending | acknowledged | resolved | false_alarm",
    )
    resolved_by: Optional[int] = Field(
        default=None, foreign_key="users.id"
    )
    resolved_at: Optional[datetime] = Field(default=None)
