"""
Modelos de acciones y notas de un incidente — Andén Seguro.

Registra el historial de interacciones de un operador con un incidente activo.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class IncidentNote(SQLModel, table=True):
    """Notas de texto introducidas por el operador durante un incidente."""

    __tablename__ = "incident_notes"

    id: Optional[int] = Field(default=None, primary_key=True)
    incident_id: int = Field(foreign_key="incidents.id", index=True)
    author_id: int = Field(foreign_key="users.id")
    text: str = Field(description="Contenido de la nota")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IncidentAction(SQLModel, table=True):
    """Acciones estructuradas tomadas por el operador."""

    __tablename__ = "incident_actions"

    id: Optional[int] = Field(default=None, primary_key=True)
    incident_id: int = Field(foreign_key="incidents.id", index=True)
    actor_id: int = Field(foreign_key="users.id")
    action_type: str = Field(
        description="Tipo de acción: step_toggle, risk_level, signal_toggle, derivation, rejection, channel_call"
    )
    action_value: str = Field(
        description="Valor de la acción: ID del paso, nivel de riesgo, etc."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
