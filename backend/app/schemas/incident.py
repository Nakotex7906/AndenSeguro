"""
DTOs de incidentes — Andén Seguro.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class IncidentResponse(BaseModel):
    """Representación de un incidente para la API."""

    id: int
    camera_id: int
    alert_level: str
    description: str
    timestamp: datetime
    duration_seconds: Optional[float]
    status: str
    resolved_by: Optional[int]
    resolved_at: Optional[datetime]

    model_config = {"from_attributes": True}


class IncidentListResponse(BaseModel):
    """Respuesta paginada de incidentes."""

    items: list[IncidentResponse]
    total: int
    page: int
    size: int


class IncidentStatusUpdate(BaseModel):
    """Payload para actualizar el estado de un incidente."""

    status: str  # acknowledged | resolved | false_alarm


class IncidentActionCreate(BaseModel):
    """Payload para crear una acción de incidente."""

    type: str
    value: str


class IncidentActionResponse(BaseModel):
    """Representación de una acción para la API."""

    id: int
    incident_id: int
    actor_id: int
    action_type: str
    action_value: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class IncidentNoteCreate(BaseModel):
    """Payload para crear una nota de incidente."""

    text: str


class IncidentNoteResponse(BaseModel):
    """Representación de una nota para la API."""

    id: int
    incident_id: int
    author_id: int
    text: str
    timestamp: datetime

    model_config = {"from_attributes": True}
