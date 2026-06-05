"""
Rutas de gestión de incidentes — Andén Seguro.

CRUD paginado de incidentes y cambio de estado (acknowledge, resolve, false_alarm).
Alimenta el hook useIncidentAlerts.ts del frontend.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, func, select

from app.api.deps import get_current_user, require_role
from app.db.session import get_db
from app.models.incident import Incident
from app.models.user import User
from app.schemas.incident import (
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=IncidentListResponse)
def get_incidents(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
    camera_id: Optional[int] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    level: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    """Listado paginado de incidentes con filtros opcionales."""
    query = select(Incident)
    count_query = select(func.count(Incident.id))

    if camera_id is not None:
        query = query.where(Incident.camera_id == camera_id)
        count_query = count_query.where(Incident.camera_id == camera_id)
    if status_filter is not None:
        query = query.where(Incident.status == status_filter)
        count_query = count_query.where(Incident.status == status_filter)
    if level is not None:
        query = query.where(Incident.alert_level == level)
        count_query = count_query.where(Incident.alert_level == level)

    total = db.exec(count_query).one()

    offset = (page - 1) * size
    items = db.exec(
        query.order_by(Incident.timestamp.desc()).offset(offset).limit(size)
    ).all()

    return IncidentListResponse(
        items=[IncidentResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(
    incident_id: int,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    """Detalle de un incidente específico."""
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )
    return incident


@router.patch("/{incident_id}/status", response_model=IncidentResponse)
def update_incident_status(
    incident_id: int,
    body: IncidentStatusUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[
        User, Depends(require_role("admin", "jefe_estacion", "seguridad"))
    ],
):
    """
    Actualiza el estado de un incidente.
    Requiere rol admin, jefe_estacion o seguridad.
    """
    valid_statuses = {"acknowledged", "resolved", "false_alarm"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Estado inválido. Valores permitidos: {valid_statuses}",
        )

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    incident.status = body.status
    if body.status in ("resolved", "false_alarm"):
        incident.resolved_by = current_user.id
        incident.resolved_at = datetime.now(timezone.utc)

    db.add(incident)
    db.commit()
    db.refresh(incident)

    logger.info(
        f"Incidente #{incident_id} actualizado a '{body.status}' "
        f"por {current_user.username}"
    )
    return incident
