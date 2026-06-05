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
    IncidentCreate,
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusUpdate,
    IncidentActionCreate,
    IncidentActionResponse,
    IncidentNoteCreate,
    IncidentNoteResponse,
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


@router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
def create_manual_incident(
    payload: IncidentCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Crea un incidente manual provocado por el operador."""
    from app.services.incident_service import create_incident

    incident = create_incident(
        db=db,
        camera_id=payload.camera_id,
        alert_level="red",
        description="Alerta de emergencia activada manualmente por operador.",
    )
    return incident


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


@router.get("/{incident_id}/protocol-state")
def get_incident_protocol_state(
    incident_id: int,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    """Devuelve el estado completo del protocolo para un incidente (acciones y notas)."""
    from app.models.incident_actions import IncidentAction, IncidentNote

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    actions = db.exec(
        select(IncidentAction).where(IncidentAction.incident_id == incident_id)
    ).all()
    
    notes = db.exec(
        select(IncidentNote).where(IncidentNote.incident_id == incident_id).order_by(IncidentNote.timestamp.asc())
    ).all()

    return {
        "incident": IncidentResponse.model_validate(incident).model_dump(),
        "actions": [a.model_dump() for a in actions],
        "notes": [n.model_dump() for n in notes]
    }


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


@router.post("/{incident_id}/actions", response_model=IncidentActionResponse)
def add_incident_action(
    incident_id: int,
    body: IncidentActionCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Registra una acción estructurada tomada por el operador en el protocolo.
    """
    from app.models.incident_actions import IncidentAction

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    action = IncidentAction(
        incident_id=incident_id,
        actor_id=current_user.id,
        action_type=body.type,
        action_value=body.value,
    )
    db.add(action)
    db.commit()
    db.refresh(action)

    logger.info(f"Acción '{body.type}' registrada en incidente #{incident_id} por {current_user.username}")
    return action


@router.post("/{incident_id}/notes", response_model=IncidentNoteResponse)
def add_incident_note(
    incident_id: int,
    body: IncidentNoteCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Registra una nota de texto introducida por el operador.
    """
    from app.models.incident_actions import IncidentNote

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    note = IncidentNote(
        incident_id=incident_id,
        author_id=current_user.id,
        text=body.text,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    logger.info(f"Nota agregada en incidente #{incident_id} por {current_user.username}")
    return note


@router.post("/{incident_id}/derivation-sheet")
def generate_derivation_sheet(
    incident_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Genera la ficha de derivación. Por ahora retorna un success simulando
    la generación de un PDF, según lo acordado en el plan.
    """
    from app.models.incident_actions import IncidentAction

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    # Registrar la acción de generación
    action = IncidentAction(
        incident_id=incident_id,
        actor_id=current_user.id,
        action_type="derivation_sheet_generated",
        action_value="true",
    )
    db.add(action)
    db.commit()

    logger.info(f"Ficha de derivación generada para incidente #{incident_id} por {current_user.username}")
    
    return {
        "status": "success",
        "message": "Ficha de derivación generada exitosamente",
        "url": f"/api/incidents/{incident_id}/derivation-sheet/download" # Mock URL
    }


@router.post("/{incident_id}/rejection")
def register_rejection(
    incident_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Registra el rechazo de atención modelado como una IncidentAction.
    """
    from app.models.incident_actions import IncidentAction

    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incidente no encontrado",
        )

    # Registrar la acción de rechazo
    action = IncidentAction(
        incident_id=incident_id,
        actor_id=current_user.id,
        action_type="rejection",
        action_value="true",
    )
    
    # Podríamos actualizar el estado del incidente a resuelto aquí
    incident.status = "resolved"
    incident.resolved_by = current_user.id
    incident.resolved_at = datetime.now(timezone.utc)
    
    db.add(action)
    db.add(incident)
    db.commit()

    logger.info(f"Rechazo de atención registrado para incidente #{incident_id} por {current_user.username}")
    
    return {
        "status": "success",
        "message": "Rechazo de atención registrado correctamente."
    }
