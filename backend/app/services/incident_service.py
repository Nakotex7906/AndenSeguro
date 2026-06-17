"""
Servicio de gestión de incidentes — Andén Seguro.

Centraliza operaciones CRUD sobre la tabla Incident.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, func, select

from app.models.incident import Incident

logger = logging.getLogger(__name__)


def create_incident(
    db: Session,
    camera_id: int,
    alert_level: str,
    description: str = "",
    image_url: Optional[str] = None,
) -> Incident:
    """
    Registra un nuevo incidente en la base de datos.

    Args:
        db: Sesión activa de la base de datos.
        camera_id: ID de la cámara que detectó el incidente.
        alert_level: Nivel de alerta (yellow | red).
        description: Descripción opcional del incidente.
        image_url: Enlace HTTP público del pantallazo almacenado en el servidor.
    """
    incident = Incident(
        camera_id=camera_id,
        alert_level=alert_level,
        description=description,
        image_url=image_url,
        timestamp=datetime.now(timezone.utc),
        status="pending",
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    logger.info(
        f"Incidente #{incident.id} creado: nivel={alert_level}, cámara={camera_id}"
    )
    return incident


def get_incidents(
    db: Session,
    camera_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
) -> list[Incident]:
    """Consulta incidentes con filtros opcionales."""
    query = select(Incident)
    if camera_id is not None:
        query = query.where(Incident.camera_id == camera_id)
    if status is not None:
        query = query.where(Incident.status == status)
    return list(
        db.exec(
            query.order_by(Incident.timestamp.desc()).offset(skip).limit(limit)
        ).all()
    )


def update_incident_status(
    db: Session,
    incident_id: int,
    new_status: str,
    user_id: Optional[int] = None,
) -> Optional[Incident]:
    """Actualiza el estado de un incidente."""
    incident = db.get(Incident, incident_id)
    if not incident:
        return None
    incident.status = new_status
    if new_status in ("resolved", "false_alarm") and user_id:
        incident.resolved_by = user_id
        incident.resolved_at = datetime.now(timezone.utc)
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def get_daily_summary(db: Session) -> dict:
    """Retorna un resumen de incidentes del día actual."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    total = db.exec(
        select(func.count(Incident.id)).where(
            Incident.timestamp >= today_start
        )
    ).one()

    pending = db.exec(
        select(func.count(Incident.id)).where(
            Incident.timestamp >= today_start,
            Incident.status == "pending",
        )
    ).one()

    orange = db.exec(
        select(func.count(Incident.id)).where(
            Incident.timestamp >= today_start,
            Incident.alert_level == "orange",
        )
    ).one()

    red = db.exec(
        select(func.count(Incident.id)).where(
            Incident.timestamp >= today_start,
            Incident.alert_level == "red",
        )
    ).one()

    return {"total": total, "pending": pending, "orange": orange, "red": red}
