"""
Servicio de gestión de estaciones y cámaras — Andén Seguro.

CRUD para estaciones, cámaras y sus polígonos de zona.
"""

import json
import logging
from typing import Optional

from sqlmodel import Session, select

from app.models.station import Camera, Station

logger = logging.getLogger(__name__)


def get_stations(db: Session) -> list[Station]:
    """Retorna todas las estaciones."""
    return list(db.exec(select(Station)).all())


def get_cameras_by_station(db: Session, station_id: int) -> list[Camera]:
    """Retorna las cámaras de una estación."""
    return list(
        db.exec(
            select(Camera).where(Camera.station_id == station_id)
        ).all()
    )


def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
    """Retorna una cámara por su ID."""
    return db.get(Camera, camera_id)


def update_camera_zones(
    db: Session,
    camera_id: int,
    yellow_points: list,
    red_points: list,
) -> Optional[Camera]:
    """Actualiza los polígonos de zona de una cámara en la base de datos."""
    camera = db.get(Camera, camera_id)
    if not camera:
        return None
    camera.yellow_zone = json.dumps(yellow_points)
    camera.red_zone = json.dumps(red_points)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    logger.info(f"Zonas actualizadas para cámara #{camera_id}")
    return camera


def get_camera_zones(db: Session, camera_id: int) -> dict:
    """Retorna los polígonos de zona de una cámara desde la DB."""
    camera = db.get(Camera, camera_id)
    if not camera:
        return {"yellow_points": [], "red_points": []}
    return {
        "yellow_points": json.loads(camera.yellow_zone) if camera.yellow_zone else [],
        "red_points": json.loads(camera.red_zone) if camera.red_zone else [],
    }
