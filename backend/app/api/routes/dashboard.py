"""
Rutas del panel de control (Dashboard) — Andén Seguro.

Provee métricas agregadas para alimentar el hook useDashboardOverview.ts
del frontend de React.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session, func, select

from app.db.session import get_db
from app.models.incident import Incident
from app.models.station import Camera, Station
from app.schemas.dashboard import (
    DashboardMetric,
    DashboardOverview,
    LineStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Timestamp de arranque del servidor para calcular uptime
_server_start_time = time.time()


@router.get("/overview", response_model=DashboardOverview)
def get_dashboard_overview(
    db: Annotated[Session, Depends(get_db)],
):
    """Retorna el resumen global del sistema para el panel de control."""
    from datetime import datetime, timezone

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Conteo de estaciones activas
    active_stations = db.exec(
        select(func.count(Station.id)).where(Station.is_active == True)  # noqa: E712
    ).one()

    # Alertas pendientes hoy
    pending_alerts = db.exec(
        select(func.count(Incident.id)).where(
            Incident.status == "pending",
            Incident.timestamp >= today_start,
        )
    ).one()

    # Total incidencias del día
    total_today = db.exec(
        select(func.count(Incident.id)).where(
            Incident.timestamp >= today_start
        )
    ).one()

    # Distribución por nivel
    orange_count = db.exec(
        select(func.count(Incident.id)).where(
            Incident.alert_level == "orange",
            Incident.timestamp >= today_start,
        )
    ).one()

    red_count = db.exec(
        select(func.count(Incident.id)).where(
            Incident.alert_level == "red",
            Incident.timestamp >= today_start,
        )
    ).one()

    # Uptime
    uptime = int(time.time() - _server_start_time)

    # Métricas
    metrics = [
        DashboardMetric(
            id="active_stations",
            label="Estaciones activas",
            value=str(active_stations),
            caption="Estaciones con cámaras operativas",
            tone="blue",
        ),
        DashboardMetric(
            id="pending_alerts",
            label="Alertas pendientes",
            value=str(pending_alerts),
            caption="Requieren atención inmediata",
            tone="red" if pending_alerts > 0 else "slate",
        ),
        DashboardMetric(
            id="total_incidents_today",
            label="Incidencias hoy",
            value=str(total_today),
            caption=f"{orange_count} naranja · {red_count} rojas",
            tone="amber" if total_today > 0 else "emerald",
        ),
        DashboardMetric(
            id="system_uptime",
            label="Tiempo operativo",
            value=f"{uptime // 3600}h {(uptime % 3600) // 60}m",
            caption="Desde último reinicio del servidor",
            tone="emerald",
        ),
    ]

    # Estado por estación
    stations = db.exec(select(Station)).all()
    line_statuses: list[LineStatus] = []

    for station in stations:
        camera_ids = db.exec(
            select(Camera.id).where(Camera.station_id == station.id)
        ).all()

        station_pending = 0
        if camera_ids:
            station_pending = db.exec(
                select(func.count(Incident.id)).where(
                    Incident.status == "pending",
                    Incident.camera_id.in_(camera_ids),
                )
            ).one()

        tone = (
            "red"
            if station_pending > 0
            else "emerald" if station.is_active else "slate"
        )
        detail = (
            f"{station_pending} alertas pendientes"
            if station_pending > 0
            else "Operativa"
        )

        line_statuses.append(
            LineStatus(
                id=str(station.id),
                code=station.code,
                name=station.name,
                detail=detail,
                tone=tone,
            )
        )

    return DashboardOverview(
        title="Panel de Control Global",
        subtitle="Andén Seguro — Monitoreo en tiempo real",
        uptimeSeconds=uptime,
        metrics=metrics,
        mapTitle="Red de Estaciones",
        mapSubtitle="Estado operativo de la red de monitoreo",
        lineStatuses=line_statuses,
    )
