"""
DTOs del dashboard — Andén Seguro.

Mapean exactamente a las interfaces TypeScript del frontend:
DashboardOverview, DashboardMetric, LineStatus.
"""

from pydantic import BaseModel


class DashboardMetric(BaseModel):
    """Métrica visible en las tarjetas del panel de control."""

    id: str
    label: str
    value: str
    caption: str
    tone: str  # slate | blue | amber | red | emerald


class LineStatus(BaseModel):
    """Estado operativo de una estación/línea de monitoreo."""

    id: str
    code: str
    name: str
    detail: str
    tone: str


class DashboardOverview(BaseModel):
    """Datos completos de la vista principal del dashboard."""

    title: str
    subtitle: str
    uptimeSeconds: int
    metrics: list[DashboardMetric]
    mapTitle: str
    mapSubtitle: str
    lineStatuses: list[LineStatus]
