"""
Modelos de estación y cámara — Andén Seguro.

Permiten mapear múltiples cámaras a estaciones de metro específicas,
almacenando los polígonos de riesgo como JSON serializado.
"""

from typing import Optional
from sqlmodel import Field, SQLModel


class Station(SQLModel, table=True):
    """Tabla de estaciones de metro."""

    __tablename__ = "stations"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100)
    code: str = Field(unique=True, index=True, max_length=10)
    line: str = Field(default="Línea 1", max_length=50)
    is_active: bool = Field(default=True)


class Camera(SQLModel, table=True):
    """
    Tabla de cámaras de seguridad, vinculadas a una estación.
    Mapea la configuración analítica requerida por el pipeline de IA.
    """

    __tablename__ = "cameras"

    id: Optional[int] = Field(default=None, primary_key=True)
    station_id: int = Field(foreign_key="stations.id")
    label: str = Field(max_length=100)
    serial: str = Field(default="", max_length=50)
    
    # NUEVA COLUMNA: Almacena la ruta del video local o la URL RTSP del streaming físico
    stream_url: str = Field(
        default="", 
        max_length=255, 
        description="Ruta del archivo MP4 local o dirección del feed de video en vivo"
    )
    
    yellow_zone: str = Field(
        default="[]",
        description="Polígono amarillo serializado como JSON",
    )
    red_zone: str = Field(
        default="[]",
        description="Polígono rojo serializado como JSON",
    )
    is_active: bool = Field(default=True)