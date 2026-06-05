"""
Configuración de la base de datos — Andén Seguro.

Crea el engine de SQLAlchemy, la fábrica de sesiones y la
dependencia de FastAPI para inyección de sesiones en endpoints.
"""

import logging

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

engine = create_engine(settings.DATABASE_URL, echo=False)


def create_db_and_tables():
    """Crea todas las tablas definidas en los modelos SQLModel."""
    # Importar modelos para que SQLModel los registre en el metadata
    from app.models.incident import Incident  # noqa: F401
    from app.models.incident_actions import IncidentAction, IncidentNote  # noqa: F401
    from app.models.station import Camera, Station  # noqa: F401
    from app.models.user import User  # noqa: F401

    SQLModel.metadata.create_all(engine)
    logger.info("Tablas de base de datos creadas/verificadas.")


def get_db():
    """Dependencia de FastAPI: genera una sesión de DB por request."""
    with Session(engine) as session:
        yield session
