"""
Punto de entrada de la aplicación FastAPI — Andén Seguro v2.0.

Configura middleware, manejo global de excepciones, logging,
inicialización de la base de datos y registro de todas las rutas.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import AndenSeguroException
from app.db.session import create_db_and_tables

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Andén Seguro API",
    description="Sistema de monitoreo de estaciones mediante visión artificial",
    version="2.0.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Manejo Global de Excepciones ---
@app.exception_handler(AndenSeguroException)
async def anden_exception_handler(
    _request: Request, exc: AndenSeguroException
):
    """Captura excepciones de negocio y las transforma en JSON limpio."""
    logger.warning(f"Excepción de negocio: [{exc.error_code}] {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception):
    """Captura cualquier excepción no manejada para evitar caídas del servidor."""
    logger.exception(f"Error interno no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "detail": "Error interno del servidor. Contacte al administrador.",
        },
    )


# --- Eventos de Startup ---
@app.on_event("startup")
def on_startup():
    """Inicializa la base de datos y crea datos semilla si no existen."""
    logger.info("Iniciando Andén Seguro API v2.0.0...")
    create_db_and_tables()
    _seed_superuser()
    _seed_demo_stations()
    logger.info("Base de datos inicializada correctamente.")


def _seed_superuser():
    """Crea el superusuario admin si no existe en la base de datos."""
    from sqlmodel import Session, select

    from app.core.security import get_password_hash
    from app.db.session import engine
    from app.models.user import User

    with Session(engine) as db:
        existing = db.exec(
            select(User).where(User.username == "admin")
        ).first()
        if not existing:
            admin = User(
                username="admin",
                full_name="Administrador del Sistema",
                hashed_password=get_password_hash("admin123"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            logger.info(
                "Superusuario 'admin' creado con password por defecto. "
                "¡Cambiar en producción!"
            )


def _seed_demo_stations():
    """Crea estaciones y cámaras de demostración si la tabla está vacía."""
    from sqlmodel import Session, select

    from app.db.session import engine
    from app.models.station import Camera, Station

    with Session(engine) as db:
        existing = db.exec(select(Station)).first()
        if not existing:
            for i in range(1, 4):
                station = Station(
                    name=f"Estación {i}",
                    code=f"EST{i}",
                    line=f"Línea {((i - 1) % 3) + 1}",
                )
                db.add(station)
                db.commit()
                db.refresh(station)

                camera = Camera(
                    station_id=station.id,
                    label=f"Cámara Andén {i}",
                    serial="",
                )
                db.add(camera)
            db.commit()
            logger.info(
                "Estaciones y cámaras de demostración creadas (Estación 1-3)"
            )


# --- Registro de Rutas ---
from app.api.routes import auth, dashboard, incidents, stream  # noqa: E402

app.include_router(
    stream.router, prefix="/api/stream", tags=["Streaming"]
)
app.include_router(
    auth.router, prefix="/api/auth", tags=["Autenticación"]
)
app.include_router(
    dashboard.router, prefix="/api/dashboard", tags=["Dashboard"]
)
app.include_router(
    incidents.router, prefix="/api/incidents", tags=["Incidentes"]
)


@app.get("/")
def health_check():
    """Endpoint de verificación de salud del servidor."""
    return {
        "status": "Operativo",
        "project": "Andén Seguro",
        "version": "2.0.0",
    }