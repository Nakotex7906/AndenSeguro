"""
Configuración centralizada del backend — Andén Seguro.

Todas las variables de entorno se validan y tipan aquí mediante Pydantic Settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración global cargada automáticamente desde variables de entorno y .env."""

    # --- Base de Datos ---
    DATABASE_URL: str = "postgresql://anden_user:anden_pass@localhost:5432/anden_seguro_db"

    # --- Seguridad / JWT ---
    SECRET_KEY: str = "change-me-in-production-please"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- YOLO / IA ---
    YOLO_MODEL: str = "yolov8n-pose.pt"
    YOLO_IMGSZ: int = 1280
    YOLO_CONF_THRESHOLD: float = 0.25

    GROQ_API_KEY: str 
    
    # URL base del servidor utilizada para construir los enlaces de los snapshots 
    # que se envían por red local hacia la aplicación móvil.
    API_BASE_URL: str = "http://localhost:8000"

    # --- Cámara Ezviz ---
    APP_KEY: str = ""
    APP_SECRET: str = ""
    SERIAL: str = ""
    BASE_URL: str = "https://isaopen.ezvizlife.com"

    # --- Debug / Multicámara ---
    USE_MOCK_CAMERA: bool = False
    DEBUG_POSE: bool = False
    CAMERA_1_URL: str = ""
    CAMERA_2_URL: str = ""
    CAMERA_3_URL: str = ""

    # --- Riesgo ---
    LOITERING_THRESHOLD_SECONDS: float = 5.0

    # --- CORS ---
    FRONTEND_URL: str = "http://localhost:5173"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton cacheado en memoria para la configuración.
    Garantiza que el archivo .env se lea una única vez reduciendo I/O redundante.
    """
    return Settings()