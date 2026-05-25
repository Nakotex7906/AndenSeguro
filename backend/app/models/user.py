"""
Modelo de usuario — Andén Seguro.

Gestiona la identidad del personal del metro con roles diferenciados.
Roles válidos: admin, jefe_estacion, seguridad, operador.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

class User(SQLModel, table=True):
    """Tabla de usuarios del sistema."""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True, max_length=50)
    full_name: str = Field(max_length=150)
    hashed_password: str
    role: str = Field(
        default="operador",
        description="Rol: admin | jefe_estacion | seguridad | operador",
    )
    is_active: bool = Field(default=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
