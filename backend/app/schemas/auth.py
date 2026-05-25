"""
DTOs de autenticación — Andén Seguro.
"""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    """Credenciales de acceso al sistema."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token JWT retornado tras autenticación exitosa."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Datos públicos de un usuario (sin contraseña)."""

    id: int
    username: str
    full_name: str
    role: str
    is_active: bool

    model_config = {"from_attributes": True}


class CreateUserRequest(BaseModel):
    """Datos requeridos para crear un nuevo usuario (solo admin)."""

    username: str
    full_name: str
    password: str
    role: str = "operador"
