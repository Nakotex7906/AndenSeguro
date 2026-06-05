"""
Rutas de autenticación y gestión de usuarios — Andén Seguro.

- Login con JWT.
- Consulta del perfil autenticado.
- Registro de nuevos usuarios (solo administradores).
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.api.deps import get_current_user, require_role
from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    CreateUserRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(
    credentials: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
):
    """Autenticación por credenciales. Retorna un JWT de acceso."""
    user = db.exec(
        select(User).where(User.username == credentials.username)
    ).first()

    if not user or not verify_password(
        credentials.password, user.hashed_password
    ):
        logger.warning(
            f"Intento de login fallido para usuario: {credentials.username}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta desactivada. Contacte al administrador.",
        )

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    logger.info(f"Login exitoso: {user.username} (rol: {user.role})")
    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Retorna los datos del usuario autenticado."""
    return current_user


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    user_data: CreateUserRequest,
    db: Annotated[Session, Depends(get_db)],
    _admin: Annotated[User, Depends(require_role("admin"))],
):
    """
    Registra un nuevo usuario en el sistema.
    Solo accesible para usuarios con rol 'admin'.
    """
    existing = db.exec(
        select(User).where(User.username == user_data.username)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El usuario '{user_data.username}' ya existe",
        )

    new_user = User(
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        role=user_data.role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(
        f"Usuario creado: {new_user.username} (rol: {new_user.role}) por admin"
    )
    return new_user
