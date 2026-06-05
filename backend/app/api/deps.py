"""
Dependencias compartidas de la API — Andén Seguro.

Provee inyección de sesiones DB, extracción del usuario autenticado
y verificación de roles para proteger los endpoints.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlmodel import Session, select

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Extrae y valida el usuario del token JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.exec(select(User).where(User.username == username)).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_role(*allowed_roles: str):
    """
    Dependencia parametrizada que verifica el rol del usuario.

    Uso: Depends(require_role("admin", "jefe_estacion"))
    """

    def role_checker(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role not in allowed_roles:
            logger.warning(
                f"Acceso denegado: {current_user.username} "
                f"(rol={current_user.role}) intentó acceder a recurso "
                f"restringido a {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"El rol '{current_user.role}' no tiene permisos "
                    f"para esta operación"
                ),
            )
        return current_user

    return role_checker
