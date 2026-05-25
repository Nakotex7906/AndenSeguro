"""
Excepciones de negocio personalizadas — Andén Seguro.

Permiten un manejo global de errores con códigos y mensajes estandarizados.
"""


class AndenSeguroException(Exception):
    """Excepción base del sistema."""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(detail)


class NotFoundException(AndenSeguroException):
    """Recurso solicitado no existe."""

    def __init__(self, detail: str = "Recurso no encontrado"):
        super().__init__(detail=detail, status_code=404, error_code="NOT_FOUND")


class UnauthorizedException(AndenSeguroException):
    """Credenciales inválidas o ausentes."""

    def __init__(self, detail: str = "No autorizado"):
        super().__init__(detail=detail, status_code=401, error_code="UNAUTHORIZED")


class ForbiddenException(AndenSeguroException):
    """El usuario no tiene permisos para esta operación."""

    def __init__(self, detail: str = "Acceso denegado"):
        super().__init__(detail=detail, status_code=403, error_code="FORBIDDEN")


class CameraConnectionError(AndenSeguroException):
    """Error de conexión con la cámara de seguridad."""

    def __init__(self, detail: str = "Error de conexión con la cámara"):
        super().__init__(
            detail=detail, status_code=503, error_code="CAMERA_CONNECTION_ERROR"
        )


class ModelLoadError(AndenSeguroException):
    """Error al cargar el modelo de inteligencia artificial."""

    def __init__(self, detail: str = "Error al cargar el modelo de IA"):
        super().__init__(
            detail=detail, status_code=503, error_code="MODEL_LOAD_ERROR"
        )
