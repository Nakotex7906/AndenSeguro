import logging
from typing import Dict, Any, List, Optional
import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ExpoNotificationService:
    """
    Servicio encargado de empaquetar y despachar notificaciones push móviles
    a través del ecosistema de servidores de Expo.
    
    Responsabilidad única: Comunicarse con la API de Expo para notificar a los
    guardias de andén sobre incidentes de seguridad críticos en tiempo real.
    """

    def __init__(self):
        """Inicializa el servicio y define el endpoint canónico de Expo."""
        self.expo_api_url = "https://exp.host/--/api/v2/push/send"

    def send_push_notification(
        self, 
        expo_token: str, 
        title: str, 
        body: str, 
        extra_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Envía una notificación push individual a un dispositivo específico.

        Args:
            expo_token (str): El token único del dispositivo del guardia (ej: 'ExponentPushToken[...]').
            title (str): Título de la notificación push.
            body (str): Mensaje o descripción corta de la alerta (generada por la IA).
            extra_data (dict, optional): Datos personalizados adjuntos (metadata del incidente, IDs, etc).

        Returns:
            bool: True si Expo aceptó la entrega del mensaje, False de lo contrario.
        """
        # Validación de seguridad básica para evitar code smells por tokens vacíos
        if not expo_token or not expo_token.startswith("ExponentPushToken"):
            logger.error(f"Formato de token de Expo inválido o ausente: '{expo_token}'")
            return False

        # Configuración del payload estructurado según el contrato de datos de Expo
        payload = {
            "to": expo_token,
            "title": title,
            "body": body,
            "sound": "default",  # Fuerza el sonido nativo de alerta en el dispositivo
            "priority": "high",  # Garantiza entrega inmediata saltándose optimizaciones de batería
            "data": extra_data or {}
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate"
        }

        try:
            # Petición HTTP síncrona con un timeout estricto para evitar bloqueos prolongados del hilo
            response = requests.post(
                self.expo_api_url, 
                json=payload, 
                headers=headers, 
                timeout=5.0
            )
            response_json = response.json()

            # Estudiar la respuesta del servidor de Expo para trazabilidad en producción
            if response.status_code == 200 and "data" in response_json:
                status = response_json["data"].get("status")
                if status == "ok":
                    logger.info(f"Notificación push aceptada por Expo para el token: {expo_token[:25]}...")
                    return True
                
                # Manejo de fallos específicos de tokens obsoletos/desinstalados
                error_message = response_json["data"].get("message", "Error desconocido")
                logger.warning(f"Expo rechazó la entrega del mensaje: {error_message}")
                return False

            logger.error(f"Error inesperado en el servidor de Expo. Código HTTP: {response.status_code}")
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Fallo de red al conectar con el servicio push de Expo: {e}")
            return False


# Instancia única global para el backend
notification_service = ExpoNotificationService()