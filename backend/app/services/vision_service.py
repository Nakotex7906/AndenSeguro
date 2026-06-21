"""
Servicio de análisis de visión artificial — Andén Seguro.

Centraliza las llamadas multimodales hacia la API de Groq para
interpretar las imágenes de incidentes capturadas en los andenes.
"""

import base64
import logging
from typing import Optional
import cv2
import numpy as np
from groq import Groq
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VisionService:
    """Gestiona la inferencia visual utilizando modelos de lenguaje multimodales."""

    MODEL_NAME = "qwen/qwen3.6-27b"

    def __init__(self):
        """Inicializa el cliente de Groq utilizando la API Key validada por Pydantic."""
        self.api_key = getattr(settings, "GROQ_API_KEY", None)
        self.client = Groq(api_key=self.api_key) if self.api_key else None

        if not self.client:
            logger.warning("VisionService inicializado en modo Fallback (sin GROQ_API_KEY).")

    def _convert_frame_to_base64(self, frame_roi: np.ndarray) -> Optional[str]:
        """
        Convierte una matriz de imagen OpenCV (BGR) a un string codificado en Base64.

        Args:
            frame_roi (np.ndarray): Recorte de la imagen en memoria.

        Returns:
            Optional[str]: String en base64 listo para transmisión remota o None si falla.
        """
        try:
            success, buffer = cv2.imencode(".jpg", frame_roi)
            if not success:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as error:
            logger.error(f"Error al codificar imagen a base64: {error}")
            return None

    def _log_rate_limit_status(self, headers) -> None:
        """
        Lee los headers x-ratelimit-* de la respuesta de Groq y loguea
        cuántos tokens/requests quedan, con alerta si el consumo es alto.

        Args:
            headers: Objeto de headers HTTP devuelto por with_raw_response.
        """
        try:
            limit_tokens = headers.get("x-ratelimit-limit-tokens")
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
            reset_tokens = headers.get("x-ratelimit-reset-tokens")

            limit_requests = headers.get("x-ratelimit-limit-requests")
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
            reset_requests = headers.get("x-ratelimit-reset-requests")

            if limit_tokens and remaining_tokens:
                limit_tokens = int(limit_tokens)
                remaining_tokens = int(remaining_tokens)
                used_tokens = limit_tokens - remaining_tokens
                pct_used = (used_tokens / limit_tokens) * 100 if limit_tokens else 0

                log_msg = (
                    f"[RateLimit][TPM] {used_tokens}/{limit_tokens} tokens usados "
                    f"({pct_used:.1f}%) | restantes: {remaining_tokens} | "
                    f"reset en: {reset_tokens}"
                )

                if pct_used >= 80:
                    logger.warning(f"TPM casi agotado. {log_msg}")
                else:
                    logger.info(log_msg)

            if limit_requests and remaining_requests:
                logger.info(
                    f"[RateLimit][RPD] restantes: {remaining_requests}/{limit_requests} "
                    f"| reset en: {reset_requests}"
                )

        except Exception as error:
            logger.debug(f"No se pudo parsear headers de rate limit: {error}")

    def analyze_incident_zone(self, frame_roi: np.ndarray, alert_level: str) -> str:
        """
        Envía el recorte del incidente a Groq para extraer un perfil físico estructurado
        del infractor y describir la acción de peligro en el andén.

        Args:
            frame_roi (np.ndarray): Matriz de la imagen recortada del sujeto en riesgo.
            alert_level (str): Nivel de criticidad de la alerta ('yellow' o 'red').

        Returns:
            str: Reporte operativo resumido de máximo 3 líneas para el guardia de seguridad.
        """
        if not self.client:
            return f"Alerta {alert_level.upper()}: Intrusión detectada en zona de vías. (Modo simulación sin IA)"

        base64_image = self._convert_frame_to_base64(frame_roi)
        if not base64_image:
            return "Alerta Crítica: Comportamiento de riesgo detectado en andén (Fallo de procesamiento visual)."

        system_instruction = (
            "Eres un analista de seguridad táctico de la red de metro. Tu único objetivo "
            "es generar reportes visuales sintéticos, ultra-directos y limpios para los guardias de turno."
        )

        user_prompt = (
            "Analiza la siguiente captura de seguridad de un andén de metro. Se ha detectado una "
            f"alerta de nivel {alert_level.upper()}. Genera un reporte operativo ultracorto "
            "(máximo 3 líneas en total) para el guardia de seguridad, indicando obligatoriamente:\n"
            "1) Género estimado.\n"
            "2) Rango de edad aparente.\n"
            "3) Estatura relativa y complexión.\n"
            "4) Color/tipo de vestimenta.\n"
            "5) Qué acción peligrosa está cometiendo exactamente.\n"
            "Sé conciso. No saludes, no uses introducciones, ve directo a los datos."
        )

        try:
            raw_response = self.client.chat.completions.with_raw_response.create(
                model=self.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_completion_tokens=150
            )

            self._log_rate_limit_status(raw_response.headers)

            response = raw_response.parse()
            ai_report = response.choices[0].message.content.strip()
            logger.info("Reporte generado exitosamente por Groq.")
            return ai_report

        except Exception as error:
            logger.error(f"Fallo en la comunicación con la API de Groq Vision: {error}")
            return f"Alerta {alert_level.upper()}: Persona en zona de peligro. Error al procesar rasgos con IA."


vision_service = VisionService()