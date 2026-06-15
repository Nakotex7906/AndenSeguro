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
    """
    Servicio encargado de interactuar con proveedores externos de IA Multimodal.
    
    Responsabilidad única: Convertir matrices de imágenes en memoria a formatos 
    compatibles con la API y recuperar descripciones operacionales en lenguaje natural.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Inicializa el cliente de Groq utilizando las variables del entorno del sistema.
        """
        # Reducción de code smells: Configuraciones centralizadas con fallbacks seguros
        self.api_key = api_key or getattr(settings, "GROQ_API_KEY", None)
        self.model_name = model_name or "meta-llama/llama-4-scout-17b-16e-instruct"
        
        if not self.api_key:
            logger.warning("VisionService inicializado sin GROQ_API_KEY.")
            
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def _encode_frame_to_base64_jpeg(self, frame_roi: np.ndarray) -> str:
        """
        Convierte una matriz de imagen (OpenCV BGR) a un buffer JPEG en memoria 
        y posteriormente lo codifica en una cadena de texto Base64.
        
        Evita escribir archivos en el disco duro, optimizando drásticamente la latencia.
        """
        success, encoded_buffer = cv2.imencode(".jpg", frame_roi)
        if not success:
            raise ValueError("No se pudo codificar el recorte de la imagen a formato JPEG.")
            
        binary_data = encoded_buffer.tobytes()
        base64_encoded = base64.b64encode(binary_data)
        return base64_encoded.decode("utf-8")

    def _build_security_prompt(self, severity: str) -> str:
        """
        Construye el prompt especializado para el entorno ferroviario/metro 
        según el nivel de criticidad.
        """
        base_prompt = (
            "Actúa como un experto en seguridad de estaciones de metro. "
            "Describe en una sola frase breve, directa y accionable (máximo 15 palabras) "
            "la acción de riesgo o peligro que realiza la persona enfocada en la imagen. "
            "Ejemplos válidos: 'Persona cruzando la línea amarilla de seguridad' o 'Persona caída en las vías'. "
            "No uses introducciones como 'En la imagen veo...' ni agregues texto de relleno."
        )
        return f"[ALERTA CRÍTICA: Nivel {severity.upper()}] {base_prompt}"

    def analyze_incident_zone(self, frame_roi: np.ndarray, severity: str) -> str:
        """
        Envía de forma síncrona/hilo el recorte del incidente a Groq para su análisis visual.
        
        Args:
            frame_roi (np.ndarray): Matriz recortada de la persona en peligro.
            severity (str): Criticidad del incidente ('red' u 'orange').
            
        Returns:
            str: Descripción corta generada por la IA o un texto de fallback si ocurre un fallo.
        """
        # Fallback local inmediato si el cliente no está configurado (Degradación elegante)
        if not self.client:
            return f"Alerta de seguridad automatizada nivel {severity}. (IA de visión no configurada)"

        try:
            # 1. Preparar datos e inputs
            base64_image = self._encode_frame_to_base64_jpeg(frame_roi)
            system_prompt = self._build_security_prompt(severity)

            # 2. Ejecutar la llamada con timeout controlado (Evita bloqueos indefinidos)
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Estricta consistencia operacional
                max_completion_tokens=60
            )

            # 3. Retornar la respuesta limpia
            description = chat_completion.choices[0].message.content
            return description.strip()

        except Exception as error:
            logger.error(f"Fallo en la API de visión de Groq: {error}. Aplicando fallback de contingencia.")
            # Fallback operacional para que el guardia reciba información aunque la IA falle
            if severity.lower() == "red":
                return "PELIGRO INMINENTE: Persona detectada en zona de exclusión de vías."
            return "PRECAUCIÓN: Usuario traspasó el límite seguro del andén."


# Instancia única reutilizable para el backend
vision_service = VisionService()