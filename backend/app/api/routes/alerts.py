"""
Rutas de Websocket y HTTP para Alertas y Métricas — Andén Seguro v2.0.

Transmite actualizaciones del dashboard en tiempo real y gestiona la inyección
de incidentes simulados directamente en el proceso activo del servidor.
"""

import asyncio
import logging
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session

from app.core.websocket_manager import ws_manager
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def alerts_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket dedicado para recibir incidentes activos y métricas globales
    del dashboard en tiempo real (reemplaza a los simuladores del frontend).
    """
    from app.db.session import engine
    
    await ws_manager.connect(websocket)
    try:
        while True:
            # Enviar métricas globales periódicamente (DashboardOverview)
            from app.api.routes.dashboard import get_dashboard_overview
            
            # Usar una sesión fresca por cada iteración para obtener datos actualizados
            with Session(engine) as session:
                overview = get_dashboard_overview(db=session)
            
            payload = {
                "type": "dashboard_metrics",
                "data": overview.model_dump()
            }
            
            await websocket.send_json(payload)
            
            # Esperar 2 segundos antes de volver a enviar
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as error:
        logger.error(f"Error en el ciclo del websocket de alertas: {error}")
        ws_manager.disconnect(websocket)


# INYECCIÓN — ENDPOINT DE INFERENCIA PARA PRUEBAS LOCALES (Fase 4)
@router.post("/inject-mock", status_code=status.HTTP_200_OK)
def inject_mock_incident(db: Session = Depends(get_db)):
    """
    Inyecta un incidente de simulación controlado directamente dentro del ciclo de vida
    y la memoria del proceso activo del servidor FastAPI.
    
    Este puente HTTP resuelve el aislamiento de memoria del entorno local, permitiendo
    que el script de prueba despache tareas al pool de hilos del servidor y propague 
    los datos hacia el celular a través del canal de WebSocket correcto.
    
    Args:
        db (Session): Sesión activa de la base de datos relacional.
        
    Returns:
        dict: Estado de la operación y mensaje descriptivo de confirmación.
    """
    from app.services.camera_manager import camera_manager
    
    logger.info("[HTTP-TEST] Solicitud de inyección de alerta de prueba recibida.")
    
    #  Recuperar u obtener la instancia activa de CameraService para la cámara #1
    camera_service = camera_manager.get_camera(camera_id=1, db=db)
    if not camera_service:
        logger.error("[HTTP-TEST] No se pudo recuperar el servicio de la cámara 1 (Instancia nula).")
        return {"status": "error", "message": "El servicio de la cámara 1 no se encuentra disponible."}

    #  Fabricar un frame artificial en memoria (Fondo oscuro con un cuadro rojo de peligro)
    # Dimensiones estándar de video: 640x480 con 3 canales BGR
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(mock_frame, (180, 120), (420, 380), (0, 0, 255), -1)
    
    # Coordenadas límites simuladas del sujeto artificial [x1, y1, x2, y2]
    mock_bbox = [180, 120, 420, 380]

    #  Despachar el flujo asíncrono concurrente en background dentro del proceso de Uvicorn
    camera_service._process_and_dispatch_snapshot(
        frame=mock_frame,
        bbox=mock_bbox,
        track_id=88,  # Identificador de tracking de simulación
        alert_level="red",
        zone="Línea de Vías - Andén Central"
    )

    return {
        "status": "success",
        "message": "Incidente simulado inyectado en el pipeline del servidor con éxito."
    }