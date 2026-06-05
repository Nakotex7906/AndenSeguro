import { useState, useEffect } from 'react'

import type { IncidentAlert } from '../types/dashboard'

/**
 * Obtiene la alerta activa mostrada en la vista de cámaras desde el WebSocket.
 * Escucha eventos de tipo "alert" emitidos por el motor de IA.
 */
export function useIncidentAlerts(): {
  data: IncidentAlert | null
  error: string | null
  isLoading: boolean
} {
  const [data, setData] = useState<IncidentAlert | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/api/alerts/ws')

    ws.onopen = () => {
      setIsLoading(false)
      setError(null)
    }

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'alert') {
          // payload.level: 'red' | 'orange'
          // payload.track_id: number
          // payload.zone: 'red' | 'yellow'
          // payload.time_in_zone: number
          // payload.bad_posture: boolean
          
          setData(prevData => {
            const trackIdsStr = prevData?.trackIds ? 
              (prevData.trackIds.includes(payload.track_id) ? prevData.trackIds : [...prevData.trackIds, payload.track_id]) : 
              [payload.track_id]

            const description = payload.zone === 'red' 
              ? `Invasión de área roja detectada (track_ids: ${trackIdsStr.join(', ')}). ${payload.bad_posture ? 'Postura crítica identificada.' : ''}`
              : `Merodeo prolongado en línea amarilla (track_ids: ${trackIdsStr.join(', ')}). Tiempo: ${payload.time_in_zone}s.`

            return {
              title: payload.level === 'red' ? 'PELIGRO INMINENTE' : 'ADVERTENCIA',
              cameraLabel: 'CAM-01-ANDEN PRINCIPAL',
              elapsedSeconds: Math.floor(payload.time_in_zone),
              description: description,
              location: 'Estación Central - Andén 1',
              primaryActionLabel: 'Activar Protocolo',
              secondaryActionLabel: 'Falsa Alarma',
              supportResources: [
                { id: 'sec-1', name: 'Personal de seguridad (Nivel 1)', eta: '2 min', tone: 'blue' }
              ],
              trackIds: trackIdsStr
            }
          })
        }
      } catch (err) {
        console.error('Error parseando alerta WS', err)
      }
    }

    ws.onerror = () => {
      setError('Error de conexión a Alertas WS')
      setIsLoading(false)
    }

    ws.onclose = () => {
      setIsLoading(false)
    }

    // Auto-limpiar alerta después de 15 segundos sin novedades
    const cleanupInterval = setInterval(() => {
      setData(null)
    }, 15000)

    return () => {
      ws.close()
      clearInterval(cleanupInterval)
    }
  }, [])

  return { data, error, isLoading }
}
