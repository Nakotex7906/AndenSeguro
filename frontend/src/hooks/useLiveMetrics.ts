import { useEffect, useRef, useState } from 'react'

/**
 * Métricas en tiempo real del panel de control.
 * Cada valor se actualiza de forma independiente simulando
 * el comportamiento que tendría con el backend conectado.
 *
 * eliminar una vez se implemente la conexión real con el backend y se reemplace por un hook que consuma eventos SSE o WebSocket.
 */
export interface LiveMetrics {
  /** Estaciones activas — número entero fijo hasta recibir cambio del backend */
  stationsActive: number
  /** Flujo de pasajeros por hora — varía ±5 cada 8 segundos simulando tráfico real */
  passengersPerHour: number
  /** Cambio porcentual respecto a ayer — calculado por el backend, aquí simulado */
  passengersDeltaPct: number
  /** Alertas activas en este momento */
  activeAlerts: number
  /** Nivel de alerta derivado del número de alertas */
  alertLevel: 'normal' | 'precaución' | 'crítico'
  /** Segundos transcurridos desde que arrancó el sistema */
  uptimeSeconds: number
}

/**
 * Valor inicial que usaría el backend al hacer GET /api/dashboard/overview
 */
const INITIAL: LiveMetrics = {
  stationsActive: 8,
  passengersPerHour: 258,
  passengersDeltaPct: 12,
  activeAlerts: 1,
  alertLevel: 'precaución',
  uptimeSeconds: 0,
}

function deriveAlertLevel(alerts: number): LiveMetrics['alertLevel'] {
  if (alerts === 0) return 'normal'
  if (alerts <= 2)  return 'precaución'
  return 'crítico'
}

export function useLiveMetrics(): LiveMetrics {
  const [metrics, setMetrics] = useState<LiveMetrics>(INITIAL)
  const uptimeRef = useRef<number>(INITIAL.uptimeSeconds)

  /* ── Uptime: incrementa cada segundo ── */
  useEffect(() => {
    const id = window.setInterval(() => {
      uptimeRef.current += 1
      setMetrics((prev) => ({ ...prev, uptimeSeconds: uptimeRef.current }))
    }, 1000)
    return () => window.clearInterval(id)
  }, [])

  /* ── Flujo de pasajeros: fluctúa ±5 cada 8 segundos ──
   */
  useEffect(() => {
    const id = window.setInterval(() => {
      setMetrics((prev) => {
        const delta = Math.floor(Math.random() * 11) - 5   // -5 a +5
        const next  = Math.max(200, prev.passengersPerHour + delta)
        return { ...prev, passengersPerHour: next }
      })
    }, 8000)
    return () => window.clearInterval(id)
  }, [])

  /* ── Alertas: puede cambiar cuando el backend emita un evento ──
   * Aquí solo se simula una variación aleatoria ocasional (cada 30s)
   * para que la UI no sea completamente estática durante desarrollo.
   */
  useEffect(() => {
    const id = window.setInterval(() => {
      setMetrics((prev) => {
        const alerts = prev.activeAlerts
        return { ...prev, alertLevel: deriveAlertLevel(alerts) }
      })
    }, 30000)
    return () => window.clearInterval(id)
  }, [])

  return metrics
}