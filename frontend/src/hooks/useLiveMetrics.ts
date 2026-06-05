import { useEffect, useState } from 'react'
import type { DashboardOverview } from '../types/dashboard'

export interface LiveMetrics {
  stationsActive: number
  passengersPerHour: number
  passengersDeltaPct: number
  activeAlerts: number
  alertLevel: 'normal' | 'precaución' | 'crítico'
  uptimeSeconds: number
}

const INITIAL: LiveMetrics = {
  stationsActive: 0,
  passengersPerHour: 0,
  passengersDeltaPct: 0,
  activeAlerts: 0,
  alertLevel: 'normal',
  uptimeSeconds: 0,
}

export function useLiveMetrics(): LiveMetrics {
  const [metrics, setMetrics] = useState<LiveMetrics>(INITIAL)

  useEffect(() => {
    // Conectar al WebSocket de alertas
    const ws = new WebSocket('ws://localhost:8000/api/alerts/ws')

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'dashboard_metrics') {
          const data: DashboardOverview = payload.data
          
          // Extraer métricas de la respuesta del backend
          const activeStationsMetric = data.metrics.find(m => m.id === 'active_stations')
          const pendingAlertsMetric = data.metrics.find(m => m.id === 'pending_alerts')
          
          const stationsActive = activeStationsMetric ? parseInt(activeStationsMetric.value) : 0
          const activeAlerts = pendingAlertsMetric ? parseInt(pendingAlertsMetric.value) : 0
          
          let alertLevel: 'normal' | 'precaución' | 'crítico' = 'normal'
          if (activeAlerts > 0) alertLevel = 'precaución'
          if (activeAlerts > 2) alertLevel = 'crítico'

          setMetrics({
            stationsActive,
            passengersPerHour: 250, // TODO: Implement passenger flow in backend
            passengersDeltaPct: 5,  // TODO: Implement passenger delta in backend
            activeAlerts,
            alertLevel,
            uptimeSeconds: data.uptimeSeconds,
          })
        }
      } catch (err) {
        console.error('Error parseando mensaje WS en useLiveMetrics:', err)
      }
    }

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err)
    }

    return () => {
      ws.close()
    }
  }, [])

  return metrics
}