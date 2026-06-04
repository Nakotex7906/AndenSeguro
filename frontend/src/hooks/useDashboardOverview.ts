import { useLiveMetrics } from './useLiveMetrics'
import type { DashboardOverview } from '../types/dashboard'

/**
 * Construye los datos del panel de control combinando métricas vivas
 * con configuración estática (títulos, líneas de metro).
 *
 * Orden de líneas:
 *   1. Roja  2. Amarilla  3. Café  4. Azul
 *   5. Verde  6. Morada  7. Gris  8. Naranja  9. Rosada
 */
export function useDashboardOverview(): {
  data: DashboardOverview | null
  error: string | null
  isLoading: boolean
} {
  const m = useLiveMetrics()

  /* Formatea segundos en "HH:MM:SS" o "MM:SS" */
  function formatUptime(s: number): string {
    const h   = Math.floor(s / 3600)
    const min = Math.floor((s % 3600) / 60)
    const sec = s % 60
    if (h > 0)
      return `${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  type AlertLevel = 'normal' | 'precaución' | 'crítico'
  const alertToneMap: Record<AlertLevel, 'emerald' | 'amber' | 'red'> = {
    normal:     'emerald',
    'precaución': 'amber',
    'crítico':   'red',
  }

  const data: DashboardOverview = {
    title: 'Panel de Control Global',
    subtitle: 'Estado del sistema en tiempo real:',
    systemStatus:
      m.activeAlerts === 0
        ? 'OPERATIVO'
        : m.activeAlerts <= 2
        ? 'PRECAUCIÓN'
        : 'FUERA DE SERVICIO',
    uptimeSeconds: m.uptimeSeconds,
    metrics: [
      {
        id: 'stations',
        label: 'Estaciones activas',
        value: String(m.stationsActive).padStart(2, '0'),
        caption: '100% disponible',
        tone: 'blue',
      },
      {
        id: 'passengers',
        label: 'Flujo de pasajeros',
        value: String(m.passengersPerHour),
        unit: '/hr',
        caption: `${m.passengersDeltaPct >= 0 ? '+' : ''}${m.passengersDeltaPct}% vs ayer`,
        tone: 'slate',
      },
      {
        id: 'alerts',
        label: 'Alertas activas',
        value: String(m.activeAlerts).padStart(2, '0'),
        caption: `Nivel: ${m.alertLevel}`,
        tone: alertToneMap[m.alertLevel],
      },
      {
        id: 'uptime',
        label: 'Tiempo de actividad',
        value: formatUptime(m.uptimeSeconds),
        caption: 'Nivel: normal',
        tone: 'emerald',
      },
    ],
    mapTitle: 'Mapa de líneas interactivo',
    mapSubtitle: 'Visualización dinámica de líneas en funcionamiento',
    lineStatuses: [
      // ── Siempre visibles (primeras 4) ──────────────────────────────
      { id: 'l1', code: 'L1', name: 'Línea Roja',     detail: 'DEMORA: +12 MIN',          tone: 'red'     },
      { id: 'l2', code: 'L2', name: 'Línea Amarilla', detail: 'Mantenimiento programado…', tone: 'amber'   },
      { id: 'l3', code: 'L3', name: 'Línea Café',     detail: 'Frecuencia: 4 min',         tone: 'orange'  },
      { id: 'l4', code: 'L4', name: 'Línea Azul',     detail: 'Frecuencia: 3.5 min',       tone: 'blue'    },
      // ── Se despliegan con "Ver detalles" (últimas 5) ───────────────
      { id: 'l5', code: 'L5', name: 'Línea Verde',    detail: 'Frecuencia: 5 min',         tone: 'emerald' },
      { id: 'l6', code: 'L6', name: 'Línea Morada',   detail: 'Frecuencia: 6 min',         tone: 'purple'  },
      { id: 'l7', code: 'L7', name: 'Línea Gris',     detail: 'Servicio reducido',         tone: 'gray'    },
      { id: 'l8', code: 'L8', name: 'Línea Naranja',  detail: 'Frecuencia: 7 min',         tone: 'orange'  },
      { id: 'l9', code: 'L9', name: 'Línea Rosada',   detail: 'Frecuencia: 8 min',         tone: 'pink'    },
    ],
  }

  return { data, error: null, isLoading: false }
}