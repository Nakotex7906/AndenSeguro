import { useMemo } from 'react'

import type { IncidentAlert } from '../types/dashboard'

/**
 * Obtiene la alerta activa mostrada en la vista de cámaras.
 * @returns El detalle de la alerta, estado de carga y error.
 */
export function useIncidentAlerts(): {
  data: IncidentAlert | null
  error: string | null
  isLoading: boolean
} {
  return useMemo(
    () => ({
      data: null,
      error: null,
      isLoading: false,
    }),
    [],
  )
}
