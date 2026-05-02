import { useMemo } from 'react'

import type { DashboardOverview } from '../types/dashboard'

/**
 * Carga de forma simulada los datos principales del panel de control.
 * @returns El estado asincrónico con datos, carga y posible error.
 */
export function useDashboardOverview(): {
  data: DashboardOverview | null
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
