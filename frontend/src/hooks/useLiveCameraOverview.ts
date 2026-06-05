import { useMemo } from 'react'

import type { LiveCameraOverview } from '../types/dashboard'

/**
 * Obtiene los datos resumidos de la cámara principal.
 * @returns El estado asincrónico con feed, carga y error.
 */
export function useLiveCameraOverview(): {
  data: LiveCameraOverview | null
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
