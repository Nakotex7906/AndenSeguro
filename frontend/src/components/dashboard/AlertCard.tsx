import type { ReactElement } from 'react'

import type { IncidentAlert } from '../../types/dashboard'

/**
 * Propiedades de la alerta crítica visible en la cámara.
 */
export interface AlertCardProps {
  alert: IncidentAlert
  elapsedTime: string
}

/**
 * Renderiza el banner de alerta con el estado crítico de la escena.
 * @param props - Información de la alerta y tiempo transcurrido.
 * @returns Un bloque superior de alta visibilidad.
 */
export function AlertCard({ alert, elapsedTime }: AlertCardProps): ReactElement {
  return (
    <section
      aria-label="Alerta activa"
      className="flex flex-col gap-4 rounded-2xl border border-red-500/25 bg-linear-to-r from-red-950 via-red-900/85 to-red-950 px-5 py-4 shadow-[0_0_50px_rgba(153,27,27,0.32)] lg:flex-row lg:items-center lg:justify-between"
    >
      <div>
        <p className="text-lg font-semibold tracking-wide text-red-100">{alert.title}</p>
        <p className="mt-1 text-sm text-red-200/80">{alert.cameraLabel}</p>
      </div>
      <div className="rounded-xl border border-white/10 bg-black/20 px-4 py-3 text-center">
        <p className="text-[0.65rem] font-semibold tracking-[0.18em] text-red-200/80 uppercase">
          Tiempo transcurrido
        </p>
        <p className="text-2xl font-semibold text-white">{elapsedTime}</p>
      </div>
    </section>
  )
}
