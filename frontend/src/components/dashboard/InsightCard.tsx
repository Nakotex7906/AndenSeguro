import type { ReactElement } from 'react'

import type { CameraInsight } from '../../types/dashboard'

/**
 * Propiedades de una tarjeta de información breve.
 */
export interface InsightCardProps {
  insight: CameraInsight
}

const toneClasses: Record<CameraInsight['tone'], string> = {
  slate: 'border-slate-400/30 bg-slate-400/10 text-slate-100',
  blue: 'border-sky-400/30 bg-sky-400/10 text-sky-100',
  amber: 'border-amber-400/30 bg-amber-400/10 text-amber-100',
  red: 'border-red-500/30 bg-red-500/10 text-red-100',
  emerald: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
}

/**
 * Renderiza una métrica compacta de la cámara con tonos críticos.
 * @param props - Información resumida del evento observado.
 * @returns Una tarjeta pequeña reutilizable.
 */
export function InsightCard({ insight }: InsightCardProps): ReactElement {
  return (
    <article className={`rounded-2xl border p-4 ${toneClasses[insight.tone]} bg-slate-900/80`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-slate-400 uppercase">
            {insight.label}
          </p>
          <p className="mt-2 text-3xl font-semibold tracking-tight text-white">{insight.value}</p>
        </div>
        <span className="mt-1 h-2.5 w-2.5 rounded-full bg-current opacity-80" aria-hidden="true" />
      </div>
    </article>
  )
}
