import type { ReactElement } from 'react'

import type { DashboardMetric } from '../../types/dashboard'

/**
 * Propiedades de una tarjeta de métrica.
 */
export interface MetricCardProps {
  metric: DashboardMetric
}

const toneClasses: Record<DashboardMetric['tone'], string> = {
  slate: 'border-slate-400/35 bg-slate-400/10 text-slate-200',
  blue: 'border-sky-400/35 bg-sky-400/10 text-sky-200',
  amber: 'border-amber-400/35 bg-amber-400/10 text-amber-200',
  red: 'border-red-500/35 bg-red-500/10 text-red-200',
  emerald: 'border-emerald-400/35 bg-emerald-400/10 text-emerald-200',
}

/**
 * Renderiza una métrica de control con jerarquía visual clara.
 * @param props - Configuración de la métrica a mostrar.
 * @returns Una tarjeta compacta con estado y valor.
 */
export function MetricCard({ metric }: MetricCardProps): ReactElement {
  const containerTone = toneClasses[metric.tone]

  return (
    <article
      className={`rounded-2xl border border-white/8 bg-slate-900/80 p-4 ${containerTone} shadow-[0_0_24px_rgba(15,23,42,0.28)]`}
    >
      <div className="flex h-full flex-col justify-between gap-5">
        <div className="flex items-center justify-between gap-3 text-xs font-medium tracking-[0.18em] text-slate-400 uppercase">
          <span>{metric.label}</span>
          <span className="h-2.5 w-2.5 rounded-full bg-current opacity-80" aria-hidden="true" />
        </div>
        <div>
          <p className="text-4xl font-semibold tracking-tight text-white">{metric.value}</p>
          <p className="mt-2 text-xs font-semibold tracking-[0.14em] text-slate-300 uppercase">
            {metric.caption}
          </p>
        </div>
      </div>
    </article>
  )
}
