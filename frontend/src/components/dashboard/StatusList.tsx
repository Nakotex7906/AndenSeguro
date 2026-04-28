import type { ReactElement } from 'react'

import type { LineStatus } from '../../types/dashboard'

/**
 * Propiedades para el listado de estados de las líneas.
 */
export interface StatusListProps {
  items: LineStatus[]
}

const toneIndicatorClasses: Record<LineStatus['tone'], string> = {
  slate: 'border-slate-500/30 bg-slate-500/10 text-slate-100',
  blue: 'border-sky-500/30 bg-sky-500/10 text-sky-100',
  amber: 'border-amber-500/30 bg-amber-500/10 text-amber-100',
  red: 'border-red-500/30 bg-red-500/10 text-red-100',
  emerald: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100',
}

/**
 * Presenta el estado resumido de cada línea en una lista apilada.
 * @param props - Estados a mostrar dentro del panel.
 * @returns Un listado de indicadores operativos.
 */
export function StatusList({ items }: StatusListProps): ReactElement {
  return (
    <div className="flex flex-col gap-3">
      {items.map((item) => {
        const toneClassName = toneIndicatorClasses[item.tone]

        return (
          <article
            key={item.id}
            className="flex items-center gap-4 rounded-xl border border-white/6 bg-slate-950/55 px-4 py-4"
          >
            <div
              className={`flex h-11 w-11 items-center justify-center rounded-md border text-sm font-semibold ${toneClassName}`}
            >
              {item.code}
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-base font-semibold text-white">{item.name}</h3>
              <p className="mt-0.5 text-sm text-slate-400">{item.detail}</p>
            </div>
          </article>
        )
      })}
    </div>
  )
}
