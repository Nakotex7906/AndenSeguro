import type { ReactNode } from 'react'
import type { ReactElement } from 'react'

/**
 * Propiedades compartidas para las superficies de contenido.
 */
export interface PanelProps {
  action?: ReactNode
  children: ReactNode
  className?: string
  description?: string
  title: string
}

/**
 * Envuelve un bloque visual con título, subtítulo opcional y contenido.
 * @param props - Texto descriptivo y contenido interno del panel.
 * @returns Un contenedor visual reutilizable para el dashboard.
 */
export function Panel({
  action,
  children,
  className,
  description,
  title,
}: PanelProps): ReactElement {
  return (
    <section
      className={
        className ??
        'rounded-2xl border border-white/8 bg-slate-900/80 p-4 shadow-[0_0_40px_rgba(15,23,42,0.42)]'
      }
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">{title}</h2>
          {description ? <p className="mt-1 text-sm text-slate-400">{description}</p> : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      {children}
    </section>
  )
}
