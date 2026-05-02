import { BellSimpleIcon, GearSixIcon } from '@phosphor-icons/react'
import type { ReactElement } from 'react'

/**
 * Propiedades para el encabezado global del panel.
 */
export interface HeaderProps {
  brandLabel: string
  isNavigationPending: boolean
}

/**
 * Renderiza el encabezado superior con la marca y acciones rápidas.
 * @param props - Configuración visual del encabezado.
 * @returns El encabezado superior del sistema.
 */
export function Header({ brandLabel, isNavigationPending }: HeaderProps): ReactElement {
  return (
    <header className="flex items-center justify-between border-b border-white/5 bg-slate-950/90 px-4 py-4 backdrop-blur-xl sm:px-6 lg:px-8">
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-slate-700/70 shadow-[0_0_24px_rgba(148,163,184,0.18)]">
          <div className="h-8 w-8 rounded-full bg-slate-400/80" aria-hidden="true" />
        </div>
        <div>
          <p className="text-lg font-semibold tracking-[0.18em] text-slate-200 uppercase">
            {brandLabel}
          </p>
          <p className="text-xs text-slate-400">Monitoreo de estaciones y alertas operativas</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div
          className={
            isNavigationPending
              ? 'hidden rounded-full border border-amber-400/20 bg-amber-400/10 px-3 py-2 text-xs font-semibold tracking-[0.16em] text-amber-200 uppercase sm:block'
              : 'hidden rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-xs font-semibold tracking-[0.16em] text-emerald-200 uppercase sm:block'
          }
        >
          {isNavigationPending ? 'Sincronizando' : 'Operativo'}
        </div>
        <button
          aria-label="Ver notificaciones"
          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-slate-900/80 text-slate-300 transition hover:border-slate-400/40 hover:text-white"
          type="button"
        >
          <BellSimpleIcon size={18} weight="regular" />
        </button>
        <button
          aria-label="Abrir configuración"
          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-slate-900/80 text-slate-300 transition hover:border-slate-400/40 hover:text-white"
          type="button"
        >
          <GearSixIcon size={18} weight="regular" />
        </button>
      </div>
    </header>
  )
}
