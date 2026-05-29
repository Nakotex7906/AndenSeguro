import {
  ClockCounterClockwiseIcon,
  IdentificationCardIcon,
  MonitorIcon,
  SignOutIcon,
  SirenIcon,
  SquaresFourIcon,
  VideoCameraIcon,
} from '@phosphor-icons/react'
import type { ReactElement } from 'react'

import type { ViewId } from '../../types/dashboard'

/**
 * Define un elemento navegable del menú lateral.
 */
interface SidebarItem {
  id: ViewId
  label: string
  icon: ReactElement
}

/**
 * Propiedades requeridas por el menú lateral.
 */
export interface SidebarProps {
  activeView: ViewId
  onViewChange: (viewId: ViewId) => void
}

const sidebarItems: SidebarItem[] = [
  { id: 'dashboard', label: 'Panel de control', icon: <SquaresFourIcon size={16} /> },
  { id: 'camera', label: 'Cámara en vivo', icon: <VideoCameraIcon size={16} /> },
  { id: 'alerts', label: 'Alertas críticas', icon: <SirenIcon size={16} /> },
  { id: 'protocols', label: 'Protocolos', icon: <IdentificationCardIcon size={16} /> },
  { id: 'history', label: 'Historial', icon: <ClockCounterClockwiseIcon size={16} /> },
]

/**
 * Renderiza la navegación lateral y las acciones críticas del sistema.
 * @param props - Estado activo y callback de navegación.
 * @returns La barra lateral del dashboard.
 */
export function Sidebar({ activeView, onViewChange }: SidebarProps): ReactElement {
  return (
    <aside className="flex h-full w-full flex-col border-r border-white/5 bg-slate-950/85 px-4 py-5 backdrop-blur-xl lg:w-72 lg:px-5">
      <div className="flex items-center gap-3 px-2 pb-8">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-400/80 text-slate-950">
          <MonitorIcon size={18} weight="fill" />
        </div>
        <div>
          <p className="text-base font-semibold tracking-[0.16em] text-slate-200 uppercase">
            Anden Seguro
          </p>
        </div>
      </div>

      <nav aria-label="Centro de control" className="flex flex-1 flex-col gap-2">
        <p className="px-2 text-sm font-semibold tracking-[0.12em] text-slate-200 uppercase">
          Centro de control
        </p>
        <div className="flex flex-col gap-1">
          {sidebarItems.map((item) => {
            const isActive = item.id === activeView
            const inactiveOpacity =
              item.id === 'alerts' || item.id === 'protocols' || item.id === 'history'

            return (
              <button
                key={item.id}
                aria-pressed={isActive}
                className={
                  isActive
                    ? 'flex items-center gap-3 rounded-lg border border-sky-400/30 bg-sky-400/10 px-3 py-3 text-left text-sm font-medium text-sky-200'
                    : 'flex items-center gap-3 rounded-lg border border-transparent px-3 py-3 text-left text-sm text-slate-400 transition hover:border-white/10 hover:bg-white/5 hover:text-slate-100'
                }
                onClick={() => {
                  onViewChange(item.id)
                }}
                type="button"
              >
                <span className={isActive ? 'text-sky-300' : 'text-inherit'}>{item.icon}</span>
                <span className={inactiveOpacity ? 'opacity-90' : 'opacity-100'}>{item.label}</span>
              </button>
            )
          })}
        </div>
      </nav>

      <div className="mt-6 space-y-3">
        <button
          className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm text-slate-400 transition hover:bg-white/5 hover:text-slate-100"
          type="button"
        >
          <IdentificationCardIcon size={16} />
          <span>Soporte</span>
        </button>
        <button
          className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm text-slate-400 transition hover:bg-white/5 hover:text-slate-100"
          type="button"
        >
          <SignOutIcon size={16} />
          <span>Cerrar sesión</span>
        </button>
      </div>
    </aside>
  )
}
