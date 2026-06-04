import {
  ClockCounterClockwiseIcon,
  IdentificationCardIcon,
  SignOutIcon,
  SirenIcon,
  SquaresFourIcon,
  VideoCameraIcon,
} from '@phosphor-icons/react'
import type { ReactElement } from 'react'

import type { ViewId } from '../../types/dashboard'

interface SidebarItem {
  id: ViewId
  label: string
  icon: ReactElement
}

export interface SidebarProps {
  activeView: ViewId
  onViewChange: (viewId: ViewId) => void
}

const mainItems: SidebarItem[] = [
  { id: 'dashboard',  label: 'Panel de control',    icon: <SquaresFourIcon size={16} /> },
  { id: 'camera',     label: 'Cámara en vivo',       icon: <VideoCameraIcon size={16} /> },
  { id: 'alerts',     label: 'Alertas Críticas',     icon: <SirenIcon size={16} /> },
  { id: 'protocols',  label: 'Pasos de Protocolo',   icon: <IdentificationCardIcon size={16} /> },
  { id: 'history',    label: 'Datos Históricos',     icon: <ClockCounterClockwiseIcon size={16} /> },
]

export function Sidebar({ activeView, onViewChange }: SidebarProps): ReactElement {
  return (
    <aside
      style={{ backgroundColor: '#111214', borderRight: '1px solid #1f2023' }}
      className="fixed top-0 left-0 z-30 flex h-screen w-64 flex-col px-4 py-5"
    >
      {/* ── Logo + marca ── */}
      <div className="mb-7 flex items-center gap-2.5 px-2">
        <img
          src="/logo.png"
          alt="Andén Seguro"
          style={{ width: 34, height: 34, objectFit: 'contain' }}
          onError={(e) => { e.currentTarget.style.display = 'none' }}
        />
        <span
          style={{ fontSize: '0.7rem', letterSpacing: '0.22em', fontWeight: 700, color: '#e2e2e2' }}
          className="uppercase"
        >
          Anden Seguro
        </span>
      </div>

      {/* ── Nav principal ── */}
      <nav className="flex flex-1 flex-col" aria-label="Centro de control">
        <p
          style={{ fontSize: '0.6rem', letterSpacing: '0.2em', color: '#4b4f56', fontWeight: 600 }}
          className="mb-1.5 px-2 uppercase"
        >
          Centro de Control
        </p>

        <div className="flex flex-col gap-0.5">
          {mainItems.map((item) => {
            const isActive = item.id === activeView
            return (
              <button
                key={item.id}
                type="button"
                aria-pressed={isActive}
                onClick={() => onViewChange(item.id)}
                style={
                  isActive
                    ? { backgroundColor: '#1c1e21', color: '#f0f0f0' }
                    : { color: '#6b7280' }
                }
                className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[0.8rem] font-medium transition-colors hover:bg-[#1a1c1f] hover:text-[#d1d5db]"
              >
                <span style={{ color: isActive ? '#e2e2e2' : '#4b4f56' }}>
                  {item.icon}
                </span>
                {item.label}
              </button>
            )
          })}
        </div>

        <div className="flex-1" />

        {/* ── Soporte + Cerrar sesión ── */}
        <div
          style={{ borderTop: '1px solid #1f2023' }}
          className="flex flex-col gap-0.5 pt-3"
        >
          <button
            type="button"
            style={{ color: '#4b4f56' }}
            className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[0.8rem] transition-colors hover:bg-[#1a1c1f] hover:text-[#9ca3af]"
          >
            <IdentificationCardIcon size={16} />
            Soporte
          </button>
          <button
            type="button"
            style={{ color: '#4b4f56' }}
            className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[0.8rem] transition-colors hover:bg-[#1a1c1f] hover:text-[#9ca3af]"
          >
            <SignOutIcon size={16} />
            Cerrar Sesión
          </button>
        </div>
      </nav>
    </aside>
  )
}