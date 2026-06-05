import { BellSimpleIcon, GearSixIcon, UserCircleIcon } from '@phosphor-icons/react'
import type { ReactElement } from 'react'

export interface HeaderProps {
  brandLabel: string
  isNavigationPending: boolean
}

export function Header({ brandLabel, isNavigationPending }: HeaderProps): ReactElement {
  return (
    <header
      style={{
        backgroundColor: '#111214',
        borderBottom: '1px solid #1f2023',
        height: '52px',
      }}
      className="sticky top-0 z-20 flex items-center justify-between px-6"
    >
      {/* Izquierda: logo + marca */}
      <div className="flex items-center gap-2.5">
        <img
          src="/logo.png"
          alt=""
          aria-hidden="true"
          style={{ width: 30, height: 30, objectFit: 'contain' }}
          onError={(e) => { e.currentTarget.style.display = 'none' }}
        />
        <span
          style={{ fontSize: '0.68rem', letterSpacing: '0.22em', fontWeight: 700, color: '#d1d5db' }}
          className="uppercase"
        >
          {brandLabel}
        </span>
      </div>

      {/* Centro: buscador */}
      <div className="mx-8 hidden max-w-sm flex-1 md:block">
        <label htmlFor="topbar-search" className="sr-only">Buscar estación o alerta</label>
        <div className="relative">
          <span className="pointer-events-none absolute inset-y-0 left-2.5 flex items-center">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="6.5" cy="6.5" r="5" stroke="#4b4f56" strokeWidth="1.6"/>
              <path d="M10.5 10.5L14 14" stroke="#4b4f56" strokeWidth="1.6" strokeLinecap="round"/>
            </svg>
          </span>
          <input
            id="topbar-search"
            type="search"
            placeholder="Buscar estación o alerta…"
            style={{
              backgroundColor: '#181a1d',
              border: '1px solid #2a2d31',
              color: '#d1d5db',
              fontSize: '0.75rem',
            }}
            className="h-7 w-full rounded-md pl-8 pr-3 placeholder:text-[#4b4f56] focus:border-[#3a3d41] focus:outline-none"
          />
        </div>
      </div>

      {/* Derecha: estado + acciones */}
      <div className="flex items-center gap-2">
        {/* Badge estado operativo */}
        <span
          style={
            isNavigationPending
              ? { backgroundColor: '#1f1a0e', border: '1px solid #3d2f0a', color: '#ca8a04', fontSize: '0.62rem', letterSpacing: '0.14em' }
              : { backgroundColor: '#0e1a14', border: '1px solid #1a3826', color: '#22c55e', fontSize: '0.62rem', letterSpacing: '0.14em' }
          }
          className="hidden rounded-full px-3 py-1 font-semibold uppercase sm:inline-flex items-center gap-1.5"
        >
          <span
            style={{
              width: 6, height: 6, borderRadius: '50%',
              backgroundColor: isNavigationPending ? '#ca8a04' : '#22c55e',
              display: 'inline-block',
            }}
          />
          {isNavigationPending ? 'Sincronizando' : 'Operativo'}
        </span>

        {/* Campana */}
        <button
          aria-label="Ver notificaciones"
          type="button"
          style={{ backgroundColor: '#181a1d', border: '1px solid #2a2d31', color: '#6b7280' }}
          className="flex h-7 w-7 items-center justify-center rounded-md transition hover:border-[#3a3d41] hover:text-[#d1d5db]"
        >
          <BellSimpleIcon size={14} />
        </button>

        {/* Configuración */}
        <button
          aria-label="Abrir configuración"
          type="button"
          style={{ backgroundColor: '#181a1d', border: '1px solid #2a2d31', color: '#6b7280' }}
          className="flex h-7 w-7 items-center justify-center rounded-md transition hover:border-[#3a3d41] hover:text-[#d1d5db]"
        >
          <GearSixIcon size={14} />
        </button>

        {/* Avatar */}
        <button
          aria-label="Perfil de usuario"
          type="button"
          style={{ backgroundColor: '#1c1e21', border: '1px solid #2a2d31', color: '#9ca3af' }}
          className="flex h-7 w-7 items-center justify-center rounded-full transition hover:border-[#3a3d41] hover:text-[#d1d5db]"
        >
          <UserCircleIcon size={17} weight="fill" />
        </button>
      </div>
    </header>
  )
}