import type { ReactNode, ReactElement } from 'react'

import type { ViewId } from '../../types/dashboard'
import { Header } from './Header'
import { Sidebar } from './Sidebar'

export interface AppLayoutProps {
  activeView: ViewId
  isNavigationPending: boolean
  children: ReactNode
  onViewChange: (viewId: ViewId) => void
  onLogout?: () => void
  userRole?: string | null
}

export function AppLayout({
  activeView,
  children,
  isNavigationPending,
  onViewChange,
  onLogout,
  userRole,
}: AppLayoutProps): ReactElement {
  return (
    <div style={{ backgroundColor: '#0d0e10', minHeight: '100vh' }} className="text-slate-100">
      {/* Sidebar fijo */}
      <Sidebar activeView={activeView} onViewChange={onViewChange} onLogout={onLogout} userRole={userRole} />

      {/* Contenido desplazado exactamente el ancho del sidebar (16rem = 256px) */}
      <div className="ml-64 flex min-h-screen flex-col">
        <Header brandLabel="ANDEN SEGURO" isNavigationPending={isNavigationPending} />
        <main
          style={{ backgroundColor: '#0d0e10' }}
          className="flex-1 px-7 py-6"
        >
          {children}
        </main>
      </div>
    </div>
  )
}