import type { ReactNode } from 'react'
import type { ReactElement } from 'react'

import type { ViewId } from '../../types/dashboard'
import { Header } from './Header'
import { Sidebar } from './Sidebar'

/**
 * Propiedades del contenedor estructural de la aplicación.
 */
export interface AppLayoutProps {
  activeView: ViewId
  isNavigationPending: boolean
  children: ReactNode
  onViewChange: (viewId: ViewId) => void
}

/**
 * Enmarca la interfaz completa con navegación, encabezado y contenido.
 * @param props - Vista actual, callback de navegación y contenido interno.
 * @returns La estructura principal de la aplicación.
 */
export function AppLayout({
  activeView,
  children,
  isNavigationPending,
  onViewChange,
}: AppLayoutProps): ReactElement {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(15,23,42,0.96),rgba(2,6,23,0.99)_45%,rgba(0,0,0,1)_100%)] text-slate-100">
      <div className="grid min-h-screen lg:grid-cols-[18rem_minmax(0,1fr)]">
        <Sidebar activeView={activeView} onViewChange={onViewChange} />
        <div className="flex min-h-screen flex-col">
          <Header brandLabel="ANDEN SEGURO" isNavigationPending={isNavigationPending} />
          <main className="flex-1 px-4 py-4 sm:px-6 lg:px-8 lg:py-6">{children}</main>
        </div>
      </div>
    </div>
  )
}
