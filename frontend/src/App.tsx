import type { ReactElement } from 'react'
import { useState, useTransition } from 'react'

import { AppLayout } from './components/layout/AppLayout'
import { ComingSoonPage } from './pages/ComingSoonPage'
import { DashboardPage } from './pages/DashboardPage'
import { LiveCameraPage } from './pages/LiveCameraPage'
import { LoginPage } from './pages/LoginPage'
import { ProtocolsPage } from './pages/ProtocolsPage'
import type { ViewId } from './types/dashboard'

/**
 * Componente raíz. Gestiona autenticación y navegación entre vistas.
 * isAuthenticated es local hasta que se implemente AuthContext con JWT.
 *
 * TODO (backend): reemplazar isAuthenticated por useAuth() que valide
 * el token almacenado y redirija al login si expiró.
 */
function App(): ReactElement {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [activeView, setActiveView]           = useState<ViewId>('dashboard')
  const [isNavigationPending, startTransition] = useTransition()

  function handleLogin(): void {
    setIsAuthenticated(true)
  }

  function handleViewChange(viewId: ViewId): void {
    startTransition(() => setActiveView(viewId))
  }

  /* Sin sesión → pantalla de login */
  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />
  }

  const pageContent =
    activeView === 'dashboard' ? <DashboardPage /> :
    activeView === 'camera'    ? <LiveCameraPage /> :
    activeView === 'protocols' ? <ProtocolsPage /> :
    <ComingSoonPage viewId={activeView} />

  return (
    <AppLayout
      activeView={activeView}
      isNavigationPending={isNavigationPending}
      onViewChange={handleViewChange}
    >
      {pageContent}
    </AppLayout>
  )
}

export default App