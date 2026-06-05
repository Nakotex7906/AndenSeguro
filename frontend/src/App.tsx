import type { ReactElement } from 'react'
import { useState, useTransition, useEffect } from 'react'

import { AppLayout } from './components/layout/AppLayout'
import { ComingSoonPage } from './pages/ComingSoonPage'
import { DashboardPage } from './pages/DashboardPage'
import { LiveCameraPage } from './pages/LiveCameraPage'
import { LoginPage } from './pages/LoginPage'
import { ProtocolsPage } from './pages/ProtocolsPage'
import { UsersPage } from './pages/UsersPage'
import type { ViewId } from './types/dashboard'

function App(): ReactElement {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false)
  const [userRole, setUserRole] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<ViewId>('dashboard')
  const [isNavigationPending, startTransition] = useTransition()

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      fetch('http://localhost:8000/api/auth/me', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      .then(res => {
        if (!res.ok) {
          localStorage.removeItem('token')
          setIsAuthenticated(false)
          throw new Error('Token invalido')
        }
        return res.json()
      })
      .then(data => {
        setUserRole(data.role)
        setIsAuthenticated(true)
      })
      .catch(console.error)
    }
  }, [])

  function handleViewChange(viewId: ViewId): void {
    startTransition(() => setActiveView(viewId))
  }

  function handleLogout(): void {
    localStorage.removeItem('token')
    setIsAuthenticated(false)
    setUserRole(null)
    setActiveView('dashboard')
  }

  const pageContent =
    activeView === 'dashboard' ? <DashboardPage /> :
    activeView === 'camera'    ? <LiveCameraPage onViewChange={handleViewChange} /> :
    activeView === 'protocols' ? <ProtocolsPage /> :
    activeView === 'users' && userRole === 'admin' ? <UsersPage /> :
    <ComingSoonPage viewId={activeView} />

  if (!isAuthenticated) {
    return <LoginPage onLogin={() => {
      // Re-fetch para obtener el rol al iniciar sesión
      const token = localStorage.getItem('token')
      if (token) {
        fetch('http://localhost:8000/api/auth/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        })
        .then(res => res.json())
        .then(data => {
          setUserRole(data.role)
          setIsAuthenticated(true)
        })
      }
    }} />
  }

  return (
    <AppLayout
      activeView={activeView}
      isNavigationPending={isNavigationPending}
      onViewChange={handleViewChange}
      onLogout={handleLogout}
      userRole={userRole}
    >
      {pageContent}
    </AppLayout>
  )
}

export default App