import type { ReactElement } from 'react'
import { useState, useTransition } from 'react'

import { AppLayout } from './components/layout/AppLayout'
import { ComingSoonPage } from './pages/ComingSoonPage'
import { DashboardPage } from './pages/DashboardPage'
import { LiveCameraPage } from './pages/LiveCameraPage'
import { ProtocolsPage } from './pages/ProtocolsPage'
import type { ViewId } from './types/dashboard'

function App(): ReactElement {
  const [activeView, setActiveView] = useState<ViewId>('dashboard')
  const [isNavigationPending, startTransition] = useTransition()

  function handleViewChange(viewId: ViewId): void {
    startTransition(() => setActiveView(viewId))
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