import type { ReactElement } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { useDashboardOverview } from '../hooks/useDashboardOverview'

/**
 * Renderiza el panel principal con métricas, mapa y estado de líneas.
 * @returns La vista principal del sistema de control.
 */
export function DashboardPage(): ReactElement | null {
  const { error } = useDashboardOverview()

  return error ? (
    <section className="surface-panel p-6">
      <h1 className="text-2xl font-semibold text-white">Panel de Control Global</h1>
      <p className="mt-3 text-slate-300">{error}</p>
    </section>
  ) : (
    <section className="space-y-5">
      {/* TODO: Aquí se debería llamar al backend para obtener el panel global y renderizar métricas, mapa y estado de líneas con TanStack Query. */}
      <Panel
        title="Panel de Control Global"
        description="Integración pendiente con el backend"
        className="surface-panel p-5"
      >
        <div className="grid gap-4 xl:grid-cols-4">
          <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-sm text-slate-400">Estaciones activas</p>
            <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-sm text-slate-400">Alertas activas</p>
            <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-sm text-slate-400">Pasajeros en la estación</p>
            <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-sm text-slate-400">Alertas críticas</p>
            <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
          </div>
        </div>

        <div className="mt-5 rounded-2xl border border-white/6 bg-slate-950/65 p-5">
          <p className="text-sm font-semibold tracking-[0.18em] text-slate-400 uppercase">
            Mapa de líneas interactivo
          </p>
          <p className="mt-2 text-slate-300">
            {/* TODO: Este bloque debería renderizar el mapa y el estado de líneas con datos reales obtenidos del backend. */}
            Visualización pendiente de integración.
          </p>
        </div>

        <div className="mt-5 rounded-2xl border border-white/6 bg-slate-950/65 p-5">
          <p className="text-sm font-semibold tracking-[0.18em] text-slate-400 uppercase">
            Matriz de estado
          </p>
          <p className="mt-2 text-slate-300">
            {/* TODO: Aquí se debería consumir el backend para listar el estado operativo de cada línea. */}
            Datos operativos pendientes de carga.
          </p>
        </div>
      </Panel>
    </section>
  )
}
