import { ArrowsOutIcon, CaretDownIcon, CaretUpIcon, XIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import type { ReactElement } from 'react'

import { MetricCard } from '../components/dashboard/MetricCard'
import { MetroMapSvg } from '../components/dashboard/MetroMapSvg'
import { Panel } from '../components/dashboard/Panel'
import { StatusList } from '../components/dashboard/StatusList'
import { Button } from '../components/ui/Button'
import { useDashboardOverview } from '../hooks/useDashboardOverview'
import { useElapsedTimer } from '../hooks/useElapsedTimer'
import type { DashboardOverview, ViewId } from '../types/dashboard'

export interface DashboardPageProps {
  onViewChange: (viewId: ViewId) => void
}

/** Color y fondo del badge de systemStatus según el estado */
const statusStyle: Record<DashboardOverview['systemStatus'], React.CSSProperties> = {
  'OPERATIVO':         { color: '#22c55e', backgroundColor: '#091a11', border: '1px solid #12382a' },
  'PRECAUCIÓN':        { color: '#f59e0b', backgroundColor: '#1f1508', border: '1px solid #3d2c0a' },
  'FUERA DE SERVICIO': { color: '#f87171', backgroundColor: '#1a0808', border: '1px solid #3d1212' },
}

export function DashboardPage({ onViewChange: _onViewChange }: DashboardPageProps): ReactElement | null {
  const { data, error } = useDashboardOverview()
  const { elapsedTime }  = useElapsedTimer(data?.uptimeSeconds ?? 0)
  const [mapExpanded,   setMapExpanded]   = useState(false)
  const [linesExpanded, setLinesExpanded] = useState(false)

  if (error) {
    return (
      <section style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 10, padding: 24 }}>
        <h1 style={{ fontSize: '1.4rem', fontWeight: 600, color: '#f0f0f0' }}>Panel de Control Global</h1>
        <p style={{ marginTop: 12, color: '#9ca3af' }}>{error}</p>
      </section>
    )
  }
  if (!data) return null

  const ss = statusStyle[data.systemStatus]

  return (
    <>
      {/* ── Modal: mapa expandido ── */}
      {mapExpanded && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Mapa de líneas expandido"
          onClick={() => setMapExpanded(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 50,
            backgroundColor: 'rgba(0,0,0,0.85)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            padding: 40,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: '#161719', border: '1px solid #242628',
              borderRadius: 12, padding: 24, width: '100%', maxWidth: 1000,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 18 }}>
              <div>
                <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f0f0f0' }}>{data.mapTitle}</h2>
                <p style={{ fontSize: '0.78rem', color: '#6b7280', marginTop: 3 }}>{data.mapSubtitle}</p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setMapExpanded(false)}>
                <XIcon size={14} /> Cerrar
              </Button>
            </div>
            <MetroMapSvg expanded />
          </div>
        </div>
      )}

      <section className="space-y-5">

        {/* ── Cabecera ── */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 style={{ fontSize: '1.9rem', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.02em' }}>
              {data.title}
            </h1>
            {/* Subtítulo con badge de systemStatus coloreado dinámicamente */}
            <p style={{ fontSize: '0.82rem', color: '#6b7280', marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              {data.subtitle}
              <span
                style={{
                  ...ss,
                  fontSize: '0.6rem',
                  fontWeight: 700,
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                  padding: '3px 10px',
                  borderRadius: 4,
                  transition: 'background-color 0.4s, color 0.4s, border-color 0.4s',
                }}
              >
                {data.systemStatus}
              </span>
            </p>
          </div>

          {/* Hora actual */}
          <div style={{
            backgroundColor: '#161719', border: '1px solid #242628',
            borderRadius: 9, padding: '10px 18px', textAlign: 'right', minWidth: 140, flexShrink: 0,
          }}>
            <p style={{ fontSize: '0.56rem', letterSpacing: '0.2em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
              Hora actual
            </p>
            <p style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f0f0f0', letterSpacing: '0.06em', marginTop: 4, fontVariantNumeric: 'tabular-nums' }}>
              {elapsedTime}
            </p>
          </div>
        </div>

        {/* ── 4 MetricCards ── */}
        <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
          {data.metrics.map((metric) => (
            <MetricCard key={metric.id} metric={metric} />
          ))}
        </div>

        {/* ── Mapa + Matriz ── */}
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(16rem,0.75fr)]">

          {/* Panel: mapa */}
          <Panel
            title={data.mapTitle}
            description={data.mapSubtitle}
            action={
              <Button variant="ghost" size="sm" onClick={() => setMapExpanded(true)}>
                <ArrowsOutIcon size={13} /> Ver más…
              </Button>
            }
          >
            <div style={{ backgroundColor: '#0d0e10', border: '1px solid #1f2023', borderRadius: 9, padding: 12, marginTop: 4 }}>
              <MetroMapSvg />
            </div>
          </Panel>

          {/* Panel: matriz de estado */}
          <Panel title="Matriz de Estado" description="Resumen de operatividad por línea">
            {/* Lista principal con las primeras 4 líneas */}
            <StatusList items={data.lineStatuses.slice(0, 4)} />

            <Button
              variant="outline"
              size="sm"
              fullWidth
              className="mt-3.5"
              onClick={() => setLinesExpanded((v) => !v)}
            >
              {linesExpanded ? 'Ocultar detalles' : 'Ver detalles de todas las líneas'}
              {linesExpanded ? <CaretUpIcon size={11} /> : <CaretDownIcon size={11} />}
            </Button>

            {/* Desplegable de detalles */}
            {linesExpanded && (
              <div style={{ marginTop: 10, borderTop: '1px solid #1f2023', paddingTop: 12 }}>
                {data.lineStatuses.length > 4 ? (
                  /* Reutilizamos StatusList para mantener exactamente el mismo diseño, colores e información */
                  <StatusList items={data.lineStatuses.slice(4)} />
                ) : (
                  <p style={{ fontSize: '0.78rem', color: '#4b4f56', textAlign: 'center', padding: '12px 0' }}>
                    No hay líneas adicionales registradas.
                  </p>
                )}
              </div>
            )}
          </Panel>

        </div>
      </section>
    </>
  )
}