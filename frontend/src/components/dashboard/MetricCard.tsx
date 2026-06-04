import { useEffect, useRef, useState } from 'react'
import type { ReactElement } from 'react'

import type { DashboardMetric } from '../../types/dashboard'

export interface MetricCardProps {
  metric: DashboardMetric

  /** Si es true, el flash solo se activa en cambios de magnitud mayor (1 hora) */
  flashOnMajorChangeOnly?: boolean
}

const dotColor: Record<DashboardMetric['tone'], string> = {
  slate:   '#6b7280',
  blue:    '#38bdf8',
  amber:   '#f59e0b',
  red:     '#ef4444',
  emerald: '#22c55e',
}

const badgeStyle: Record<DashboardMetric['tone'], React.CSSProperties> = {
  slate:   { backgroundColor: '#1c1e21', color: '#9ca3af', border: '1px solid #2e3135' },
  blue:    { backgroundColor: '#0e1929', color: '#38bdf8', border: '1px solid #1a3451' },
  amber:   { backgroundColor: '#1f1508', color: '#f59e0b', border: '1px solid #3d2c0a' },
  red:     { backgroundColor: '#1a0808', color: '#f87171', border: '1px solid #3d1212' },
  emerald: { backgroundColor: '#091a11', color: '#22c55e', border: '1px solid #12382a' },
}

const flashColor: Record<DashboardMetric['tone'], string> = {
  slate:   'rgba(107,114,128,0.12)',
  blue:    'rgba(56,189,248,0.10)',
  amber:   'rgba(245,158,11,0.12)',
  red:     'rgba(239,68,68,0.12)',
  emerald: 'rgba(34,197,94,0.10)',
}

/**
 * Detecta si un cambio de valor de timer es de magnitud mayor.
 * "Magnitud mayor" = el valor de horas cambió, se cruzó una hora).
 * Formato esperado: "MM:SS" o "HH:MM:SS"
 */
function isHourCrossing(prev: string, next: string): boolean {
  const prevParts = prev.split(':')
  const nextParts = next.split(':')
  if (prevParts.length !== nextParts.length) return true   
  if (nextParts.length === 3) {
    return prevParts[0] !== nextParts[0]                   
  }
  return false                                             
}

/**
 * Tarjeta de métrica con icono, valor reactivo y badge de estado.
 * El flash visual solo se activa cuando corresponde según el tipo de métrica:
 * - Timers (uptime): solo al cruzar una hora completa
 * - Demás valores (pasajeros, alertas, estaciones): al cualquier cambio
 */
export function MetricCard({ metric }: MetricCardProps): ReactElement {
  const isTimer    = metric.value.includes(':')
  const valueFSize = isTimer ? '1.7rem' : '2.6rem'

  const prevValueRef = useRef(metric.value)
  const [flashing, setFlashing] = useState(false)

  useEffect(() => {
    const prev = prevValueRef.current
    const next = metric.value

    if (prev === next) return

    prevValueRef.current = next

    /* Para timers: solo flashear al cruzar una hora */
    const shouldFlash = isTimer ? isHourCrossing(prev, next) : true

    if (shouldFlash) {
      setFlashing(true)
      const t = setTimeout(() => setFlashing(false), 800)
      return () => clearTimeout(t)
    }
  }, [metric.value, isTimer])

  return (
    <article
      style={{
        backgroundColor: flashing ? flashColor[metric.tone] : '#161719',
        border: `1px solid ${flashing ? dotColor[metric.tone] + '40' : '#242628'}`,
        borderRadius: 10,
        padding: '18px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        transition: 'background-color 0.5s ease, border-color 0.5s ease',
      }}
    >
      {/* Fila: icono + dot */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <MetricIcon tone={metric.tone} />
        <span
          style={{
            width: 8, height: 8, borderRadius: '50%',
            backgroundColor: dotColor[metric.tone],
            display: 'inline-block',
            boxShadow: flashing ? `0 0 6px ${dotColor[metric.tone]}` : 'none',
            transition: 'box-shadow 0.5s ease',
          }}
          aria-hidden="true"
        />
      </div>

      {/* Label */}
      <span style={{ fontSize: '0.62rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
        {metric.label}
      </span>

      {/* Valor */}
      <p style={{
        fontSize: valueFSize,
        fontWeight: 600,
        color: flashing ? dotColor[metric.tone] : '#f0f0f0',
        lineHeight: 1,
        letterSpacing: isTimer ? '0.04em' : '-0.02em',
        marginTop: -4,
        fontVariantNumeric: 'tabular-nums',
        transition: 'color 0.5s ease',
      }}>
        {metric.value}
        {metric.unit && !isTimer && (
          <span style={{ fontSize: '1rem', color: '#6b7280', marginLeft: 4 }}>{metric.unit}</span>
        )}
      </p>

      {/* Badge */}
      <span
        style={{
          ...badgeStyle[metric.tone],
          display: 'inline-flex',
          alignSelf: 'flex-start',
          fontSize: '0.58rem',
          fontWeight: 700,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          padding: '3px 8px',
          borderRadius: 4,
        }}
      >
        {metric.caption}
      </span>
    </article>
  )
}

function MetricIcon({ tone }: { tone: DashboardMetric['tone'] }): ReactElement {
  const c = dotColor[tone]
  const icons: Record<DashboardMetric['tone'], ReactElement> = {
    blue: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <rect x="3" y="3" width="7" height="7" rx="1.5" stroke={c} strokeWidth="1.6"/>
        <rect x="14" y="3" width="7" height="7" rx="1.5" stroke={c} strokeWidth="1.6"/>
        <rect x="3" y="14" width="7" height="7" rx="1.5" stroke={c} strokeWidth="1.6"/>
        <rect x="14" y="14" width="7" height="7" rx="1.5" stroke={c} strokeWidth="1.6"/>
      </svg>
    ),
    slate: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" stroke={c} strokeWidth="1.6" strokeLinecap="round"/>
        <circle cx="9" cy="7" r="4" stroke={c} strokeWidth="1.6"/>
        <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" stroke={c} strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
    ),
    amber: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke={c} strokeWidth="1.6" strokeLinejoin="round"/>
        <line x1="12" y1="9" x2="12" y2="13" stroke={c} strokeWidth="1.6" strokeLinecap="round"/>
        <line x1="12" y1="17" x2="12.01" y2="17" stroke={c} strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
    red: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="10" stroke={c} strokeWidth="1.6"/>
        <line x1="12" y1="8" x2="12" y2="12" stroke={c} strokeWidth="1.6" strokeLinecap="round"/>
        <line x1="12" y1="16" x2="12.01" y2="16" stroke={c} strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
    emerald: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="10" stroke={c} strokeWidth="1.6"/>
        <polyline points="12 6 12 12 16 14" stroke={c} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  }
  return icons[tone]
}