import type { ReactElement } from 'react'

import type { ProtocolStep } from '../../types/dashboard'

interface Props {
  steps: ProtocolStep[]
  onToggle: (id: string) => void
}

/**
 * Lista de pasos del protocolo operativo con estado de completitud.
 * El operador marca cada paso a medida que lo ejecuta.
 * Reutilizable en cualquier vista que requiera un checklist de pasos.
 */
export function ProtocolSteps({ steps, onToggle }: Props): ReactElement {
  const completed = steps.filter((s) => s.completed).length

  return (
    <div>
      {/* Barra de progreso */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <p style={{ fontSize: '0.62rem', letterSpacing: '0.16em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
            Guía operativa — Pasos
          </p>
          <span style={{ fontSize: '0.68rem', color: '#6b7280' }}>{completed}/{steps.length}</span>
        </div>
        <div style={{ backgroundColor: '#1f2023', borderRadius: 4, height: 3 }}>
          <div style={{
            backgroundColor: completed === steps.length ? '#22c55e' : '#38bdf8',
            borderRadius: 4,
            height: 3,
            width: `${(completed / steps.length) * 100}%`,
            transition: 'width 0.4s ease, background-color 0.4s',
          }} />
        </div>
      </div>

      {/* Pasos */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {steps.map((step, idx) => (
          <button
            key={step.id}
            type="button"
            onClick={() => onToggle(step.id)}
            style={{
              backgroundColor: step.completed ? '#091a11' : '#0d0e10',
              border: `1px solid ${step.completed ? '#12382a' : '#1f2023'}`,
              borderRadius: 8,
              padding: '12px 14px',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 12,
              cursor: 'pointer',
              transition: 'background-color 0.2s, border-color 0.2s',
              textAlign: 'left',
            }}
          >
            {/* Número / check */}
            <span style={{
              width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
              backgroundColor: step.completed ? '#22c55e' : '#1c1e21',
              border: `1.5px solid ${step.completed ? '#22c55e' : '#3a3d41'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.6rem', fontWeight: 700,
              color: step.completed ? '#0d0e10' : '#6b7280',
              transition: 'background-color 0.2s, border-color 0.2s',
            }}>
              {step.completed
                ? <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><polyline points="1.5,5 4,7.5 8.5,2.5" stroke="#0d0e10" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
                : idx + 1}
            </span>
            <div>
              <p style={{ fontSize: '0.78rem', fontWeight: 600, color: step.completed ? '#6b7280' : '#e2e2e2', textDecoration: step.completed ? 'line-through' : 'none' }}>
                {step.title}
              </p>
              <p style={{ fontSize: '0.7rem', color: '#4b4f56', marginTop: 3, lineHeight: 1.5 }}>
                {step.description}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}