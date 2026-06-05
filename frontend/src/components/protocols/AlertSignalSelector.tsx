import type { ReactElement } from 'react'

import type { AlertSignal } from '../../types/dashboard'

interface Props {
  signals: AlertSignal[]
  onToggle: (id: string) => void
}

/**
 * Menú de selección rápida de señales de alerta observables.
 * El operador marca las conductas que identifica visualmente.
 * Reutilizable en alertas críticas y protocolos.
 */
export function AlertSignalSelector({ signals, onToggle }: Props): ReactElement {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <p style={{ fontSize: '0.62rem', letterSpacing: '0.16em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase', marginBottom: 4 }}>
        Señales de alerta observadas
      </p>
      {signals.map((signal) => (
        <button
          key={signal.id}
          type="button"
          onClick={() => onToggle(signal.id)}
          style={{
            backgroundColor: signal.selected ? '#0e1929' : '#0d0e10',
            border: `1px solid ${signal.selected ? '#1a3451' : '#1f2023'}`,
            borderRadius: 6,
            padding: '8px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            cursor: 'pointer',
            transition: 'background-color 0.15s, border-color 0.15s',
            textAlign: 'left',
          }}
        >
          {/* Checkbox visual */}
          <span style={{
            width: 14, height: 14, borderRadius: 3, flexShrink: 0,
            backgroundColor: signal.selected ? '#38bdf8' : 'transparent',
            border: `1.5px solid ${signal.selected ? '#38bdf8' : '#3a3d41'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background-color 0.15s, border-color 0.15s',
          }}>
            {signal.selected && (
              <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                <polyline points="1,4 3,6 7,2" stroke="#0d0e10" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </span>
          <span style={{ fontSize: '0.73rem', color: signal.selected ? '#d1d5db' : '#6b7280', fontWeight: signal.selected ? 500 : 400 }}>
            {signal.label}
          </span>
        </button>
      ))}
    </div>
  )
}