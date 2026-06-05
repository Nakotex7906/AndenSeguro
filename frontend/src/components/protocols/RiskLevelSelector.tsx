import type { ReactElement } from 'react'

import type { RiskLevel } from '../../types/dashboard'

interface Props {
  value: RiskLevel | null
  onChange: (level: RiskLevel) => void
}

const LEVELS: { id: RiskLevel; label: string; description: string; color: string; bg: string; border: string }[] = [
  {
    id: 'leve',
    label: 'Leve',
    description: 'Ideas generales sin plan específico',
    color: '#22c55e',
    bg: '#091a11',
    border: '#12382a',
  },
  {
    id: 'moderado',
    label: 'Moderado',
    description: 'Métodos específicos identificados',
    color: '#f59e0b',
    bg: '#1f1508',
    border: '#3d2c0a',
  },
  {
    id: 'alto',
    label: 'Alto / Inminente',
    description: 'Plan detallado o intento en curso',
    color: '#f87171',
    bg: '#1a0808',
    border: '#5c1f1f',
  },
]

/**
 * Selector de nivel de riesgo C-SSRS.
 * Permite al operador clasificar la situación observada.
 * Reutilizable en otras vistas donde se requiera categorización de riesgo.
 */
export function RiskLevelSelector({ value, onChange }: Props): ReactElement {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <p style={{ fontSize: '0.62rem', letterSpacing: '0.16em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase', marginBottom: 4 }}>
        Escala C-SSRS — Nivel de riesgo
      </p>
      {LEVELS.map((level) => {
        const active = value === level.id
        return (
          <button
            key={level.id}
            type="button"
            onClick={() => onChange(level.id)}
            style={{
              backgroundColor: active ? level.bg : '#0d0e10',
              border: `1px solid ${active ? level.border : '#1f2023'}`,
              borderRadius: 7,
              padding: '10px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              cursor: 'pointer',
              transition: 'background-color 0.2s, border-color 0.2s',
              textAlign: 'left',
            }}
          >
            {/* Indicador */}
            <span style={{
              width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
              backgroundColor: active ? level.color : '#2a2d31',
              boxShadow: active ? `0 0 8px ${level.color}` : 'none',
              transition: 'background-color 0.2s, box-shadow 0.2s',
            }} />
            <div>
              <p style={{ fontSize: '0.78rem', fontWeight: 700, color: active ? level.color : '#9ca3af' }}>
                {level.label}
              </p>
              <p style={{ fontSize: '0.68rem', color: '#6b7280', marginTop: 1 }}>
                {level.description}
              </p>
            </div>
          </button>
        )
      })}
    </div>
  )
}