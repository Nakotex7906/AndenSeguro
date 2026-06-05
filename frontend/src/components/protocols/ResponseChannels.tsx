import type { ReactElement } from 'react'

import type { ResponseChannel } from '../../types/dashboard'

interface Props {
  channels: ResponseChannel[]
  onCall: (id: string) => void
}

const iconColor: Record<ResponseChannel['tone'], string> = {
  slate:   '#9ca3af',
  blue:    '#38bdf8',
  amber:   '#f59e0b',
  red:     '#f87171',
  emerald: '#22c55e',
  purple:  '#a78bfa',
  gray:    '#6b7280',
  orange:  '#fb923c',
  pink:    '#f472b6',
}

const iconBg: Record<ResponseChannel['tone'], string> = {
  slate:   '#1c1e21',
  blue:    '#0e1929',
  amber:   '#1f1508',
  red:     '#1a0808',
  emerald: '#091a11',
  purple:  '#160e29',
  gray:    '#161719',
  orange:  '#1a0e05',
  pink:    '#1a0814',
}

/** Iconos SVG por tipo de canal */
function ChannelIcon({ type, color }: { type: ResponseChannel['icon']; color: string }): ReactElement {
  const icons: Record<ResponseChannel['icon'], ReactElement> = {
    security:     <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2L4 6v6c0 5.5 3.8 10.7 8 12 4.2-1.3 8-6.5 8-12V6L12 2z" stroke={color} strokeWidth="1.6" strokeLinejoin="round"/></svg>,
    firefighters: <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2c0 0-6 4-6 10 0 3.3 2.7 6 6 6s6-2.7 6-6c0-6-6-10-6-10z" stroke={color} strokeWidth="1.6" strokeLinejoin="round"/><path d="M12 12c0 0-2 1.5-2 3s.9 2 2 2 2-.9 2-2-2-3-2-3z" stroke={color} strokeWidth="1.4"/></svg>,
    paramedics:   <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="3" stroke={color} strokeWidth="1.6"/><path d="M12 8v8M8 12h8" stroke={color} strokeWidth="1.8" strokeLinecap="round"/></svg>,
    megaphone:    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M18 8.5c1.5.8 2.5 2.3 2.5 4s-1 3.2-2.5 4" stroke={color} strokeWidth="1.6" strokeLinecap="round"/><path d="M3 9h4l8-5v14l-8-5H3V9z" stroke={color} strokeWidth="1.6" strokeLinejoin="round"/></svg>,
    health:       <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>,
    police:       <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2l2 4h4l-3 3 1 4-4-2-4 2 1-4-3-3h4z" stroke={color} strokeWidth="1.6" strokeLinejoin="round"/><rect x="8" y="16" width="8" height="6" rx="1" stroke={color} strokeWidth="1.4"/></svg>,
  }
  return icons[type]
}

/**
 * Grid de canales de respuesta rápida.
 * Cada canal muestra su icono, label y teléfono/extensión.
 * Al hacer click registra el contacto en el protocolo.
 * Reutilizable en alertas críticas.
 */
export function ResponseChannels({ channels, onCall }: Props): ReactElement {
  return (
    <div>
      <p style={{ fontSize: '0.62rem', letterSpacing: '0.16em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase', marginBottom: 10 }}>
        Canales de respuesta rápida
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {channels.map((ch) => {
          const color = iconColor[ch.tone]
          const bg    = iconBg[ch.tone]
          return (
            <button
              key={ch.id}
              type="button"
              onClick={() => onCall(ch.id)}
              style={{
                backgroundColor: '#0d0e10',
                border: '1px solid #1f2023',
                borderRadius: 8,
                padding: '10px 10px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 6,
                cursor: 'pointer',
                transition: 'background-color 0.15s, border-color 0.15s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = bg
                e.currentTarget.style.borderColor = color + '40'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#0d0e10'
                e.currentTarget.style.borderColor = '#1f2023'
              }}
            >
              <ChannelIcon type={ch.icon} color={color} />
              <span style={{ fontSize: '0.64rem', fontWeight: 600, color: '#d1d5db', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                {ch.label}
              </span>
              <span style={{ fontSize: '0.6rem', color: '#4b4f56' }}>{ch.phone}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}