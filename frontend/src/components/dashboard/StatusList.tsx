import type { ReactElement } from 'react'
import type { LineStatus } from '../../types/dashboard'

export interface StatusListProps {
  items: LineStatus[]
}

const badgeColors: Record<LineStatus['tone'], { bg: string; color: string; border: string }> = {
  slate:   { bg: '#1c1e21', color: '#9ca3af', border: '#2e3135' },
  blue:    { bg: '#0e1929', color: '#38bdf8', border: '#1a3451' },
  amber:   { bg: '#1a1305', color: '#f59e0b', border: '#3d2e09' },
  red:     { bg: '#1a0a0a', color: '#ef4444', border: '#3d1212' },
  emerald: { bg: '#091a11', color: '#22c55e', border: '#12382a' },
  purple:  { bg: '#140d1f', color: '#a78bfa', border: '#2e1a4a' },
  gray:    { bg: '#131415', color: '#6b7280', border: '#27292c' },
  orange:  { bg: '#1a0f03', color: '#fb923c', border: '#3d2409' },
  pink:    { bg: '#1a0a12', color: '#f472b6', border: '#3d1228' },
}

export function StatusList({ items }: StatusListProps): ReactElement {
  return (
    <div className="flex flex-col gap-2.5">
      {items.map((item) => {
        const c = badgeColors[item.tone]
        return (
          <article
            key={item.id}
            style={{ backgroundColor: '#0d0e10', border: '1px solid #1f2023', borderRadius: 8, padding: '12px 14px' }}
            className="flex items-center gap-3"
          >
            {/* Badge código de línea */}
            <div
              style={{
                width: 40, height: 40, borderRadius: 7, flexShrink: 0,
                backgroundColor: c.bg, border: `1px solid ${c.border}`,
                color: c.color, fontSize: '0.7rem', fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                letterSpacing: '0.06em',
              }}
            >
              {item.code}
            </div>
            <div>
              <p style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e2e2e2' }}>{item.name}</p>
              <p style={{ fontSize: '0.73rem', color: '#6b7280', marginTop: 2 }}>{item.detail}</p>
            </div>
          </article>
        )
      })}
    </div>
  )
}