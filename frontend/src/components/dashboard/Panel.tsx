import type { ReactNode, ReactElement } from 'react'

export interface PanelProps {
  action?: ReactNode
  children: ReactNode
  className?: string
  description?: string
  title: string
}

export function Panel({ action, children, className, description, title }: PanelProps): ReactElement {
  return (
    <section
      style={
        className
          ? undefined
          : {
              backgroundColor: '#161719',
              border: '1px solid #242628',
              borderRadius: '10px',
              padding: '20px 22px',
            }
      }
      className={className}
    >
      <div style={{ marginBottom: 16 }} className="flex items-start justify-between gap-4">
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#f0f0f0' }}>{title}</h2>
          {description
            ? <p style={{ fontSize: '0.8rem', color: '#6b7280', marginTop: 3 }}>{description}</p>
            : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      {children}
    </section>
  )
}