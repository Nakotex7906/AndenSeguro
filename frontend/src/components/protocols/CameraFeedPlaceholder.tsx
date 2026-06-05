import type { ReactElement } from 'react'

interface Props {
  label: string
}

/**
 * Placeholder del feed de cámara en la vista de protocolos.
 * Se reemplaza con el stream real cuando el backend esté conectado.
 *
 * TODO (backend): sustituir por <img src={`/api/cameras/${cameraId}/stream`} />
 * o un componente de streaming MJPEG/WebRTC según la integración con Ezviz.
 */
export function CameraFeedPlaceholder({ label }: Props): ReactElement {
  return (
    <div style={{
      backgroundColor: '#0a0b0d',
      border: '1px solid #1f2023',
      borderRadius: 8,
      aspectRatio: '16/9',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 8,
      position: 'relative',
    }}>
      {/* Badge de cámara */}
      <div style={{
        position: 'absolute', top: 10, left: 10,
        backgroundColor: 'rgba(0,0,0,0.6)',
        border: '1px solid #2a2d31',
        borderRadius: 5,
        padding: '3px 8px',
        fontSize: '0.6rem',
        fontWeight: 600,
        color: '#9ca3af',
        letterSpacing: '0.1em',
        display: 'flex', alignItems: 'center', gap: 5,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#22c55e', display: 'inline-block' }} />
        {label}
      </div>

      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M23 7l-7 5 7 5V7z" stroke="#2a2d31" strokeWidth="1.4" strokeLinejoin="round"/>
        <rect x="1" y="5" width="15" height="14" rx="2" stroke="#2a2d31" strokeWidth="1.4"/>
      </svg>
      <p style={{ fontSize: '0.65rem', color: '#2a2d31', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
        Feed no disponible
      </p>
    </div>
  )
}