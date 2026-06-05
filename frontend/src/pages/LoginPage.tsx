import { EnvelopeIcon, LockIcon, UserIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import type { ReactElement, FormEvent } from 'react'

export interface LoginPageProps {
  onLogin: () => void
}

/**
 * Vista de autenticación de Andén Seguro.
 * Solo inicio de sesión. El registro está reservado para el superadmin en otra vista.
 */
export function LoginPage({ onLogin }: LoginPageProps): ReactElement {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState<string | null>(null)
  const [loading,  setLoading]  = useState(false)

  async function handleSubmit(e: FormEvent): Promise<void> {
    e.preventDefault()
    setError(null)

    if (!username.trim() || !password.trim()) {
      setError('Completa todos los campos.')
      return
    }

    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      })
      if (!res.ok) throw new Error('Credenciales incorrectas')
      const data = await res.json()
      localStorage.setItem('token', data.access_token)
      
      onLogin()
    } catch {
      setError('Credenciales incorrectas. Intenta nuevamente.')
    } finally {
      setLoading(false)
    }
  }

  const inputStyle: React.CSSProperties = {
    width: '100%',
    backgroundColor: '#0d0e10',
    border: '1px solid #242628',
    borderRadius: 8,
    padding: '11px 12px 11px 36px',
    fontSize: '0.82rem',
    color: '#f0f0f0',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.15s',
  }

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#0d0e10',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
    }}>
      <div style={{
        width: '100%',
        maxWidth: 880,
        backgroundColor: '#111214',
        border: '1px solid #1f2023',
        borderRadius: 16,
        overflow: 'hidden',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        minHeight: 500,
      }}>

        {/* ── Panel izquierdo: formulario ── */}
        <div style={{ padding: '48px 44px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>

          {/* Logo + marca pequeños */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 32 }}>
            <img
              src="/logo.png"
              alt="Andén Seguro"
              style={{ width: 22, height: 22, objectFit: 'contain' }}
              onError={(e) => { e.currentTarget.style.display = 'none' }}
            />
            <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.22em', color: '#4b4f56', textTransform: 'uppercase' }}>
              Anden Seguro
            </span>
          </div>

          {/* Título según modo */}
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f0f0f0', letterSpacing: '-0.02em', marginBottom: 4 }}>
            Iniciar sesión
          </h1>
          <p style={{ fontSize: '0.78rem', color: '#6b7280', marginBottom: 24 }}>
            Accede al sistema de monitoreo de andenes.
          </p>

          {/* Formulario */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

            {/* Usuario */}
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#4b4f56' }}>
                <UserIcon size={15} />
              </span>
              <input
                type="text"
                placeholder="Nombre de usuario"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                style={inputStyle}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#3a3d41' }}
                onBlur={(e)  => { e.currentTarget.style.borderColor = '#242628' }}
              />
            </div>

            {/* Contraseña */}
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#4b4f56' }}>
                <LockIcon size={15} />
              </span>
              <input
                type="password"
                placeholder="Contraseña"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={inputStyle}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#3a3d41' }}
                onBlur={(e)  => { e.currentTarget.style.borderColor = '#242628' }}
              />
            </div>

            {/* Error */}
            {error && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, backgroundColor: '#1a0808', border: '1px solid #3d1212', borderRadius: 7, padding: '8px 12px' }}>
                <WarningCircleIcon size={14} color="#f87171" />
                <span style={{ fontSize: '0.75rem', color: '#f87171' }}>{error}</span>
              </div>
            )}

            {/* Botón principal */}
            <button
              type="submit"
              disabled={loading}
              style={{
                marginTop: 6,
                width: '100%',
                backgroundColor: loading ? '#1c1e21' : '#22c55e',
                border: 'none',
                borderRadius: 8,
                padding: '11px 0',
                fontSize: '0.82rem',
                fontWeight: 700,
                color: loading ? '#6b7280' : '#0a0f0a',
                cursor: loading ? 'not-allowed' : 'pointer',
                letterSpacing: '0.04em',
                transition: 'background-color 0.2s',
              }}
            >
              {loading ? 'Verificando…' : 'Ingresar al sistema'}
            </button>
          </form>
        </div>

        {/* ── Panel derecho: logo y marca ── */}
        <div style={{
          backgroundColor: '#0d0e10',
          borderLeft: '1px solid #1f2023',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 20,
          padding: 48,
        }}>
          {/* Logo grande */}
          <img
            src="/logo.png"
            alt="Andén Seguro"
            style={{ width: 148, height: 148, objectFit: 'contain' }}
            onError={(e) => {
              e.currentTarget.style.display = 'none'
              const fb = document.getElementById('logo-fallback')
              if (fb) fb.style.display = 'flex'
            }}
          />
          {/* Fallback SVG */}
          <div id="logo-fallback" style={{ display: 'none', width: 148, height: 148, alignItems: 'center', justifyContent: 'center' }}>
            <svg viewBox="0 0 80 80" width="148" height="148" aria-hidden="true">
              <path d="M40 4L8 18v22c0 18 13 34 32 38 19-4 32-20 32-38V18L40 4z" fill="none" stroke="#22c55e" strokeWidth="3" strokeLinejoin="round"/>
              <circle cx="40" cy="38" r="14" fill="none" stroke="#22c55e" strokeWidth="2.5"/>
              <rect x="33" y="30" width="14" height="12" rx="2" fill="#22c55e" opacity="0.9"/>
              <circle cx="36.5" cy="44" r="2.5" fill="#22c55e"/>
              <circle cx="43.5" cy="44" r="2.5" fill="#22c55e"/>
            </svg>
          </div>

          {/* Textos */}
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.28em', color: '#4b4f56', textTransform: 'uppercase', marginBottom: 8 }}>
              Sistema de monitoreo
            </p>
            <h2 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#f0f0f0', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              Andén Seguro
            </h2>
          </div>
        </div>

      </div>
    </div>
  )
}