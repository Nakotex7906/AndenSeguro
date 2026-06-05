import { useState } from 'react'
import type { ReactElement, FormEvent } from 'react'
import { UserPlusIcon, UsersIcon } from '@phosphor-icons/react'

import { Panel } from '../components/dashboard/Panel'
import { Button } from '../components/ui/Button'

export function UsersPage(): ReactElement {
  const [name, setName] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('operador')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ text: string, type: 'error' | 'success' } | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setMessage(null)

    if (!name.trim() || !username.trim() || !password.trim()) {
      setMessage({ text: 'Por favor, completa todos los campos.', type: 'error' })
      return
    }

    if (password.length < 8) {
      setMessage({ text: 'La contraseña debe tener al menos 8 caracteres.', type: 'error' })
      return
    }

    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const res = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          username: username,
          full_name: name,
          password: password,
          role: role
        })
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Error al crear usuario')
      }

      setMessage({ text: 'Usuario creado exitosamente.', type: 'success' })
      setName('')
      setUsername('')
      setPassword('')
      setRole('operador')
    } catch (err: any) {
      setMessage({ text: err.message, type: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const inputStyle: React.CSSProperties = {
    width: '100%',
    backgroundColor: '#0d0e10',
    border: '1px solid #242628',
    borderRadius: 8,
    padding: '10px 12px',
    fontSize: '0.82rem',
    color: '#f0f0f0',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.15s',
  }

  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: '0.75rem',
    fontWeight: 600,
    color: '#9ca3af',
    marginBottom: '6px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em'
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-8">
        <div style={{ backgroundColor: '#1c1e21', padding: '10px', borderRadius: '10px', color: '#38bdf8' }}>
          <UsersIcon size={24} weight="duotone" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Gestión de Operadores</h1>
          <p className="text-sm text-slate-400 mt-1">Crea y administra credenciales para el personal de la estación.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Panel Izquierdo: Formulario */}
        <Panel title="Nuevo Usuario" description="Registra un nuevo operador en el sistema">
          <form onSubmit={handleSubmit} className="space-y-4 mt-4">
            {message && (
              <div style={{
                padding: '12px',
                borderRadius: '8px',
                fontSize: '0.8rem',
                backgroundColor: message.type === 'error' ? '#2d0a0a' : '#022c22',
                border: message.type === 'error' ? '1px solid #ef4444' : '1px solid #10b981',
                color: message.type === 'error' ? '#f87171' : '#34d399',
              }}>
                {message.text}
              </div>
            )}

            <div>
              <label style={labelStyle}>Nombre Completo</label>
              <input
                type="text"
                placeholder="Ej: Juan Pérez"
                value={name}
                onChange={e => setName(e.target.value)}
                style={inputStyle}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#3a3d41' }}
                onBlur={(e)  => { e.currentTarget.style.borderColor = '#242628' }}
              />
            </div>

            <div>
              <label style={labelStyle}>Nombre de Usuario</label>
              <input
                type="text"
                placeholder="Ej: admin123"
                value={username}
                onChange={e => setUsername(e.target.value)}
                style={inputStyle}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#3a3d41' }}
                onBlur={(e)  => { e.currentTarget.style.borderColor = '#242628' }}
              />
            </div>

            <div>
              <label style={labelStyle}>Contraseña</label>
              <input
                type="password"
                placeholder="Mínimo 8 caracteres"
                value={password}
                onChange={e => setPassword(e.target.value)}
                style={inputStyle}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#3a3d41' }}
                onBlur={(e)  => { e.currentTarget.style.borderColor = '#242628' }}
              />
            </div>

            <div>
              <label style={labelStyle}>Rol del Sistema</label>
              <select
                value={role}
                onChange={e => setRole(e.target.value)}
                style={{...inputStyle, cursor: 'pointer', appearance: 'none'}}
              >
                <option value="operador">Operador de Monitoreo</option>
                <option value="seguridad">Personal de Seguridad</option>
                <option value="jefe_estacion">Jefe de Estación</option>
                <option value="admin">Administrador (Superadmin)</option>
              </select>
            </div>

            <div className="pt-2">
              <Button type="submit" variant="primary" fullWidth disabled={loading}>
                <UserPlusIcon size={16} />
                {loading ? 'Registrando...' : 'Crear Cuenta'}
              </Button>
            </div>
          </form>
        </Panel>

        {/* Panel Derecho: Info / Instrucciones */}
        <Panel title="Información de Roles" description="Niveles de acceso del sistema">
          <div className="mt-4 space-y-4">
            <div style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '12px' }}>
              <h4 className="text-[0.8rem] font-bold text-slate-200 mb-1">Operador de Monitoreo</h4>
              <p className="text-[0.75rem] text-slate-400">Puede visualizar las cámaras, métricas y gestionar protocolos de incidentes activos.</p>
            </div>
            <div style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '12px' }}>
              <h4 className="text-[0.8rem] font-bold text-slate-200 mb-1">Personal de Seguridad / Jefe Estación</h4>
              <p className="text-[0.75rem] text-slate-400">Además de monitorear, tienen permisos para cerrar incidentes y cambiar la configuración del motor de IA.</p>
            </div>
            <div style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '12px' }}>
              <h4 className="text-[0.8rem] font-bold text-slate-200 mb-1">Administrador (Superadmin)</h4>
              <p className="text-[0.75rem] text-slate-400">Acceso total al sistema, incluyendo esta vista de gestión de usuarios y configuraciones globales.</p>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
