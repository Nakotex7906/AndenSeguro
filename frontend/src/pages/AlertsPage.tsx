import { SirenIcon, CheckCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import { useState, useEffect, useCallback, type ReactElement } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { Button } from '../components/ui/Button'

interface IncidentResponse {
  id: number
  camera_id: number
  alert_level: string
  description: string | null
  timestamp: string
  status: string
  image_url?: string | null
}

export function AlertsPage(): ReactElement {
  const [incidents, setIncidents] = useState<IncidentResponse[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [filter, setFilter] = useState<'active' | 'all'>('active')
  const [resolving, setResolving] = useState<number | null>(null)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const fetchIncidents = useCallback(async () => {
    setLoading(true)
    try {
      const url = new URL('http://localhost:8000/api/incidents')
      url.searchParams.set('page', '1')
      url.searchParams.set('size', '50')

      const res = await fetch(url.toString(), {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token') || ''}` }
      })
      if (res.ok) {
        const data = await res.json()
        const items: IncidentResponse[] = Array.isArray(data)
          ? data
          : (data.items ?? data.incidents ?? data.data ?? [])
        setIncidents(items)
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchIncidents()
    const interval = setInterval(fetchIncidents, 10000)
    return () => clearInterval(interval)
  }, [fetchIncidents])

  const updateStatus = async (id: number, newStatus: string) => {
    setResolving(id)
    try {
      await fetch(`http://localhost:8000/api/incidents/${id}/status`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify({ status: newStatus }),
      })
      setIncidents(prev => prev.map(inc => inc.id === id ? { ...inc, status: newStatus } : inc))
    } catch (err) {
      console.error(err)
    } finally {
      setResolving(null)
    }
  }

  const getLevelLabel = (level: string) => {
    const color = level === 'red' ? '#ef4444' : '#f59e0b'
    const label = level === 'red' ? 'CRÍTICO' : 'ADVERTENCIA'
    return <span className="text-sm font-bold uppercase" style={{ color }}>{label}</span>
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending':      return <span className="text-orange-500 font-bold bg-orange-500/10 px-2 py-1 rounded">Pendiente</span>
      case 'acknowledged': return <span className="text-blue-500 font-bold bg-blue-500/10 px-2 py-1 rounded">En atención</span>
      case 'resolved':     return <span className="text-emerald-500 font-bold bg-emerald-500/10 px-2 py-1 rounded">Resuelto</span>
      case 'false_alarm':  return <span className="text-zinc-500 font-bold bg-zinc-500/10 px-2 py-1 rounded">Falsa Alarma</span>
      default:             return <span className="text-gray-500 font-bold bg-gray-500/10 px-2 py-1 rounded">{status}</span>
    }
  }

  const displayed = filter === 'active'
    ? incidents.filter(i => i.status === 'pending' || i.status === 'acknowledged')
    : incidents

  const activeCount = incidents.filter(i => i.status === 'pending' || i.status === 'acknowledged').length

  return (
    <section className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 style={{ fontSize: '1.9rem', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.02em' }} className="flex items-center gap-2">
            <SirenIcon size={28} color="#ef4444" />
            Alertas Críticas
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '0.85rem' }} className="mt-1">
            Incidentes activos detectados por el sistema de visión IA · Auto-actualiza cada 10s
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => fetchIncidents()}>Refrescar</Button>
      </div>

      {/* Banner de alertas activas */}
      {activeCount > 0 && (
        <div style={{ backgroundColor: '#1a0808', border: '1px solid #5b1717', borderRadius: 8, padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 8 }}>
          <SirenIcon size={16} color="#f87171" weight="fill" />
          <span style={{ fontSize: '0.82rem', color: '#f87171', fontWeight: 600 }}>
            {activeCount} alerta{activeCount > 1 ? 's' : ''} activa{activeCount > 1 ? 's' : ''} sin resolver
          </span>
        </div>
      )}

      <Panel title="Alertas Críticas">
        {/* Toolbar / Filters */}
        <div className="flex gap-3 mb-4">
          <select
            value={filter}
            onChange={e => setFilter(e.target.value as 'active' | 'all')}
            style={{ backgroundColor: '#161719', border: '1px solid #242628', color: '#e2e2e2', padding: '6px 12px', borderRadius: 6, fontSize: '0.8rem' }}
          >
            <option value="active">Activas ({activeCount})</option>
            <option value="all">Todas ({incidents.length})</option>
          </select>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr style={{ borderBottom: '1px solid #242628' }}>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">ID</th>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">Fecha / Hora</th>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">Cámara</th>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">Nivel</th>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">Estado</th>
                <th className="p-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="text-center p-6 text-gray-500 text-sm">Cargando...</td></tr>
              ) : displayed.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center p-8 text-gray-500 text-sm">
                    <CheckCircleIcon size={28} color="#34d399" style={{ margin: '0 auto 8px' }} />
                    <p>{filter === 'active' ? 'Sin alertas activas — el sistema está operativo.' : 'No hay incidentes registrados.'}</p>
                  </td>
                </tr>
              ) : (
                displayed.map(inc => (
                  <>
                    <tr
                      key={inc.id}
                      style={{ borderBottom: '1px solid #1a1c1f', cursor: 'pointer' }}
                      className="hover:bg-[#161719] transition-colors"
                      onClick={() => setExpandedId(expandedId === inc.id ? null : inc.id)}
                    >
                      <td className="p-3 text-sm text-gray-300 font-mono">#{inc.id}</td>
                      <td className="p-3 text-sm text-gray-300">
                        {new Date(inc.timestamp).toLocaleString('es-CL')}
                      </td>
                      <td className="p-3 text-sm text-gray-300">CAM-0{inc.camera_id}</td>
                      <td className="p-3">{getLevelLabel(inc.alert_level)}</td>
                      <td className="p-3 text-sm">{getStatusLabel(inc.status)}</td>
                      <td className="p-3 text-sm" onClick={e => e.stopPropagation()}>
                        {(inc.status === 'pending' || inc.status === 'acknowledged') ? (
                          <div className="flex gap-2">
                            <Button
                              variant="outline" size="sm"
                              disabled={resolving === inc.id}
                              onClick={() => updateStatus(inc.id, 'resolved')}
                            >
                              <CheckCircleIcon size={13} /> Resolver
                            </Button>
                            <Button
                              variant="ghost" size="sm"
                              disabled={resolving === inc.id}
                              onClick={() => updateStatus(inc.id, 'false_alarm')}
                            >
                              <XCircleIcon size={13} /> Falsa
                            </Button>
                          </div>
                        ) : (
                          <span className="text-xs text-gray-600">—</span>
                        )}
                      </td>
                    </tr>

                    {/* Fila expandible con detalle */}
                    {expandedId === inc.id && (
                      <tr key={`${inc.id}-detail`} style={{ backgroundColor: '#0d0e10', borderBottom: '1px solid #1a1c1f' }}>
                        <td colSpan={6} style={{ padding: '14px 20px' }}>
                          <div className="flex gap-5 items-start">
                            {inc.image_url && (
                              <img
                                src={inc.image_url}
                                alt="Captura del incidente"
                                style={{ width: 200, height: 130, objectFit: 'cover', borderRadius: 8, border: '1px solid #242628', flexShrink: 0 }}
                              />
                            )}
                            <div>
                              <p style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.18em', color: '#f87171', textTransform: 'uppercase', marginBottom: 6 }}>
                                Reporte de IA
                              </p>
                              <p style={{ fontSize: '0.82rem', color: '#9ca3af', lineHeight: 1.6 }}>
                                {inc.description ?? 'Sin descripción analítica disponible para este incidente.'}
                              </p>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </section>
  )
}