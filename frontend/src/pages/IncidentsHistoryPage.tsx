import { ClockCounterClockwiseIcon, EyeIcon } from '@phosphor-icons/react'
import { useState, useEffect, type ReactElement } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { Button } from '../components/ui/Button'
import type { ViewId } from '../types/dashboard'

interface IncidentResponse {
  id: number
  camera_id: number
  alert_level: string
  description: string
  timestamp: string
  duration_seconds: number | null
  status: string
  resolved_by: number | null
  resolved_at: string | null
}

interface IncidentListResponse {
  items: IncidentResponse[]
  total: number
  page: number
  size: number
}

export function IncidentsHistoryPage({ onViewChange }: { onViewChange?: (view: ViewId) => void }): ReactElement {
  const [incidents, setIncidents] = useState<IncidentResponse[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [page, setPage] = useState<number>(1)
  const [statusFilter, setStatusFilter] = useState<string>('')
  
  const fetchIncidents = async () => {
    setLoading(true)
    try {
      const url = new URL('http://localhost:8000/api/incidents')
      url.searchParams.append('page', page.toString())
      if (statusFilter) url.searchParams.append('status', statusFilter)
      
      const res = await fetch(url.toString(), {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
        }
      })
      if (res.ok) {
        const data: IncidentListResponse = await res.json()
        setIncidents(data.items)
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchIncidents()
  }, [page, statusFilter])

  const getStatusLabel = (status: string) => {
    switch(status) {
      case 'pending': return <span className="text-orange-500 font-bold bg-orange-500/10 px-2 py-1 rounded">Pendiente</span>
      case 'acknowledged': return <span className="text-blue-500 font-bold bg-blue-500/10 px-2 py-1 rounded">Atendido</span>
      case 'resolved': return <span className="text-emerald-500 font-bold bg-emerald-500/10 px-2 py-1 rounded">Resuelto</span>
      case 'false_alarm': return <span className="text-zinc-500 font-bold bg-zinc-500/10 px-2 py-1 rounded">Falsa Alarma</span>
      default: return <span className="text-gray-500 font-bold bg-gray-500/10 px-2 py-1 rounded">{status}</span>
    }
  }

  return (
    <section className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 style={{ fontSize: '1.9rem', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.02em' }} className="flex items-center gap-2">
            <ClockCounterClockwiseIcon size={28} color="#9ca3af" />
            Historial de Emergencias
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '0.85rem' }} className="mt-1">
            Registro inmutable de casos reportados y gestionados
          </p>
        </div>
      </div>

      <Panel title="Historial de Emergencias">
        {/* Toolbar / Filters */}
        <div className="flex gap-3 mb-4">
          <select 
            value={statusFilter} 
            onChange={e => { setStatusFilter(e.target.value); setPage(1); }}
            style={{ backgroundColor: '#161719', border: '1px solid #242628', color: '#e2e2e2', padding: '6px 12px', borderRadius: 6, fontSize: '0.8rem' }}
          >
            <option value="">Todos los Estados</option>
            <option value="pending">Pendientes</option>
            <option value="acknowledged">En Atención</option>
            <option value="resolved">Resueltos</option>
            <option value="false_alarm">Falsas Alarmas</option>
          </select>
          <Button variant="outline" size="sm" onClick={() => fetchIncidents()}>Refrescar</Button>
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
              ) : incidents.length === 0 ? (
                <tr><td colSpan={6} className="text-center p-6 text-gray-500 text-sm">No se encontraron incidentes.</td></tr>
              ) : (
                incidents.map((inc) => (
                  <tr key={inc.id} style={{ borderBottom: '1px solid #1a1c1f' }} className="hover:bg-[#161719] transition-colors">
                    <td className="p-3 text-sm text-gray-300 font-mono">#{inc.id}</td>
                    <td className="p-3 text-sm text-gray-300">
                      {new Date(inc.timestamp).toLocaleString('es-CL')}
                    </td>
                    <td className="p-3 text-sm text-gray-300">CAM-0{inc.camera_id}</td>
                    <td className="p-3 text-sm text-gray-300 uppercase font-bold" style={{ color: inc.alert_level === 'red' ? '#ef4444' : '#f59e0b' }}>
                      {inc.alert_level}
                    </td>
                    <td className="p-3 text-sm">
                      {getStatusLabel(inc.status)}
                    </td>
                    <td className="p-3 text-sm">
                      <Button variant="ghost" size="sm" onClick={() => {
                        window.history.pushState({}, '', `/?incidentId=${inc.id}`)
                        if (onViewChange) onViewChange('protocols')
                      }}>
                        <EyeIcon size={16} /> Ver Detalles
                      </Button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between mt-4">
          <Button variant="outline" size="sm" disabled={page === 1} onClick={() => setPage(p => p - 1)}>Anterior</Button>
          <span className="text-xs text-gray-500 uppercase tracking-wider">Página {page}</span>
          <Button variant="outline" size="sm" disabled={incidents.length < 20} onClick={() => setPage(p => p + 1)}>Siguiente</Button>
        </div>
      </Panel>

    </section>
  )
}
