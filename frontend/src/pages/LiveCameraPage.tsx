import { useState, useRef, useEffect, type ReactElement, type MouseEvent } from 'react'
import { BellIcon, SirenIcon, GearIcon, VideoCameraIcon } from '@phosphor-icons/react'

import { MetricCard } from '../components/dashboard/MetricCard'
import { Panel } from '../components/dashboard/Panel'
import { Button } from '../components/ui/Button'
import { useElapsedTimer } from '../hooks/useElapsedTimer'
import { useIncidentAlerts } from '../hooks/useIncidentAlerts'
import { useLiveCameraOverview } from '../hooks/useLiveCameraOverview'
import type { DashboardMetric, ViewId } from '../types/dashboard'

type Point = { x: number; y: number }
type CameraStats = { total_persons: number; risk_persons: number; danger_persons: number }

const CAMERA_NAME   = 'ESTACION CENTRAL — ANDEN'
const CAMERA_LABEL  = 'CAMARA EN VIVO  •  CAM-01-ANDEN'
const REC_START     = 2532   // segundos grabados al montar

const LINKED_CAMERAS = [
  { id: 1, label: 'CAM-01-ANDEN',     thumb: null },
  { id: 2, label: 'CAM-02-PASILLO',   thumb: null },
  { id: 3, label: 'CAM-03-ENTRADA',   thumb: null },
]

export function LiveCameraPage({ onViewChange }: { onViewChange?: (view: ViewId) => void }): ReactElement | null {
  // Estado para la cámara seleccionada
  const [activeCameraId, setActiveCameraId] = useState<number>(1)

  const { data: alertData, error: alertError } = useIncidentAlerts(activeCameraId)
  const { error: cameraError } = useLiveCameraOverview()
  const { elapsedTime: recTime } = useElapsedTimer(REC_START)

  // Estados para configuración de zonas
  const [isConfiguring, setIsConfiguring] = useState(false)
  const [configMode, setConfigMode] = useState<'YELLOW' | 'RED' | 'DONE'>('YELLOW')
  const [yellowPoints, setYellowPoints] = useState<Point[]>([])
  const [redPoints, setRedPoints] = useState<Point[]>([])
  const originalZonesRef = useRef<{ yellow_points: Point[], red_points: Point[] } | null>(null)
  
  // Estado para estadísticas en tiempo real
  const [stats, setStats] = useState<CameraStats>({
    total_persons: 0,
    risk_persons: 0,
    danger_persons: 0
  })

  // Polling de estadísticas cada 1 segundo
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/stream/stats/${activeCameraId}`)
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        }
      } catch (error) {
        // Silenciamos el error
      }
    }
    const intervalId = setInterval(fetchStats, 1000)
    return () => clearInterval(intervalId)
  }, [activeCameraId])
  
  // Referencia a div para calcular posiciones relativas
  const overlayRef = useRef<HTMLDivElement>(null)

  const handleOverlayClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!isConfiguring || configMode === 'DONE' || !overlayRef.current) return

    // Calcular el porcentaje respecto a las coordenadas del overlay superpuesto
    const rect = overlayRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    if (configMode === 'YELLOW') {
      setYellowPoints((prev) => [...prev, { x, y }])
    } else if (configMode === 'RED') {
      setRedPoints((prev) => [...prev, { x, y }])
    }
  }

  const startConfiguration = async () => {
    setIsConfiguring(true)
    setYellowPoints([])
    setRedPoints([])
    setConfigMode('YELLOW')

    // 1. Obtener configuración actual de la DB para respaldarla
    try {
      const res = await fetch(`http://localhost:8000/api/stream/config/${activeCameraId}`)
      if (res.ok) {
        const data = await res.json()
        originalZonesRef.current = {
          yellow_points: data.yellow_points ? data.yellow_points.map((p: any) => ({ x: p[0], y: p[1] })) : [],
          red_points: data.red_points ? data.red_points.map((p: any) => ({ x: p[0], y: p[1] })) : []
        }
      }
      
      // 2. Limpiar las zonas en la memoria del backend temporalmente para tener una vista limpia
      // Como ahora los endpoints se guardan directamente a la DB, solo limpiaremos la vista (estado local)
      // Y no enviamos POST de limpieza hasta que el usuario decida guardar.
    } catch (err) {
      console.error('Error al iniciar configuración', err)
    }
  }

  const handleNextStep = () => {
    if (configMode === 'YELLOW' && yellowPoints.length >= 3) {
      setConfigMode('RED')
    } else if (configMode === 'RED' && redPoints.length >= 3) {
      setConfigMode('DONE')
    }
  }

  const handleSaveConfig = async () => {
    try {
      await fetch(`http://localhost:8000/api/stream/config/${activeCameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          yellow_points: yellowPoints.map(p => [p.x, p.y]),
          red_points: redPoints.map(p => [p.x, p.y])
        })
      })
      setIsConfiguring(false)
      setConfigMode('YELLOW')
      originalZonesRef.current = null
    } catch (err) {
      console.error('Error guardando configuración', err)
    }
  }

  const handleCancelConfig = async () => {
    // Restaurar el estado local de configuración a null
    setIsConfiguring(false)
    setConfigMode('YELLOW')
    originalZonesRef.current = null
  }

  // Utilidad para dibujar polígonos SVG con porcentajes (0 a 1 -> 0 a 100%)
  const createSvgPolygon = (points: Point[]) => {
    if (points.length === 0) return ''
    return points.map(p => `${p.x * 100},${p.y * 100}`).join(' ')
  }

  const metrics: DashboardMetric[] = [
    { 
      id: 'zona', 
      label: 'Personas en la zona', 
      value: String(stats.total_persons).padStart(2, '0'), 
      caption: 'En rango de cámara', 
      tone: 'blue' 
    },
    { 
      id: 'en-riesgo', 
      label: 'Precaución / Merodeo', 
      value: String(stats.risk_persons).padStart(2, '0'), 
      caption: 'Zona amarilla', 
      tone: stats.risk_persons > 0 ? 'amber' : 'slate' 
    },
    { 
      id: 'peligro', 
      label: 'En zona de riesgo', 
      value: String(stats.danger_persons).padStart(2, '0'), 
      caption: 'Zona roja de peligro', 
      tone: stats.danger_persons > 0 ? 'red' : 'slate' 
    },
  ]

  if (alertError || cameraError) {
    return (
      <section className="surface-panel p-6">
        <h1 className="text-2xl font-semibold text-white">Vista Cámara 1: Plataforma</h1>
        <p className="mt-3 text-slate-300">{alertError ?? cameraError}</p>
      </section>
    )
  }

  return (
    <section className="space-y-4">

      {/* ── Cabecera ── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p style={{ fontSize: '0.62rem', letterSpacing: '0.2em', color: '#4b4f56', fontWeight: 600 }}
             className="mb-1 uppercase flex items-center gap-1.5">
            <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} />
            CAMARA EN VIVO • {LINKED_CAMERAS.find(c => c.id === activeCameraId)?.label || 'CAMARA'}
          </p>
          <h1 style={{ fontSize: '1.9rem', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.02em' }}>
            ESTACIÓN CENTRAL — {LINKED_CAMERAS.find(c => c.id === activeCameraId)?.label.split('-')[2] || 'ANDEN'}
          </h1>
        </div>

        {/* Badges: personas + nivel de riesgo */}
        <div className="flex items-center gap-2 shrink-0">
          <div style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
            <p style={{ fontSize: '0.55rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Personas</p>
            <p style={{ fontSize: '1.3rem', fontWeight: 700, color: '#f0f0f0', fontVariantNumeric: 'tabular-nums' }}>
              {stats.total_persons}
            </p>
          </div>
          <div style={{ 
            backgroundColor: stats.danger_persons > 0 ? '#2d0a0a' : stats.risk_persons > 0 ? '#261a05' : '#0e1929', 
            border: stats.danger_persons > 0 ? '1px solid #5a1414' : stats.risk_persons > 0 ? '1px solid #4d330b' : '1px solid #1a3451', 
            borderRadius: 8, 
            padding: '8px 16px', 
            textAlign: 'center' 
          }}>
            <p style={{ fontSize: '0.55rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Nivel de riesgo</p>
            <p style={{ 
              fontSize: '1.3rem', 
              fontWeight: 700, 
              color: stats.danger_persons > 0 ? '#f87171' : stats.risk_persons > 0 ? '#fbbf24' : '#38bdf8' 
            }}>
              {stats.danger_persons > 0 ? 'Alto' : stats.risk_persons > 0 ? 'Moderado' : 'Bajo'}
            </p>
          </div>
        </div>
      </div>

      {/* ── Layout principal ── */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(14rem,0.7fr)]">

        {/* ── Columna izquierda: feed + botones + métricas ── */}
        <div className="flex flex-col gap-4">

          {/* Feed de cámara */}
          <Panel title="" description="">
            <div style={{ position: 'relative', backgroundColor: '#0a0b0d', borderRadius: 8, overflow: 'hidden', minHeight: 320 }}
                 className="flex items-center justify-center aspect-video max-h-[540px] mx-auto w-full">

              {/* Badge REC */}
              <div style={{ position: 'absolute', top: 12, right: 12, zIndex: 10, backgroundColor: 'rgba(0,0,0,0.7)', border: '1px solid #ef444440', borderRadius: 6, padding: '3px 10px', fontSize: '0.65rem', fontWeight: 700, color: '#ef4444', letterSpacing: '0.1em', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                </span>
                REC {recTime}
              </div>

              {/* Video feed image */}
              <img
                key={activeCameraId}
                src={`http://localhost:8000/api/stream/video_feed/${activeCameraId}`}
                alt="Feed en vivo"
                className="w-full h-full object-fill select-none pointer-events-none"
                style={{ opacity: isConfiguring ? 0.6 : 1 }}
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  const placeholder = document.getElementById('camera-placeholder');
                  if (placeholder) {
                    placeholder.style.display = 'flex';
                  }
                }}
              />

              {/* Placeholder fallback for when image load fails */}
              <div 
                id="camera-placeholder"
                className="absolute inset-0 flex flex-col items-center justify-center gap-3" 
                style={{ color: '#2a2d31', display: 'none' }}
              >
                <VideoCameraIcon size={40} />
                <p style={{ fontSize: '0.8rem', color: '#ef4444' }}>
                  Feed no disponible. Revisa la conexión al backend.
                </p>
              </div>

              {/* Capa de interacción y dibujo SVG (100% del contenedor) */}
              <div
                ref={overlayRef}
                className="absolute inset-0 z-10 w-full h-full"
                onClick={handleOverlayClick}
                style={{ cursor: isConfiguring ? 'crosshair' : 'default' }}
              >
                {isConfiguring && (
                  <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Relleno translúcido de ayuda y bordes */}
                    {yellowPoints.length > 0 && (
                      <polygon 
                        points={createSvgPolygon(yellowPoints)} 
                        fill="rgba(255, 204, 0, 0.2)" 
                        stroke="#FFCC00" 
                        strokeWidth="0.5" 
                        strokeDasharray={configMode === 'YELLOW' ? "1 1" : "0"}
                      />
                    )}
                    {redPoints.length > 0 && (
                      <polygon 
                        points={createSvgPolygon(redPoints)} 
                        fill="rgba(255, 0, 0, 0.2)" 
                        stroke="#FF0000" 
                        strokeWidth="0.5" 
                        strokeDasharray={configMode === 'RED' ? "1 1" : "0"}
                      />
                    )}
                    
                    {/* Dibujar los puntos seleccionados para mejor UX */}
                    {yellowPoints.map((p, i) => (
                      <circle key={`yp-${i}`} cx={p.x * 100} cy={p.y * 100} r="0.8" fill="#FFCC00" />
                    ))}
                    {redPoints.map((p, i) => (
                      <circle key={`rp-${i}`} cx={p.x * 100} cy={p.y * 100} r="0.8" fill="#FF0000" />
                    ))}
                  </svg>
                )}
              </div>
            </div>

            {/* Botones de acción o panel de configuración debajo del feed */}
            {isConfiguring ? (
              <div className="mt-3 flex flex-col sm:flex-row items-center justify-between" style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '16px', gap: '16px' }}>
                <div className="text-left flex-1">
                  <h3 style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f0f0f0' }}>
                    {configMode === 'YELLOW' && "Dibujando Zona de Precaución (Amarilla)"}
                    {configMode === 'RED' && "Dibujando Zona de Peligro (Roja)"}
                    {configMode === 'DONE' && "¡Zonas configuradas! Listo para guardar."}
                  </h3>
                  <p style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: 4 }}>
                    {configMode === 'YELLOW' && "Dibuja un polígono de cualquier forma agregando múltiples puntos. Presiona 'Siguiente' al finalizar."}
                    {configMode === 'RED' && "Dibuja el área roja agregando múltiples puntos. Presiona 'Siguiente' al finalizar."}
                    {configMode === 'DONE' && "Revisa que los indicadores estén correctos antes de guardar."}
                  </p>
                </div>
                <div className="flex gap-2 shrink-0">
                  <Button variant="ghost" size="md" onClick={handleCancelConfig}>
                    Cancelar
                  </Button>
                  {configMode !== 'DONE' ? (
                    <Button 
                      variant="primary" 
                      size="md"
                      onClick={handleNextStep}
                      disabled={
                        (configMode === 'YELLOW' && yellowPoints.length < 3) || 
                        (configMode === 'RED' && redPoints.length < 3)
                      }
                    >
                      Siguiente
                    </Button>
                  ) : (
                    <Button 
                      variant="primary" 
                      size="md"
                      onClick={handleSaveConfig}
                      style={{ backgroundColor: '#10b981', borderColor: '#059669', color: 'white' }}
                    >
                      Guardar Zonas
                    </Button>
                  )}
                </div>
              </div>
            ) : (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Button 
                  variant="ghost" 
                  size="md" 
                  fullWidth 
                  onClick={startConfiguration}
                >
                  <GearIcon size={14} />
                  Configurar zonas
                </Button>
                <Button 
                  variant="danger" 
                  size="md" 
                  fullWidth
                  onClick={async () => {
                    try {
                      // Usar el token del storage si usamos auth, acá asumo que pasa derecho si es dev
                      const res = await fetch('http://localhost:8000/api/incidents', {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
                        },
                        body: JSON.stringify({ camera_id: activeCameraId })
                      })
                      if (res.ok) {
                        const incident = await res.json()
                        window.history.pushState({}, '', `/?incidentId=${incident.id}`)
                        if (onViewChange) onViewChange('protocols')
                      } else {
                        console.error('Failed to create manual incident', await res.text())
                      }
                    } catch (err) {
                      console.error('Error on manual alert:', err)
                    }
                  }}
                >
                  <SirenIcon size={14} />
                  Alerta emergencia
                </Button>
              </div>
            )}
          </Panel>

          {/* 3 MetricCards */}
          <div className="grid grid-cols-3 gap-3">
            {metrics.map((m) => (
              <MetricCard key={m.id} metric={m} />
            ))}
          </div>

          {/* ── Card de Alertas Críticas ── */}
          <Panel title="" description="">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 pb-2" style={{ borderBottom: '1px solid #242628' }}>
                <BellIcon size={18} color="#ef4444" />
                <h3 style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', color: '#f0f0f0', textTransform: 'uppercase' }}>
                  Alertas Críticas de la Estación
                </h3>
              </div>
              {alertData ? (
                <div style={{ 
                  backgroundColor: alertData.title === 'PELIGRO INMINENTE' ? '#2d0a0a' : '#261a05', 
                  border: alertData.title === 'PELIGRO INMINENTE' ? '1px solid #ef4444' : '1px solid #f59e0b', 
                  borderRadius: 8, 
                  padding: '12px 16px' 
                }} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span style={{ 
                      width: 8, 
                      height: 8, 
                      borderRadius: '50%', 
                      backgroundColor: alertData.title === 'PELIGRO INMINENTE' ? '#ef4444' : '#f59e0b', 
                      display: 'inline-block',
                      animation: 'pulse 1s infinite'
                    }} />
                    <div>
                      <p style={{ fontSize: '0.8rem', fontWeight: 600, color: alertData.title === 'PELIGRO INMINENTE' ? '#f87171' : '#fbbf24' }}>
                        {alertData.description}
                      </p>
                      <p style={{ fontSize: '0.65rem', color: '#6b7280' }}>
                        Hace {alertData.elapsedSeconds}s • {alertData.cameraLabel}
                      </p>
                    </div>
                  </div>
                  <span style={{ 
                    fontSize: '0.6rem', 
                    fontWeight: 700, 
                    backgroundColor: alertData.title === 'PELIGRO INMINENTE' ? '#ef4444' : '#f59e0b', 
                    color: alertData.title === 'PELIGRO INMINENTE' ? '#ffffff' : '#000000', 
                    padding: '2px 6px', 
                    borderRadius: 4, 
                    textTransform: 'uppercase' 
                  }}>
                    {alertData.title}
                  </span>
                </div>
              ) : (
                <div style={{ 
                  backgroundColor: '#161719', 
                  border: '1px solid #242628', 
                  borderRadius: 8, 
                  padding: '12px 16px',
                  display: 'flex',
                  justifyContent: 'center'
                }}>
                  <p style={{ fontSize: '0.75rem', color: '#6b7280' }}>No hay alertas activas en esta cámara.</p>
                </div>
              )}
            </div>
          </Panel>

        </div>

        {/* ── Columna derecha: cámaras vinculadas ── */}
        <div className="flex flex-col gap-4">

          {/* Header cámaras vinculadas */}
          <div className="flex items-center justify-between">
            <p style={{ fontSize: '0.62rem', letterSpacing: '0.2em', color: '#4b4f56', fontWeight: 600 }}
               className="uppercase">
              Cámaras vinculadas
            </p>
            <span style={{ fontSize: '0.62rem', letterSpacing: '0.14em', fontWeight: 700,
                           backgroundColor: '#0e1929', color: '#38bdf8',
                           border: '1px solid #1a3451', borderRadius: 4,
                           padding: '2px 8px' }}
                  className="uppercase">
              {LINKED_CAMERAS.length} Cámaras en total
            </span>
          </div>

          {/* Lista de cámaras */}
          <div className="flex flex-col gap-2">
            {LINKED_CAMERAS.map((cam) => {
              const isActive = cam.id === activeCameraId;
              return (
                <button
                  key={cam.id}
                  type="button"
                  onClick={() => setActiveCameraId(cam.id)}
                  style={{ backgroundColor: isActive ? '#1c1e21' : '#161719', border: isActive ? '1px solid #38bdf8' : '1px solid #242628', borderRadius: 8,
                           overflow: 'hidden', cursor: 'pointer', textAlign: 'left' }}
                  className="transition hover:border-[#3a3d41]"
                >
                  {/* Thumbnail placeholder */}
                  <div style={{ backgroundColor: '#0a0b0d', height: 80,
                                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <VideoCameraIcon size={20} color={isActive ? "#38bdf8" : "#2a2d31"} />
                  </div>
                  <p style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.12em',
                              color: isActive ? '#38bdf8' : '#9ca3af', padding: '6px 10px', textTransform: 'uppercase' }}>
                    {cam.label}
                  </p>
                </button>
              )
            })}
          </div>

        </div>

      </div>
    </section>
  )
}