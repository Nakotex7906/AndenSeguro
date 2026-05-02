import { useState, useRef, useEffect, type ReactElement, type MouseEvent } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { useIncidentAlerts } from '../hooks/useIncidentAlerts'
import { useLiveCameraOverview } from '../hooks/useLiveCameraOverview'

type Point = { x: number; y: number }
type CameraStats = { total_persons: number; risk_persons: number; danger_persons: number }

/**
 * Renderiza la vista de monitoreo de cámara con la alerta activa.
 * @returns La pantalla de videovigilancia operativa.
 */
export function LiveCameraPage(): ReactElement | null {
  const { error: alertError } = useIncidentAlerts()
  const { error: cameraError } = useLiveCameraOverview()

  // Estados para configuración de zonas
  const [isConfiguring, setIsConfiguring] = useState(false)
  const [configMode, setConfigMode] = useState<'YELLOW' | 'RED' | 'DONE'>('YELLOW')
  const [yellowPoints, setYellowPoints] = useState<Point[]>([])
  const [redPoints, setRedPoints] = useState<Point[]>([])
  
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
        const response = await fetch('http://localhost:8000/api/stream/stats')
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        }
      } catch (error) {
        // Silenciamos el error para no llenar la consola en caso de caída temporal del backend
      }
    }
    const intervalId = setInterval(fetchStats, 1000)
    return () => clearInterval(intervalId)
  }, [])
  
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

  const handleNextStep = () => {
    if (configMode === 'YELLOW' && yellowPoints.length >= 3) {
      setConfigMode('RED')
    } else if (configMode === 'RED' && redPoints.length >= 3) {
      setConfigMode('DONE')
    }
  }

  const handleSaveConfig = async () => {
    try {
      await fetch('http://localhost:8000/api/stream/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          yellow_points: yellowPoints.map(p => [p.x, p.y]),
          red_points: redPoints.map(p => [p.x, p.y])
        })
      })
      setIsConfiguring(false)
      setConfigMode('YELLOW')
      // Opcional: mostrar notificación de éxito
    } catch (err) {
      console.error('Error guardando configuración', err)
    }
  }

  const handleCancelConfig = () => {
    setIsConfiguring(false)
    setConfigMode('YELLOW')
    setYellowPoints([])
    setRedPoints([])
  }

  // Utilidad para dibujar polígonos SVG con porcentajes (0 a 1 -> 0 a 100%)
  const createSvgPolygon = (points: Point[]) => {
    if (points.length === 0) return ''
    return points.map(p => `${p.x * 100},${p.y * 100}`).join(' ')
  }

  return alertError || cameraError ? (
    <section className="surface-panel p-6">
      <h1 className="text-2xl font-semibold text-white">Vista Cámara 1: Plataforma</h1>
      <p className="mt-3 text-slate-300">{alertError ?? cameraError}</p>
    </section>
  ) : (
    <section className="space-y-5">
      {/* TODO: Aquí se debería llamar al backend para obtener el feed en vivo, detecciones, alertas y recursos asociados. */}
      <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">
            Vista Cámara 1: Plataforma
          </h1>
          <p className="mt-2 text-lg text-slate-300">
            Monitorizando cámara principal en tiempo real.
          </p>
        </div>

        <div className={`surface-panel-strong rounded-2xl border ${stats.danger_persons > 0 ? 'border-red-500/80 bg-red-950/40 shadow-[0_0_15px_rgba(239,68,68,0.4)]' : stats.risk_persons > 0 ? 'border-yellow-500/80 bg-yellow-950/40' : 'border-sky-500/30'} px-6 py-4 text-right transition-all duration-300`}>
          <p className={`text-[0.65rem] font-semibold tracking-[0.22em] uppercase ${stats.danger_persons > 0 ? 'text-red-400' : stats.risk_persons > 0 ? 'text-yellow-400' : 'text-slate-400'}`}>
            Estado de Operación
          </p>
          <div className="flex items-center justify-end gap-3 mt-1.5">
            {stats.danger_persons > 0 && <span className="relative flex h-3 w-3"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span><span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span></span>}
            <p className={`text-lg font-bold tracking-wide ${stats.danger_persons > 0 ? 'text-red-500' : stats.risk_persons > 0 ? 'text-yellow-500' : 'text-emerald-400'}`}>
              {stats.danger_persons > 0 ? 'ALERTA CRÍTICA' : stats.risk_persons > 0 ? 'SOBREVIGILANCIA' : 'NORMAL'}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(22rem,0.8fr)]">
        <div className="space-y-5">
          <Panel
            title="VideoFeed"
            description={isConfiguring ? "Modo configuración: Haz clic en la imagen para dibujar" : "Feed de video en vivo de la estación"}
            className="surface-panel p-5 relative"
          >
            {/* Contenedor centralizado con aspect-video para alinear coordenadas visuales y de cámara */}
            <div className="relative flex w-full aspect-video items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black text-center text-slate-300 group max-h-[720px] mx-auto">
              <img
                src="http://localhost:8000/api/stream/video_feed"
                alt="Stream en vivo"
                className={`w-full h-full object-fill select-none pointer-events-none ${isConfiguring ? 'opacity-60' : ''}`}
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.alt = "Feed no disponible. Revisa la conexión al backend.";
                  target.className = "text-red-400 p-8";
                }}
              />
              
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
                      />
                    )}
                    {redPoints.length > 0 && (
                      <polygon 
                        points={createSvgPolygon(redPoints)} 
                        fill="rgba(255, 0, 0, 0.2)" 
                        stroke="#FF0000" 
                        strokeWidth="0.5" 
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

              {/* Interfaz de configuración superpuesta */}
              {isConfiguring ? (
                <div className="absolute bottom-4 inset-x-4 flex items-center justify-between bg-slate-900/90 backdrop-blur-md border border-white/10 rounded-xl p-4 shadow-xl z-20">
                  <div>
                    <h3 className="text-white font-semibold text-sm">
                      {configMode === 'YELLOW' && "Dibujando Zona de Precaución (Amarilla)"}
                      {configMode === 'RED' && "Dibujando Zona de Peligro (Roja)"}
                      {configMode === 'DONE' && "¡Zonas configuradas! Listo para guardar."}
                    </h3>
                    <p className="text-xs text-slate-300 mt-1">
                      {configMode === 'YELLOW' && "Define al menos 3 puntos para enmarcar el polígono en la imagen."}
                      {configMode === 'RED' && "Define al menos 3 puntos para abarcar el área roja."}
                      {configMode === 'DONE' && "Revisa que los indicadores esten correctos."}
                    </p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <button 
                      onClick={handleCancelConfig}
                      className="px-3 py-1.5 text-sm font-medium text-slate-300 hover:text-white hover:bg-white/10 rounded-lg transition"
                    >
                      Cancelar
                    </button>
                    {configMode !== 'DONE' ? (
                      <button 
                        onClick={handleNextStep}
                        disabled={
                          (configMode === 'YELLOW' && yellowPoints.length < 3) || 
                          (configMode === 'RED' && redPoints.length < 3)
                        }
                        className="px-4 py-1.5 text-sm font-medium bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:hover:bg-sky-600 text-white rounded-lg transition shadow-sm"
                      >
                        Siguiente
                      </button>
                    ) : (
                      <button 
                        onClick={handleSaveConfig}
                        className="px-4 py-1.5 text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition shadow-sm font-semibold"
                      >
                        Guardar Zonas
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-20">
                  <button 
                    onClick={() => {
                      setIsConfiguring(true)
                      setYellowPoints([])
                      setRedPoints([])
                      setConfigMode('YELLOW')
                    }}
                    className="bg-slate-900/80 hover:bg-slate-800 text-white px-3 py-1.5 rounded-lg text-sm font-medium border border-white/10 backdrop-blur-md shadow-lg"
                  >
                    Configurar Zonas
                  </button>
                </div>
              )}
            </div>
          </Panel>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4 flex flex-col justify-center">
              <p className="text-sm text-slate-400">Personas totales en vista</p>
              <p className="mt-2 text-3xl font-semibold text-white">{stats.total_persons}</p>
            </div>
            <div className={`rounded-2xl border ${stats.risk_persons > 0 ? 'border-yellow-500/50 bg-yellow-500/10' : 'border-white/6 bg-slate-950/65'} p-4 flex flex-col justify-center transition-colors`}>
              <p className={`text-sm ${stats.risk_persons > 0 ? 'text-yellow-400' : 'text-slate-400'}`}>Precaución / Merodeando</p>
              <p className={`mt-2 text-3xl font-semibold ${stats.risk_persons > 0 ? 'text-yellow-400' : 'text-white'}`}>{stats.risk_persons}</p>
            </div>
            <div className={`rounded-2xl border ${stats.danger_persons > 0 ? 'border-red-500/50 bg-red-500/20' : 'border-white/6 bg-slate-950/65'} p-4 flex flex-col justify-center transition-colors`}>
              <p className={`text-sm ${stats.danger_persons > 0 ? 'text-red-400 font-semibold uppercase tracking-wider' : 'text-slate-400'}`}>Personas en Zona Roja</p>
              <p className={`mt-2 text-3xl font-semibold ${stats.danger_persons > 0 ? 'text-red-500 animate-pulse' : 'text-white'}`}>{stats.danger_persons}</p>
            </div>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <Panel title="Descripción" className="surface-panel p-5">
              {/* TODO: Este texto debería venir de la respuesta del backend con la descripción física de la persona detectada. */}
              <p className="text-base leading-7 text-slate-300">
                Se debería mostrar la descripción procedente del backend.
              </p>
            </Panel>

            <Panel title="Ubicación" className="surface-panel p-5">
              {/* TODO: Este dato debe llegar desde el backend como ubicación estimada del incidente. */}
              <p className="text-base leading-7 text-slate-300">
                Se debería mostrar la ubicación estimada procedente del backend.
              </p>
            </Panel>
          </div>
        </div>

        <Panel
          title="Acciones de protocolo"
          description="Siguiente paso operativo sugerido para el incidente"
          className="surface-panel p-5"
        >
          {/* TODO: Aquí el backend debería indicar las acciones recomendadas y el estado de la alerta. */}
          <div className="flex flex-col gap-3">
            <button
              className="flex items-center justify-between gap-3 rounded-xl border border-red-500/30 bg-red-600/90 px-4 py-4 text-left font-semibold text-white transition hover:bg-red-500"
              type="button"
            >
              <span>Acción pendiente de backend</span>
            </button>

            <button
              className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/6 px-4 py-4 text-left font-semibold text-slate-100 transition hover:border-slate-400/30 hover:bg-white/10"
              type="button"
            >
              <span>Acción secundaria pendiente de backend</span>
            </button>
          </div>

          <div className="mt-6">
            <h3 className="text-sm font-semibold tracking-[0.16em] text-slate-200 uppercase">
              Redes de apoyo disponibles
            </h3>

            {/* TODO: Este listado debe venir desde el backend con los recursos de apoyo disponibles. */}
            <div className="mt-3 space-y-3">
              <article className="flex items-center justify-between rounded-xl border border-white/6 bg-slate-950/60 px-4 py-3">
                <div>
                  <p className="font-medium text-white">Recursos de apoyo</p>
                  <p className="text-sm text-slate-400">Pendiente de backend</p>
                </div>
                <span className="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-semibold tracking-[0.14em] text-slate-200 uppercase">
                  Integración pendiente
                </span>
              </article>
            </div>
          </div>

          <div className="mt-6 rounded-2xl border border-white/6 bg-slate-950/60 p-4">
            <h3 className="text-sm font-semibold tracking-[0.16em] text-slate-200 uppercase">
              Cámaras relacionadas
            </h3>
            {/* TODO: El backend debe devolver cámaras relacionadas y acciones por cada una. */}
            <div className="mt-3 grid gap-3">
              <div className="flex items-center justify-between rounded-xl border border-white/6 bg-white/5 px-4 py-3">
                <div>
                  <p className="text-sm font-medium text-slate-100">Cámaras relacionadas</p>
                  <p className="text-xs tracking-[0.14em] text-slate-500 uppercase">
                    Pendiente de backend
                  </p>
                </div>
                <button
                  className="rounded-full border border-white/10 bg-slate-900 px-3 py-2 text-xs font-semibold text-slate-200"
                  type="button"
                >
                  Ver
                </button>
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </section>
  )
}
