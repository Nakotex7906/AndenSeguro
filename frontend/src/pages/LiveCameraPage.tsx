import type { ReactElement } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { useIncidentAlerts } from '../hooks/useIncidentAlerts'
import { useLiveCameraOverview } from '../hooks/useLiveCameraOverview'

/**
 * Renderiza la vista de monitoreo de cámara con la alerta activa.
 * @returns La pantalla de videovigilancia operativa.
 */
export function LiveCameraPage(): ReactElement | null {
  const { error: alertError } = useIncidentAlerts()
  const { error: cameraError } = useLiveCameraOverview()

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

        <div className="surface-panel-strong rounded-2xl border-sky-500/30 px-4 py-3 text-right">
          <p className="text-[0.65rem] font-semibold tracking-[0.22em] text-slate-400 uppercase">
            Estado de incidente
          </p>
          <p className="mt-1 text-sm text-slate-200">Pendiente de backend</p>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(22rem,0.8fr)]">
        <div className="space-y-5">
          <Panel
            title="VideoFeed"
            description="Feed de video en vivo de la estación"
            className="surface-panel p-5"
          >
            <div className="flex relative min-h-96 w-full items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black text-center text-slate-300">
              <img
                src="http://localhost:8000/api/stream/video_feed"
                alt="Stream en vivo"
                className="w-full object-cover"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.alt = "Feed no disponible. Revisa la conexión al backend.";
                  target.className = "text-red-400 p-8";
                }}
              />
            </div>
          </Panel>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-sm text-slate-400">Personas en la zona</p>
              <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-sm text-slate-400">Riesgo por hora</p>
              <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-sm text-slate-400">Personas en riesgo</p>
              <p className="mt-2 text-2xl font-semibold text-white">Pendiente de backend</p>
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
