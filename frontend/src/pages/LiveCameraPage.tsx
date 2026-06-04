import { BellIcon, SirenIcon, GearIcon, VideoCameraIcon } from '@phosphor-icons/react'
import type { ReactElement } from 'react'

import { MetricCard } from '../components/dashboard/MetricCard'
import { Panel } from '../components/dashboard/Panel'
import { Button } from '../components/ui/Button'
import { useElapsedTimer } from '../hooks/useElapsedTimer'
import type { DashboardMetric } from '../types/dashboard'

/* ── Datos simulados (reemplazar con hooks reales al conectar backend) ── */
const CAMERA_NAME   = 'ESTACION CENTRAL — ANDEN'
const CAMERA_LABEL  = 'CAMARA EN VIVO  •  CAM-01-ANDEN'
const REC_START     = 2532   // segundos grabados al montar

const LINKED_CAMERAS = [
  { id: 'cam-02', label: 'CAM-02-PASILLO',   thumb: null },
  { id: 'cam-03', label: 'CAM-03-ENTRADA',   thumb: null },
  { id: 'cam-04', label: 'CAM-04-ESCALERAS', thumb: null },
  { id: 'cam-05', label: 'CAM-05-BOLETERIA', thumb: null },
]

const METRICS: DashboardMetric[] = [
  { id: 'riesgo',    label: 'Riesgo por hora',    value: '04', unit: '/hrs', caption: 'Información', tone: 'blue'  },
  { id: 'en-riesgo', label: 'Personas en riesgo', value: '01',              caption: 'Precaución',  tone: 'amber' },
  { id: 'zona',     label: 'Personas en la zona', value: '11',              caption: 'Información', tone: 'blue'  },
]

export function LiveCameraPage(): ReactElement {
  const { elapsedTime: recTime } = useElapsedTimer(REC_START)

  return (
    <section className="space-y-4">

      {/* ── Cabecera ── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p style={{ fontSize: '0.62rem', letterSpacing: '0.2em', color: '#4b4f56', fontWeight: 600 }}
             className="mb-1 uppercase flex items-center gap-1.5">
            <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} />
            {CAMERA_LABEL}
          </p>
          <h1 style={{ fontSize: '1.9rem', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.02em' }}>
            {CAMERA_NAME}
          </h1>
        </div>

        {/* Badges: personas + nivel de riesgo */}
        <div className="flex items-center gap-2 shrink-0">
          <div style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
            <p style={{ fontSize: '0.55rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Personas</p>
            <p style={{ fontSize: '1.3rem', fontWeight: 700, color: '#f0f0f0', fontVariantNumeric: 'tabular-nums' }}>11</p>
          </div>
          <div style={{ backgroundColor: '#0e1929', border: '1px solid #1a3451', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
            <p style={{ fontSize: '0.55rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Nivel de riesgo</p>
            <p style={{ fontSize: '1.3rem', fontWeight: 700, color: '#38bdf8' }}>Bajo</p>
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
                 className="flex items-center justify-center">

              {/* Badge REC */}
              <div style={{ position: 'absolute', top: 12, right: 12, zIndex: 10, backgroundColor: 'rgba(0,0,0,0.7)', border: '1px solid #ef444440', borderRadius: 6, padding: '3px 10px', fontSize: '0.65rem', fontWeight: 700, color: '#ef4444', letterSpacing: '0.1em', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} />
                REC {recTime}
              </div>

              {/* Placeholder feed */}
              <div className="flex flex-col items-center gap-3" style={{ color: '#2a2d31' }}>
                <VideoCameraIcon size={40} />
                <p style={{ fontSize: '0.8rem', color: '#4b4f56' }}>
                  Feed de cámara pendiente de integración con backend
                </p>
              </div>
            </div>

            {/* Botones de acción debajo del feed */}
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Button variant="ghost" size="md" fullWidth>
                <GearIcon size={14} />
                Configurar zona
              </Button>
              <Button variant="danger" size="md" fullWidth>
                <SirenIcon size={14} />
                Alerta emergencia
              </Button>
            </div>
          </Panel>

          {/* 3 MetricCards */}
          <div className="grid grid-cols-3 gap-3">
            {METRICS.map((m) => (
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
              <div style={{ backgroundColor: '#1a0808', border: '1px solid #3d1212', borderRadius: 8, padding: '12px 16px' }} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} />
                  <div>
                    <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f87171' }}>Objeto sospechoso detectado en vía</p>
                    <p style={{ fontSize: '0.65rem', color: '#6b7280' }}>Hace 3 minutos • CAM-01-ANDEN</p>
                  </div>
                </div>
                <span style={{ fontSize: '0.6rem', fontWeight: 700, backgroundColor: '#3d1212', color: '#f87171', padding: '2px 6px', borderRadius: 4, textTransform: 'uppercase' }}>
                  Activa
                </span>
              </div>
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
              {LINKED_CAMERAS.length + 1} Cámaras activas
            </span>
          </div>

          {/* Lista de cámaras */}
          <div className="flex flex-col gap-2">
            {LINKED_CAMERAS.map((cam) => (
              <button
                key={cam.id}
                type="button"
                style={{ backgroundColor: '#161719', border: '1px solid #242628', borderRadius: 8,
                         overflow: 'hidden', cursor: 'pointer', textAlign: 'left' }}
                className="transition hover:border-[#3a3d41]"
              >
                {/* Thumbnail placeholder */}
                <div style={{ backgroundColor: '#0a0b0d', height: 80,
                              display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <VideoCameraIcon size={20} color="#2a2d31" />
                </div>
                <p style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.12em',
                            color: '#9ca3af', padding: '6px 10px', textTransform: 'uppercase' }}>
                  {cam.label}
                </p>
              </button>
            ))}
          </div>

        </div>

      </div>
    </section>
  )
}