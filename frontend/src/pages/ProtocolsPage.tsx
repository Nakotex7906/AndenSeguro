import { ClipboardTextIcon, FileXIcon, VideoCameraIcon } from '@phosphor-icons/react'
import type { ReactElement } from 'react'

import { Panel } from '../components/dashboard/Panel'
import { AlertSignalSelector } from '../components/protocols/AlertSignalSelector'
import { NotesLog } from '../components/protocols/NotesLog'
import { ProtocolSteps } from '../components/protocols/ProtocolSteps'
import { ResponseChannels } from '../components/protocols/ResponseChannels'
import { RiskLevelSelector } from '../components/protocols/RiskLevelSelector'
import { Button } from '../components/ui/Button'
import { useActiveProtocol } from '../hooks/useActiveProtocol'
import { useElapsedTimer } from '../hooks/useElapsedTimer'

/**
 * Vista de Protocolos — fiel al diseño Figma.
 * Layout: cabecera de incidente + columna izquierda (pasos + notas)
 * y columna derecha (cámara + canales + riesgo + señales + acciones de cierre).
 *
 * Componentes reutilizados: Panel, Button, useElapsedTimer.
 * Componentes nuevos: RiskLevelSelector, AlertSignalSelector,
 *   ProtocolSteps, ResponseChannels, NotesLog, CameraFeedPlaceholder.
 */
export function ProtocolsPage(): ReactElement {
  const searchParams = new URLSearchParams(window.location.search)
  const incidentIdParam = searchParams.get('incidentId')
  const incidentId = incidentIdParam ? parseInt(incidentIdParam, 10) : 1

  const {
    protocol,
    toggleStep,
    setRiskLevel,
    toggleSignal,
    addNote,
    callChannel,
    generateDerivationSheet,
    registerRejection,
  } = useActiveProtocol(incidentId)

  const { elapsedTime } = useElapsedTimer(protocol.elapsedSeconds)

  return (
    <section className="space-y-4">

      {/* ── Cabecera de incidente — fiel al Figma ── */}
      <div style={{
        backgroundColor: '#161719',
        border: '1px solid #242628',
        borderRadius: 10,
        overflow: 'hidden',
      }}>
        {/* Banda superior roja — estado activo */}
        <div style={{ backgroundColor: '#1a0808', borderBottom: '1px solid #3d1212', padding: '6px 20px', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', backgroundColor: '#f87171', display: 'inline-block' }} />
          <span style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.18em', color: '#f87171', textTransform: 'uppercase' }}>
            {protocol.incidentLabel}
          </span>
        </div>

        {/* Fila principal */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 0, alignItems: 'stretch' }}>
          {/* Título + ubicación */}
          <div style={{ padding: '16px 20px' }}>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#f0f0f0', letterSpacing: '-0.01em', textTransform: 'uppercase' }}>
              Protocolo a Seguir
            </h1>
            <p style={{ fontSize: '0.78rem', color: '#6b7280', marginTop: 4 }}>{protocol.location}</p>
          </div>

          {/* Timer */}
          <div style={{ borderLeft: '1px solid #242628', padding: '16px 24px', minWidth: 160 }}>
            <p style={{ fontSize: '0.56rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
              Tiempo transcurrido
            </p>
            <p style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f0f0f0', letterSpacing: '0.06em', marginTop: 6, fontVariantNumeric: 'tabular-nums' }}>
              {elapsedTime}
            </p>
          </div>

          {/* Personas afectadas */}
          <div style={{ borderLeft: '1px solid #242628', padding: '16px 24px', minWidth: 140 }}>
            <p style={{ fontSize: '0.56rem', letterSpacing: '0.18em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
              Personas afectadas
            </p>
            <p style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f0f0f0', letterSpacing: '0.06em', marginTop: 6, fontVariantNumeric: 'tabular-nums' }}>
              {String(protocol.affectedPersons).padStart(2, '0')}
            </p>
          </div>
        </div>
      </div>

      {/* ── Cuerpo principal: 2 columnas ── */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">

        {/* ── COLUMNA IZQUIERDA ── */}
        <div className="flex flex-col gap-4">

          {/* Panel: pasos del protocolo */}
          <Panel title="Guía Operativa" description="Marque cada paso al ejecutarlo">
            <ProtocolSteps steps={protocol.steps} onToggle={toggleStep} />
          </Panel>

          {/* Panel: notas del operador */}
          <Panel title="Notas del Operador" description="Lenguaje seguro — sin términos estigmatizantes">
            <NotesLog notes={protocol.notes} onAdd={addNote} />
          </Panel>

          {/* Panel: acciones de cierre */}
          <Panel title="Cierre de Protocolo" description="Documentación legal del incidente">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <Button variant="ghost" size="md" fullWidth onClick={generateDerivationSheet}>
                <ClipboardTextIcon size={15} />
                Generar ficha de derivación a centro de salud
              </Button>
              <Button variant="outline" size="md" fullWidth onClick={registerRejection}>
                <FileXIcon size={15} />
                Registrar rechazo de atención
              </Button>
            </div>
          </Panel>

        </div>

        {/* ── COLUMNA DERECHA ── */}
        <div className="flex flex-col gap-4">

          {/* Feed de cámara */}
          <Panel title="Cámara en vivo" description={protocol.station}>
            <div style={{ position: 'relative', backgroundColor: '#0a0b0d', borderRadius: 8, overflow: 'hidden', minHeight: 220 }}
                 className="flex items-center justify-center aspect-video w-full">
              <img
                key={protocol.cameraId || 1}
                src={`http://localhost:8000/api/stream/video_feed/${protocol.cameraId || 1}`}
                alt="Feed en vivo"
                className="w-full h-full object-fill select-none pointer-events-none"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  const placeholder = document.getElementById('protocol-camera-placeholder');
                  if (placeholder) placeholder.style.display = 'flex';
                }}
              />
              <div 
                id="protocol-camera-placeholder"
                className="absolute inset-0 flex flex-col items-center justify-center gap-3" 
                style={{ color: '#2a2d31', display: 'none' }}
              >
                <VideoCameraIcon size={40} />
                <p style={{ fontSize: '0.8rem', color: '#ef4444' }}>
                  Feed no disponible.
                </p>
              </div>
            </div>
          </Panel>

          {/* Canales de respuesta rápida */}
          <Panel title="Respuesta Rápida" description="Contacto directo con equipos de intervención">
            <ResponseChannels channels={protocol.channels} onCall={callChannel} />
          </Panel>

          {/* Categorización de riesgo C-SSRS */}
          <Panel title="Categorización de Riesgo" description="Escala Columbia — clasificación del operador">
            <RiskLevelSelector value={protocol.riskLevel} onChange={setRiskLevel} />
          </Panel>

          {/* Señales de alerta observadas */}
          <Panel title="Señales de Alerta" description="Actos preparatorios observados en cámara">
            <AlertSignalSelector signals={protocol.alertSignals} onToggle={toggleSignal} />
          </Panel>

        </div>
      </div>
    </section>
  )
}