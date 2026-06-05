import { useCallback, useState, useEffect } from 'react'

import type { ActiveProtocol, OperatorNote, RiskLevel } from '../types/dashboard'

/**
 * Hook principal de la vista de Protocolos.
 * Gestiona el estado local del incidente activo: pasos, nivel de riesgo,
 * señales de alerta, notas del operador y cierre de protocolo.
 * Hidrata el estado inicial desde el backend usando la base de datos.
 */
export function useActiveProtocol(incidentId: number = 1): {
  protocol: ActiveProtocol
  toggleStep:       (stepId: string) => void
  setRiskLevel:     (level: RiskLevel) => void
  toggleSignal:     (signalId: string) => void
  addNote:          (text: string) => void
  callChannel:      (channelId: string) => void
  generateDerivationSheet: () => void
  registerRejection: () => void
} {
  const [protocol, setProtocol] = useState<ActiveProtocol>({
    incidentLabel:  'INCIDENTE ACTIVO',
    location:       'Estación Central • Andén',
    station:        'CAM-01 ANDÉN',
    elapsedSeconds: 0,
    affectedPersons: 1,
    riskLevel: null,

    /* Señales de alerta observables — menú de selección rápida (C-SSRS) */
    alertSignals: [
      { id: 's1', label: 'Traspaso de barrera / zona crítica',   selected: false },
      { id: 's2', label: 'Manipulación de medios letales',        selected: false },
      { id: 's3', label: 'Conducta de cierre (deja pertenencias)', selected: false },
      { id: 's4', label: 'Inmovilidad prolongada en zona de riesgo', selected: false },
      { id: 's5', label: 'Agitación o llanto extremo',            selected: false },
    ],

    /* Pasos del protocolo operativo */
    steps: [
      {
        id: 'p1',
        title: 'Notificar servicios de emergencias',
        description: 'Confirmar recepción de alerta por parte de central de despacho externa.',
        completed: false,
      },
      {
        id: 'p2',
        title: 'Identificación y filtro conductual',
        description: 'Identificar características de la persona para despachar personal de ayuda.',
        completed: false,
      },
      {
        id: 'p3',
        title: 'Despacho de personal adecuado',
        description: 'Cierre de torniquetes y activación de salidas de emergencia en todos los niveles.',
        completed: false,
      },
      {
        id: 'p4',
        title: 'Custodia por videovigilancia',
        description: 'Mantener seguimiento en tiempo real del personal de seguridad durante la intervención.',
        completed: false,
      },
    ],

    /* Canales de respuesta rápida */
    channels: [
      { id: 'c1', label: 'Seguridad',   phone: 'Int. 100', icon: 'security',     tone: 'blue'    },
      { id: 'c2', label: 'Bomberos',    phone: '132',       icon: 'firefighters', tone: 'red'     },
      { id: 'c3', label: 'Paramédicos', phone: 'SAMU 131',  icon: 'paramedics',   tone: 'emerald' },
      { id: 'c4', label: 'Megafonía',   phone: 'Int. 200',  icon: 'megaphone',    tone: 'amber'   },
      { id: 'c5', label: 'Línea Salud', phone: '600 360 7777', icon: 'health',    tone: 'purple'  },
      { id: 'c6', label: 'Carabineros', phone: '133',       icon: 'police',       tone: 'slate'   },
    ],

    notes: [],
  })

  useEffect(() => {
    fetch(`http://localhost:8000/api/incidents/${incidentId}/protocol-state`, {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token') || ''}` }
    })
      .then(res => res.json())
      .then(data => {
        if (!data.incident) return

        const startTime = new Date(data.incident.timestamp).getTime()
        const elapsed = Math.floor((Date.now() - startTime) / 1000)

        const hydratedNotes = data.notes.map((n: any) => ({
          id: `n${n.id}`,
          timestamp: new Date(n.timestamp).toLocaleTimeString('es-CL', { hour: '2-digit', minute: '2-digit' }),
          text: n.text
        }))

        const riskActions = data.actions.filter((a: any) => a.action_type === 'risk_level')
        const finalRiskLevel = riskActions.length > 0 ? riskActions[riskActions.length - 1].action_value : null

        const stepToggles = data.actions.filter((a: any) => a.action_type === 'step_toggle')
        const stepCounts: Record<string, number> = {}
        stepToggles.forEach((a: any) => {
          stepCounts[a.action_value] = (stepCounts[a.action_value] || 0) + 1
        })
        
        const signalToggles = data.actions.filter((a: any) => a.action_type === 'signal_toggle')
        const signalCounts: Record<string, number> = {}
        signalToggles.forEach((a: any) => {
          signalCounts[a.action_value] = (signalCounts[a.action_value] || 0) + 1
        })

        setProtocol(prev => ({
          ...prev,
          incidentLabel: `INCIDENTE #${data.incident.id} ACTIVO`,
          cameraId: data.incident.camera_id,
          elapsedSeconds: elapsed,
          riskLevel: finalRiskLevel,
          notes: hydratedNotes.length > 0 ? hydratedNotes : prev.notes,
          steps: prev.steps.map(s => ({
            ...s,
            completed: (stepCounts[s.id] || 0) % 2 !== 0
          })),
          alertSignals: prev.alertSignals.map(s => ({
            ...s,
            selected: (signalCounts[s.id] || 0) % 2 !== 0
          }))
        }))
      })
      .catch(console.error)
  }, [incidentId])

  /** Marca/desmarca un paso como completado */
  const toggleStep = useCallback((stepId: string) => {
    setProtocol((prev) => ({
      ...prev,
      steps: prev.steps.map((s) =>
        s.id === stepId ? { ...s, completed: !s.completed } : s
      ),
    }))
    
    fetch(`http://localhost:8000/api/incidents/${incidentId}/actions`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify({ type: 'step_toggle', value: stepId })
    }).catch(console.error)
  }, [incidentId])

  /** Asigna el nivel de riesgo C-SSRS */
  const setRiskLevel = useCallback((level: RiskLevel) => {
    setProtocol((prev) => ({ ...prev, riskLevel: level }))
    
    fetch(`http://localhost:8000/api/incidents/${incidentId}/actions`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify({ type: 'risk_level', value: level })
    }).catch(console.error)
  }, [incidentId])

  /** Selecciona/deselecciona una señal de alerta observable */
  const toggleSignal = useCallback((signalId: string) => {
    setProtocol((prev) => ({
      ...prev,
      alertSignals: prev.alertSignals.map((s) =>
        s.id === signalId ? { ...s, selected: !s.selected } : s
      ),
    }))
    
    fetch(`http://localhost:8000/api/incidents/${incidentId}/actions`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify({ type: 'signal_toggle', value: signalId })
    }).catch(console.error)
  }, [incidentId])

  /** Agrega una nota del operador con timestamp */
  const addNote = useCallback((text: string) => {
    if (!text.trim()) return
    /* Validación: filtro de lenguaje estigmatizante */
    const stigmatizing = ['suicida', 'intento fallido', 'se mató']
    const hasStigma = stigmatizing.some((w) => text.toLowerCase().includes(w))
    if (hasStigma) {
      alert(
        '⚠️ Término no recomendado detectado.\n\n' +
        'Usar: "persona que intentó suicidarse" o "muerte por suicidio".\n' +
        'Edita la nota antes de guardar.'
      )
      return
    }
    const note: OperatorNote = { id: `n${Date.now()}`, timestamp: formatNow(), text: text.trim() }
    setProtocol((prev) => ({ ...prev, notes: [...prev.notes, note] }))
    
    fetch(`http://localhost:8000/api/incidents/${incidentId}/notes`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify({ text: text.trim() })
    }).catch(console.error)
  }, [incidentId])

  /** Simula la llamada a un canal de respuesta */
  const callChannel = useCallback((channelId: string) => {
    const ch = protocol.channels.find((c) => c.id === channelId)
    if (!ch) return
    console.info(`[Protocolo] Contactando ${ch.label} — ${ch.phone}`)
    
    fetch(`http://localhost:8000/api/incidents/${incidentId}/actions`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify({ type: 'channel_call', value: channelId })
    }).catch(console.error)
  }, [incidentId, protocol.channels])

  /** Genera ficha de derivación a centro de salud */
  const generateDerivationSheet = useCallback(() => {
    fetch(`http://localhost:8000/api/incidents/${incidentId}/derivation-sheet`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token') || ''}` }
    })
    .then(res => res.json())
    .then(data => alert(data.message))
    .catch(console.error)
  }, [incidentId])

  /** Registra el rechazo formal de atención por parte de la persona */
  const registerRejection = useCallback(() => {
    fetch(`http://localhost:8000/api/incidents/${incidentId}/rejection`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token') || ''}` }
    })
    .then(res => res.json())
    .then(data => alert(data.message))
    .catch(console.error)
  }, [incidentId])

  return {
    protocol,
    toggleStep,
    setRiskLevel,
    toggleSignal,
    addNote,
    callChannel,
    generateDerivationSheet,
    registerRejection,
  }
}

function formatNow(): string {
  return new Date().toLocaleTimeString('es-CL', { hour: '2-digit', minute: '2-digit' })
}