export type ViewId = 'dashboard' | 'camera' | 'alerts' | 'protocols' | 'history' | 'users'
export type Tone = 'slate' | 'blue' | 'amber' | 'red' | 'emerald' | 'purple' | 'gray' | 'orange' | 'pink'

export interface DashboardMetric {
  id: string
  label: string
  value: string
  unit?: string
  caption: string
  tone: Tone
}

export interface LineStatus {
  id: string
  code: string
  name: string
  detail: string
  tone: Tone
}

export interface DashboardOverview {
  title: string
  subtitle: string
  systemStatus: 'OPERATIVO' | 'PRECAUCIÓN' | 'FUERA DE SERVICIO'
  uptimeSeconds: number
  metrics: DashboardMetric[]
  mapTitle: string
  mapSubtitle: string
  lineStatuses: LineStatus[]
}

export interface SupportResource {
  id: string
  name: string
  eta: string
  tone: Tone
}

export interface IncidentAlert {
  title: string
  cameraLabel: string
  elapsedSeconds: number
  description: string
  location: string
  primaryActionLabel: string
  secondaryActionLabel: string
  supportResources: SupportResource[]
  trackIds?: number[]
}

export interface CameraInsight {
  id: string
  label: string
  value: string
  tone: Tone
}

export interface LiveCameraOverview {
  title: string
  subtitle: string
  cameraLabel: string
  cameraSubtitle: string
  elapsedSeconds: number
  insights: CameraInsight[]
  supportResources: SupportResource[]
}
/* ── PROTOCOLOS ─────────────────────────────────────────────────── */

/** Nivel de riesgo según escala C-SSRS */
export type RiskLevel = 'leve' | 'moderado' | 'alto'

/** Una señal de alerta observable seleccionable por el operador */
export interface AlertSignal {
  id: string
  label: string
  selected: boolean
}

/** Un paso del protocolo operativo con estado de completitud */
export interface ProtocolStep {
  id: string
  title: string
  description: string
  completed: boolean
}

/** Canal de respuesta rápida disponible */
export interface ResponseChannel {
  id: string
  label: string
  phone: string
  icon: 'security' | 'firefighters' | 'paramedics' | 'megaphone' | 'health' | 'police'
  tone: Tone
}

/** Nota registrada por el operador con timestamp */
export interface OperatorNote {
  id: string
  timestamp: string
  text: string
}

/** Estado completo del incidente activo en la vista de protocolos */
export interface ActiveProtocol {
  incidentLabel: string
  location: string
  station: string
  elapsedSeconds: number
  affectedPersons: number
  riskLevel: RiskLevel | null
  alertSignals: AlertSignal[]
  steps: ProtocolStep[]
  channels: ResponseChannel[]
  notes: OperatorNote[]
}