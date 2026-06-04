export type ViewId = 'dashboard' | 'camera' | 'alerts' | 'protocols' | 'history'
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