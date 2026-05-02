/**
 * Identifica la vista principal activa dentro del dashboard.
 */
export type ViewId = 'dashboard' | 'camera' | 'alerts' | 'protocols' | 'history'

/**
 * Define los tonos visuales usados para resaltar métricas y estados.
 */
export type Tone = 'slate' | 'blue' | 'amber' | 'red' | 'emerald'

/**
 * Describe una métrica visible en las tarjetas del panel de control.
 */
export interface DashboardMetric {
  id: string
  label: string
  value: string
  caption: string
  tone: Tone
}

/**
 * Describe el estado de una línea de monitoreo.
 */
export interface LineStatus {
  id: string
  code: string
  name: string
  detail: string
  tone: Tone
}

/**
 * Agrupa los datos visibles en la vista principal del dashboard.
 */
export interface DashboardOverview {
  title: string
  subtitle: string
  uptimeSeconds: number
  metrics: DashboardMetric[]
  mapTitle: string
  mapSubtitle: string
  lineStatuses: LineStatus[]
}

/**
 * Describe una tarjeta de soporte o recurso disponible para una alerta.
 */
export interface SupportResource {
  id: string
  name: string
  eta: string
  tone: Tone
}

/**
 * Representa una alerta crítica mostrada en la vista de cámaras.
 */
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

/**
 * Describe una métrica breve usada dentro de la vista de cámara.
 */
export interface CameraInsight {
  id: string
  label: string
  value: string
  tone: Tone
}

/**
 * Agrupa los datos simulados de la vista de cámara en vivo.
 */
export interface LiveCameraOverview {
  title: string
  subtitle: string
  cameraLabel: string
  cameraSubtitle: string
  elapsedSeconds: number
  insights: CameraInsight[]
  supportResources: SupportResource[]
}
