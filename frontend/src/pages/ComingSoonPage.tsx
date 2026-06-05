import type { ReactElement } from 'react'

import type { ViewId } from '../types/dashboard'

/**
 * Propiedades de la vista de apoyo para secciones no implementadas.
 */
export interface ComingSoonPageProps {
  viewId: ViewId
}

/**
 * Renderiza una pantalla de transición para áreas aún no desarrolladas.
 * @param props - Identificador de la sección solicitada.
 * @returns Una vista de estado vacío con contexto funcional.
 */
export function ComingSoonPage({ viewId }: ComingSoonPageProps): ReactElement {
  const title =
    viewId === 'alerts' ? 'Alertas críticas' : viewId === 'protocols' ? 'Protocolos' : 'Historial'

  return (
    <section className="surface-panel p-6">
      <p className="text-sm font-semibold tracking-[0.18em] text-sky-300 uppercase">
        Módulo en construcción
      </p>
      <h1 className="mt-3 text-3xl font-semibold text-white">{title}</h1>
      <p className="mt-3 max-w-2xl text-slate-300">
        Esta estructura ya queda preparada para incorporar la vista completa sin cambiar el layout
        global ni la navegación.
      </p>
    </section>
  )
}
