import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReactElement } from 'react'

export interface MetroMapSvgProps {
  expanded?: boolean
}

const MIN_ZOOM = 0.5
const MAX_ZOOM = 5
const ZOOM_STEP = 0.3

export function MetroMapSvg({ expanded = false }: MetroMapSvgProps): ReactElement {
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const dragStart = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)

  // Reset zoom/pan when switching modes
  useEffect(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [expanded])

  const clampOffset = useCallback(
    (z: number, ox: number, oy: number) => {
      if (!containerRef.current) return { x: ox, y: oy }
      const cw = containerRef.current.clientWidth
      const ch = containerRef.current.clientHeight
      const maxX = ((z - 1) * cw) / 2
      const maxY = ((z - 1) * ch) / 2
      return {
        x: Math.max(-maxX, Math.min(maxX, ox)),
        y: Math.max(-maxY, Math.min(maxY, oy)),
      }
    },
    [],
  )

  const applyZoom = useCallback(
    (delta: number, cx?: number, cy?: number) => {
      setZoom((prev) => {
        const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev + delta))
        const container = containerRef.current
        if (container && cx !== undefined && cy !== undefined) {
          const rect = container.getBoundingClientRect()
          const px = cx - rect.left - rect.width / 2
          const py = cy - rect.top - rect.height / 2
          setOffset((o) => {
            const scaleFactor = next / prev
            const ox = px - (px - o.x) * scaleFactor
            const oy = py - (py - o.y) * scaleFactor
            return clampOffset(next, ox, oy)
          })
        } else {
          setOffset((o) => clampOffset(next, o.x, o.y))
        }
        return next
      })
    },
    [clampOffset],
  )

  // Wheel zoom
  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      if (!expanded) return
      e.preventDefault()
      applyZoom(e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP, e.clientX, e.clientY)
    },
    [expanded, applyZoom],
  )

  // Drag pan
  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!expanded) return
      e.currentTarget.setPointerCapture(e.pointerId)
      dragStart.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y }
      setDragging(true)
    },
    [expanded, offset],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging || !dragStart.current) return
      const dx = e.clientX - dragStart.current.x
      const dy = e.clientY - dragStart.current.y
      setOffset(clampOffset(zoom, dragStart.current.ox + dx, dragStart.current.oy + dy))
    },
    [dragging, zoom, clampOffset],
  )

  const onPointerUp = useCallback(() => {
    setDragging(false)
    dragStart.current = null
  }, [])

  const zoomIn  = () => applyZoom(ZOOM_STEP)
  const zoomOut = () => applyZoom(-ZOOM_STEP)
  const reset   = () => { setZoom(1); setOffset({ x: 0, y: 0 }) }

  const zoomPct = Math.round(zoom * 100)

  if (!expanded) {
    // Compact panel view — static image, no zoom controls
    return (
      <div
        style={{
          width: '100%',
          height: 280,
          backgroundColor: '#0a0b0d',
          borderRadius: 8,
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <img
          src="/mapa.png"
          alt="Mapa de la red de metro"
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
          draggable={false}
        />
      </div>
    )
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '70vh', backgroundColor: '#0a0b0d', borderRadius: 8, overflow: 'hidden' }}>

      {/* ── Zoom controls ── */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          zIndex: 10,
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          alignItems: 'center',
        }}
      >
        {/* Zoom % badge */}
        <div
          style={{
            backgroundColor: 'rgba(22,23,25,0.92)',
            border: '1px solid #2a2d31',
            borderRadius: 6,
            padding: '3px 8px',
            fontSize: '0.65rem',
            fontWeight: 700,
            color: '#9ca3af',
            letterSpacing: '0.06em',
            fontVariantNumeric: 'tabular-nums',
            minWidth: 44,
            textAlign: 'center',
            backdropFilter: 'blur(6px)',
          }}
        >
          {zoomPct}%
        </div>

        {/* + button */}
        <button
          onClick={zoomIn}
          disabled={zoom >= MAX_ZOOM}
          title="Acercar"
          style={controlBtn(zoom >= MAX_ZOOM)}
        >
          +
        </button>

        {/* reset button */}
        <button
          onClick={reset}
          title="Restablecer vista"
          style={controlBtn(false)}
        >
          ⊙
        </button>

        {/* − button */}
        <button
          onClick={zoomOut}
          disabled={zoom <= MIN_ZOOM}
          title="Alejar"
          style={controlBtn(zoom <= MIN_ZOOM)}
        >
          −
        </button>
      </div>

      {/* ── Hint ── */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          zIndex: 10,
          backgroundColor: 'rgba(22,23,25,0.82)',
          border: '1px solid #2a2d31',
          borderRadius: 6,
          padding: '4px 10px',
          fontSize: '0.6rem',
          color: '#4b4f56',
          backdropFilter: 'blur(6px)',
          pointerEvents: 'none',
          letterSpacing: '0.04em',
        }}
      >
        Rueda para zoom · Arrastra para mover
      </div>

      {/* ── Map canvas ── */}
      <div
        ref={containerRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        style={{
          width: '100%',
          height: '100%',
          cursor: dragging ? 'grabbing' : 'grab',
          userSelect: 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          touchAction: 'none',
        }}
      >
        <img
          src="/mapa.png"
          alt="Mapa de la red de metro"
          draggable={false}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            display: 'block',
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: dragging ? 'none' : 'transform 0.15s ease',
            willChange: 'transform',
          }}
        />
      </div>
    </div>
  )
}

function controlBtn(disabled: boolean): React.CSSProperties {
  return {
    width: 32,
    height: 32,
    borderRadius: 7,
    border: '1px solid #2a2d31',
    backgroundColor: disabled ? 'rgba(22,23,25,0.5)' : 'rgba(22,23,25,0.92)',
    color: disabled ? '#3a3d41' : '#d1d5db',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backdropFilter: 'blur(6px)',
    transition: 'background-color 0.15s, color 0.15s',
    lineHeight: 1,
  }
}