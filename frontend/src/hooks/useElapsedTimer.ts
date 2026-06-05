import { useEffect, useState } from 'react'

/**
 * Convierte segundos en una cadena HH:MM:SS.
 * @param totalSeconds - Tiempo total expresado en segundos.
 * @returns Tiempo formateado como texto legible.
 */
function formatElapsedTime(totalSeconds: number): string {
  const safeSeconds = Math.max(0, totalSeconds)
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const seconds = safeSeconds % 60

  return [hours, minutes, seconds].map((value) => value.toString().padStart(2, '0')).join(':')
}

/**
 * Mantiene un contador visible en pantalla con incremento por segundo.
 * @param initialSeconds - Valor inicial del contador.
 * @returns El tiempo transcurrido formateado y estable.
 */
export function useElapsedTimer(initialSeconds: number): { elapsedTime: string } {
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(initialSeconds)

  useEffect(() => {
    setElapsedSeconds(initialSeconds)
  }, [initialSeconds])

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setElapsedSeconds((currentSeconds) => currentSeconds + 1)
    }, 1000)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  return { elapsedTime: formatElapsedTime(elapsedSeconds) }
}
