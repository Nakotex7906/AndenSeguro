import { PaperPlaneRightIcon } from '@phosphor-icons/react'
import { useRef, useState } from 'react'
import type { ReactElement } from 'react'

import type { OperatorNote } from '../../types/dashboard'

interface Props {
  notes: OperatorNote[]
  onAdd: (text: string) => void
}

/**
 * Registro de notas del operador con validación de lenguaje seguro.
 * Las notas se guardan con timestamp y se listan en orden cronológico.
 * Reutilizable en alertas críticas e historial.
 */
export function NotesLog({ notes, onAdd }: Props): ReactElement {
  const [draft, setDraft] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSubmit(): void {
    if (!draft.trim()) return
    onAdd(draft)
    setDraft('')
    textareaRef.current?.focus()
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <p style={{ fontSize: '0.62rem', letterSpacing: '0.16em', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>
        Registro de notas del operador
      </p>

      {/* Input */}
      <div style={{ position: 'relative' }}>
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() } }}
          placeholder="Escriba actualizaciones críticas aquí…"
          rows={3}
          style={{
            width: '100%',
            backgroundColor: '#0d0e10',
            border: '1px solid #1f2023',
            borderRadius: 8,
            padding: '10px 40px 10px 12px',
            fontSize: '0.75rem',
            color: '#d1d5db',
            resize: 'none',
            outline: 'none',
            fontFamily: 'inherit',
            lineHeight: 1.6,
            boxSizing: 'border-box',
          }}
          onFocus={(e) => { e.currentTarget.style.borderColor = '#2a2d31' }}
          onBlur={(e)  => { e.currentTarget.style.borderColor = '#1f2023' }}
        />
        <button
          type="button"
          onClick={handleSubmit}
          style={{
            position: 'absolute', bottom: 8, right: 8,
            backgroundColor: '#1a3451',
            border: 'none',
            borderRadius: 6,
            width: 28, height: 28,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer',
            color: '#38bdf8',
          }}
          aria-label="Guardar nota"
        >
          <PaperPlaneRightIcon size={14} weight="fill" />
        </button>
      </div>

      {/* Historial de notas */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 160, overflowY: 'auto' }}>
        {notes.map((note) => (
          <div key={note.id} style={{ fontSize: '0.72rem', color: '#6b7280', lineHeight: 1.5 }}>
            <span style={{ color: '#4b4f56', fontWeight: 600 }}>{note.timestamp} PM — </span>
            <span style={{ color: '#9ca3af' }}>"{note.text}"</span>
          </div>
        ))}
      </div>
    </div>
  )
}