import type { ReactNode, ReactElement, ButtonHTMLAttributes } from 'react'

export type ButtonVariant = 'ghost' | 'solid' | 'danger' | 'outline' | 'primary'
export type ButtonSize = 'sm' | 'md' | 'lg'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  children: ReactNode
  fullWidth?: boolean
}

const VARIANT_STYLE: Record<ButtonVariant, string> = {
  ghost:   'bg-[#1c1e21] border border-[#2e3135] text-[#d1d5db] hover:bg-[#252729] hover:text-white',
  solid:   'bg-[#252729] border border-[#3a3d41] text-[#f0f0f0] hover:bg-[#2e3135] hover:text-white',
  danger:  'bg-[#2a0f0f] border border-[#5c1f1f] text-[#f87171] hover:bg-[#3d1212] hover:text-[#fca5a5]',
  outline: 'bg-transparent border border-[#242628] text-[#6b7280] hover:bg-[#1a1c1f] hover:text-[#9ca3af] uppercase tracking-[0.14em]',
  primary: 'bg-[#1a3451] border border-[#1e4976] text-[#38bdf8] hover:bg-[#1e4976] hover:text-white',
}

const SIZE_STYLE: Record<ButtonSize, string> = {
  sm: 'text-[0.68rem] px-3 py-1.5 rounded-md',
  md: 'text-[0.76rem] px-4 py-2 rounded-lg',
  lg: 'text-[0.82rem] px-5 py-2.5 rounded-lg',
}

/**
 * Botón base del sistema Andén Seguro.
 * Usa clases Tailwind para hover — sin conflictos con onMouseLeave.
 * Reutilizable en todas las vistas.
 */
export function Button({
  variant = 'ghost',
  size = 'md',
  children,
  fullWidth = false,
  className = '',
  ...rest
}: ButtonProps): ReactElement {
  return (
    <button
      type="button"
      className={[
        'inline-flex items-center justify-center gap-1.5 font-semibold transition-colors cursor-pointer',
        VARIANT_STYLE[variant],
        SIZE_STYLE[size],
        fullWidth ? 'w-full' : '',
        className,
      ].join(' ')}
      {...rest}
    >
      {children}
    </button>
  )
}