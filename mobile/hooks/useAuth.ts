// Este hook está mantenido por compatibilidad.
// La autenticación principal ahora está en store/auth.tsx (AuthProvider + useAuth).
// Si necesitas useAuth fuera del AuthProvider, usa store/auth en su lugar.

export { useAuth } from '../store/auth';