export interface AuthUser {
  id: number;
  fullName: string;
  username: string;
  role: 'admin' | 'jefe_estacion' | 'seguridad' | 'operador';
  isActive: boolean;
  photoUri?: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}