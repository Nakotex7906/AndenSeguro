export interface AuthUser {
  id: string;
  name: string;
  username: string;
  role: 'agent' | 'supervisor' | 'admin';
  badge: string;
  assignment: string;
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
