import React, {
  createContext, useContext, useState,
  useCallback, ReactNode,
} from 'react';
import type { AuthState, AuthUser, LoginCredentials } from '../types/auth';

// Cambia esta URL por la IP/dominio real del backend en tu red
const API_BASE = 'http://localhost:8000';

interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  updateProfile: (data: Partial<AuthUser>) => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null, isAuthenticated: false, isLoading: false, error: null,
  });

  // Guardamos el token en memoria (dentro del contexto) para usarlo en /me y otras llamadas
  const [token, setToken] = useState<string | null>(null);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setState(s => ({ ...s, isLoading: true, error: null }));

    try {
      // 1. Obtener JWT
      const loginRes = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password,
        }),
      });

      if (!loginRes.ok) {
        const err = await loginRes.json().catch(() => ({}));
        const msg = err?.detail ?? 'Usuario o contraseña incorrectos.';
        setState(s => ({ ...s, isLoading: false, error: msg }));
        return;
      }

      const { access_token } = await loginRes.json();
      setToken(access_token);

      // 2. Obtener datos del usuario autenticado
      const meRes = await fetch(`${API_BASE}/api/auth/me`, {
        headers: { Authorization: `Bearer ${access_token}` },
      });

      if (!meRes.ok) {
        setState(s => ({ ...s, isLoading: false, error: 'Error al obtener el perfil.' }));
        return;
      }

      const me = await meRes.json();

      const user: AuthUser = {
        id:       me.id,
        fullName: me.full_name,
        username: me.username,
        role:     me.role,
        isActive: me.is_active,
      };

      setState({ user, isAuthenticated: true, isLoading: false, error: null });
    } catch (_) {
      setState(s => ({ ...s, isLoading: false, error: 'No se pudo conectar con el servidor.' }));
    }
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setState({ user: null, isAuthenticated: false, isLoading: false, error: null });
  }, []);

  const updateProfile = useCallback((data: Partial<AuthUser>) =>
    setState(s => s.user ? { ...s, user: { ...s.user, ...data } } : s), []);

  const clearError = useCallback(() =>
    setState(s => ({ ...s, error: null })), []);

  return (
    <AuthContext.Provider value={{ ...state, login, logout, updateProfile, clearError }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}