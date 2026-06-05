import React, {
  createContext, useContext, useState,
  useCallback, ReactNode,
} from 'react';
import type { AuthState, AuthUser, LoginCredentials } from '../types/auth';

const MOCK_USERS: Record<string, { password: string; user: AuthUser }> = {
  'agente.essus': {
    password: '1234',
    user: {
      id: '1', name: 'Agente R. Essus', username: 'agente.essus',
      role: 'agent', badge: 'UNX-9928', assignment: 'Estación Central L1',
    },
  },
  supervisor: {
    password: 'admin',
    user: {
      id: '2', name: 'Supervisora M. Torres', username: 'supervisor',
      role: 'supervisor', badge: 'SUP-0042', assignment: 'Centro de Control',
    },
  },
};

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

  const login = useCallback(async (credentials: LoginCredentials) => {
    setState(s => ({ ...s, isLoading: true, error: null }));
    await new Promise(r => setTimeout(r, 800));
    const match = MOCK_USERS[credentials.username.toLowerCase()];
    if (match && match.password === credentials.password) {
      setState({ user: match.user, isAuthenticated: true, isLoading: false, error: null });
    } else {
      setState(s => ({ ...s, isLoading: false, error: 'Usuario o contraseña incorrectos.' }));
    }
  }, []);

  const logout = useCallback(() =>
    setState({ user: null, isAuthenticated: false, isLoading: false, error: null }), []);

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