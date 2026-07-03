import React, {
  createContext, useContext, useState,
  useCallback, ReactNode,
} from 'react';
import type { AuthState, AuthUser, LoginCredentials } from '../types/auth';

/**
 * Extensión de la interfaz de usuario para incluir de manera persistente 
 * el token de acceso proveído por el backend de FastAPI.
 */
type AuthenticatedUser = AuthUser & { accessToken?: string };

interface AuthContextType extends Omit<AuthState, 'user'> {
  user: AuthenticatedUser | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  updateProfile: (data: Partial<AuthenticatedUser>) => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

/**
 * Dirección IP base del backend de Andén Seguro.
 * Reemplazar con la IP de tu entorno de desarrollo local (ej: '192.168.1.X') o de producción.
 */
const BACKEND_API_URL = 'http://192.168.1.88:8000';

/**
 * Proveedor del contexto de autenticación global de la plataforma Andén Seguro.
 * Gestiona de forma centralizada la comunicación con los endpoints de seguridad de FastAPI.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState & { user: AuthenticatedUser | null }>({
    user: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
  });

  /**
   * Realiza la llamada HTTP POST al endpoint de autenticación para validar credenciales.
   * * @param credentials Contrato con el nombre de usuario y contraseña del agente.
   * @returns El string del JWT si la operación es exitosa.
   * @throws Error si el servidor responde con un estado no exitoso.
   */
  const requestAccessToken = async (credentials: LoginCredentials): Promise<string> => {
    const response = await fetch(`${BACKEND_API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: credentials.username.toLowerCase(),
        password: credentials.password,
      }),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Usuario o contraseña incorrectos.');
      }
      if (response.status === 403) {
        throw new Error('Cuenta desactivada. Contacte al administrador.');
      }
      throw new Error('Error de conexión con el servidor de seguridad.');
    }

    const data = await response.json();
    return data.access_token; 
  };

  /**
   * Solicita los datos de perfil del usuario actualmente autenticado utilizando el JWT obtenido.
   * * @param token JSON Web Token de acceso válido.
   * @returns Un objeto parcial mapeado al formato esperado por el Frontend.
   */
  const fetchUserProfile = async (token: string): Promise<Partial<AuthenticatedUser>> => {
    const response = await fetch(`${BACKEND_API_URL}/api/auth/me`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error('No se pudo recuperar la información del perfil operativo.');
    }

    const backendUser = await response.json();

    // Mapeo seguro de los atributos del backend (SQLModel) a los contratos del Frontend (Typescript)
    return {
      id: String(backendUser.id || '0'),
      name: backendUser.full_name || backendUser.username,
      username: backendUser.username,
      role: backendUser.role,
      badge: backendUser.badge || 'N/A', 
      assignment: backendUser.assignment || 'No Asignado',
    };
  };

  /**
   * Orquesta el flujo completo de inicio de sesión de manera secuencial y segura.
   * Modifica el estado global de la aplicación garantizando la consistencia de los datos.
   */
  const login = useCallback(async (credentials: LoginCredentials) => {
    setState(currentState => ({ ...currentState, isLoading: true, error: null }));

    try {
      // Obtener Token Seguro
      const validToken = await requestAccessToken(credentials);

      // Obtener Perfil Operativo real usando el token recién emitido
      const userProfile = await fetchUserProfile(validToken);

      // Consolidar el estado de autenticación global
      setState({
        user: { ...userProfile, accessToken: validToken } as AuthenticatedUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });

    } catch (authError: any) {
      setState(currentState => ({
        ...currentState,
        isLoading: false,
        error: authError.message || 'Ocurrió un error inesperado al intentar ingresar.',
      }));
    }
  }, []);

  /**
   * Revoca la sesión del usuario limpiando los estados locales de memoria.
   */
  const logout = useCallback(() =>
    setState({ user: null, isAuthenticated: false, isLoading: false, error: null }), []);

  /**
   * Permite actualizar dinámicamente datos del perfil en memoria del agente.
   */
  const updateProfile = useCallback((data: Partial<AuthenticatedUser>) =>
    setState(currentState => currentState.user ? { ...currentState, user: { ...currentState.user, ...data } } : currentState), []);

  /**
   * Remueve cualquier mensaje de error activo en la interfaz de login.
   */
  const clearError = useCallback(() =>
    setState(currentState => ({ ...currentState, error: null })), []);

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