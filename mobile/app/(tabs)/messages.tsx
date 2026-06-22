import React, { useState, useEffect } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View, Platform, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Notifications from 'expo-notifications';
import { Image } from 'expo-image';

import { useAuth } from '../../store/auth';
import { TopBar } from '../../components/layout/TopBar';
import { StatusPill } from '../../components/ui/StatusPill';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { AppButton } from '../../components/ui/AppButton';
import { Palette, FontSize, FontWeight, Space, Radius, Shadow } from '../../constants/theme';
import type { IncidentAlert, AlertStatus } from '../../types/alert';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

const STATUS_CONFIG: Record<AlertStatus, { label: string; pillStatus: 'active' | 'warning' | 'error' | 'neutral' | 'info' }> = {
  in_progress: { label: 'EN CURSO', pillStatus: 'error' },
  pending: { label: 'PENDIENTE', pillStatus: 'warning' },
  completed: { label: 'COMPLETADO', pillStatus: 'active' },
  false_alarm: { label: 'FALSA ALARMA', pillStatus: 'neutral' },
};

const RISK_COLORS: Record<string, string> = {
  high: Palette.red,
  medium: Palette.amber,
  low: Palette.green,
};

/**
 * Calcula de forma semántica el tiempo transcurrido desde la detección del incidente.
 * * @param isoString Representación de fecha en formato ISO.
 * @returns Cadena con formato corto simplificado (ej: 4s, 12min, 2h).
 */
function timeAgo(isoString: string): string {
  const diffInSeconds = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (diffInSeconds < 60) return `${diffInSeconds}s`;
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}min`;
  return `${Math.floor(diffInSeconds / 3600)}h`;
}

/**
 * Solicita permisos de notificaciones al sistema operativo y genera el token único de Expo.
 * * @returns El token push generado o undefined en caso de denegación o fallo.
 */
async function registerForPushNotificationsAsync(): Promise<string | undefined> {
  let pushToken: string | undefined;

  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'default',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: Palette.red,
    });
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;

  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  if (finalStatus !== 'granted') {
    console.warn('[PUSH] Permiso de notificaciones denegado.');
    return undefined;
  }

  try {
    const projectId = 'dbca63a5-9ed1-4883-8688-df3c1dfba377';
    pushToken = (await Notifications.getExpoPushTokenAsync({ projectId })).data;
  } catch (error) {
    console.error('[PUSH] Error al obtener el token de Firebase:', error);
  }

  return pushToken;
}

/**
 * Petición HTTP segura al backend para extraer el historial de incidentes usando el token JWT.
 * * @param baseIp Dirección IP del servidor de FastAPI.
 * @param token Token de acceso JWT real provisto por la autenticación.
 * @returns Promesa con la respuesta cruda del servidor (puede ser un Array o un Objeto).
 */
async function fetchSecureIncidentsHistory(baseIp: string, token: string | null): Promise<any> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`http://${baseIp}:8000/api/incidents`, {
    method: 'GET',
    headers: headers,
  });

  if (response.status === 401) {
    throw new Error('401 Unauthorized: El token JWT expiró o es inválido.');
  }

  if (!response.ok) {
    throw new Error(`Fallo operativo del servidor: Código ${response.status}`);
  }

  return await response.json();
}

interface AlertCardProps {
  alert: IncidentAlert & { description?: string; imageUrl?: string };
  onComplete: (id: string) => void;
  onFalseAlarm: (id: string) => void;
}

function AlertCard({ alert, onComplete, onFalseAlarm }: AlertCardProps) {
  const [expanded, setExpanded] = useState<boolean>(alert.status === 'in_progress');
  const config = STATUS_CONFIG[alert.status];
  const isResolved = alert.status === 'completed' || alert.status === 'false_alarm';

  return (
    <View style={[styles.card, alert.status === 'in_progress' && Shadow.alert]}>
      <TouchableOpacity
        style={styles.cardHeader}
        onPress={() => setExpanded(prev => !prev)}
        activeOpacity={0.8}
      >
        <View style={[styles.riskBar, { backgroundColor: RISK_COLORS[alert.riskLevel] }]} />
        <View style={styles.cardHeaderContent}>
          <View style={styles.cardTitleRow}>
            <Text style={styles.alertId}>{alert.id}</Text>
            <StatusPill label={config.label} status={config.pillStatus} />
          </View>
          <Text style={styles.alertZone}>{alert.zone}</Text>
          <View style={styles.cardMeta}>
            <Ionicons name="time-outline" size={11} color={Palette.textDim} />
            <Text style={styles.alertTime}>Hace {timeAgo(alert.detectedAt)}</Text>
          </View>
        </View>
        <Ionicons name={expanded ? 'chevron-up' : 'chevron-down'} size={16} color={Palette.textDim} style={{ marginLeft: 4 }} />
      </TouchableOpacity>

      {expanded && (
        <View style={styles.cardBody}>
          {alert.description ? (
            <View style={styles.aiReportContainer}>
              <View style={styles.aiHeader}>
                <Ionicons name="analytics-outline" size={14} color={Palette.red} />
                <Text style={styles.aiTitle}>REPORTE DE IA DE VISIÓN</Text>
              </View>
              <Text style={styles.aiDescriptionText}>{alert.description}</Text>
            </View>
          ) : (
            <View style={styles.infoGrid}>
              <View style={styles.fallbackCell}>
                <Text style={styles.fallbackText}>Sin descripción analítica disponible.</Text>
              </View>
            </View>
          )}

          {alert.imageUrl ? (
            <View style={styles.imageContainer}>
              <View style={styles.imageHeader}>
                <Ionicons name="image-outline" size={12} color={Palette.textDim} />
                <Text style={styles.imageTitle}>CAPTURA EN TIEMPO REAL</Text>
              </View>
              <Image source={{ uri: alert.imageUrl }} style={styles.screenshotImage} contentFit="cover" transition={300} />
            </View>
          ) : (
            <View style={styles.feedPlaceholder}>
              <Ionicons name="image-outline" size={22} color={Palette.textDim} />
              <Text style={styles.feedText}>CAPTURA DEL INCIDENTE</Text>
              <Text style={styles.feedSub}>Procesando archivo visual…</Text>
            </View>
          )}

          {!isResolved && (
            <View style={styles.actions}>
              <View style={styles.actionBtn}>
                <AppButton label="Completado" variant="primary" icon="checkmark-circle-outline" onPress={() => onComplete(alert.id)} />
              </View>
              <View style={styles.actionBtn}>
                <AppButton label="Falsa alarma" variant="warning" icon="close-circle-outline" onPress={() => onFalseAlarm(alert.id)} />
              </View>
            </View>
          )}

          {isResolved && (
            <View style={styles.resolvedBanner}>
              <Ionicons name={alert.status === 'completed' ? 'checkmark-circle' : 'close-circle'} size={15} color={alert.status === 'completed' ? Palette.green : Palette.amber} />
              <Text style={[styles.resolvedText, { color: alert.status === 'completed' ? Palette.green : Palette.amber }]}>
                {alert.status === 'completed' ? 'Incidente cerrado correctamente' : 'Marcado como falsa alarma'}
              </Text>
            </View>
          )}
        </View>
      )}
    </View>
  );
}

/**
 * Componente de Pantalla de Gestión de Mensajes e Alertas Críticas de Andén Seguro.
 */
export default function MensajesScreen() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const { user } = useAuth();

  // IP Unificada del Backend de Andén Seguro para evitar duplicaciones
  const BACKEND_BASE_IP = '192.168.1.88';

  useEffect(() => {
    /**
     * Controlador encargado de orquestar la llamada al servicio seguro e indexar
     * los resultados transformados en el estado local de React.
     */
    /**
     * Controlador encargado de orquestar la llamada al servicio seguro e indexar
     * los resultados transformados en el estado local de React.
     */
    const loadIncidentsHistory = async () => {
      try {
        const securityToken = user?.accessToken || null;
        const rawIncidents: any = await fetchSecureIncidentsHistory(BACKEND_BASE_IP, securityToken);

        console.log('[DEBUG ANDÉN SEGURO] Respuesta del servidor:', rawIncidents);

        const incidentsArray = Array.isArray(rawIncidents)
          ? rawIncidents
          : (rawIncidents.items || rawIncidents.incidents || rawIncidents.data || []);

        const mappedAlerts = incidentsArray.map((incident: any) => ({
          id: `INC-${incident.id}`,
          status: incident.status || 'in_progress',
          riskLevel: incident.alert_level === 'red' ? 'high' : 'medium',
          zone: `ANDÉN - CÁMARA #${incident.camera_id}`,
          description: incident.description,
          imageUrl: incident.image_url,
          detectedAt: incident.timestamp || incident.created_at || new Date().toISOString(),
        }));

        setAlerts(mappedAlerts.reverse());
      } catch (error) {
        console.error('[HTTP MÓVIL] Falla crítica al sincronizar historial:', error);
      } finally {
        setLoading(false);
      }
    };

    loadIncidentsHistory();
  }, [user?.accessToken]); // Solo se ejecuta si cambia el token JWT real de inicio de sesión

  /**
   * Responsable exclusivo de la inicialización y registro del Token Push de Notificaciones
   */
  useEffect(() => {
    if (!user?.accessToken) return;

    registerForPushNotificationsAsync().then(async (generatedToken) => {
      if (!generatedToken) return;
      try {
        await fetch(`http://${BACKEND_BASE_IP}:8000/api/auth/register-push-token`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${user.accessToken}` 
          },
          body: JSON.stringify({ push_token: generatedToken }),
        });
        console.log('[PUSH] Token sincronizado exitosamente con el backend.');
      } catch (error) {
        console.error('[PUSH] Fallo al sincronizar token con el servidor:', error);
      }
    });
  }, [user?.accessToken]); // Se ejecuta una sola vez cuando el usuario se loguea con éxito

  /**
   * Responsable exclusivo de mantener el canal reactivo en tiempo real por WebSockets
   */
  useEffect(() => {
    if (!user?.accessToken) return;

    const socketUrl = `ws://${BACKEND_BASE_IP}:8000/api/alerts/ws`;
    const wsClient = new WebSocket(socketUrl);

    wsClient.onopen = () => {
      console.log('[WS MÓVIL] Canal en tiempo real establecido de forma segura.');
    };

    wsClient.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'incident_enriched') {
          const newAlert = {
            id: `INC-${payload.incident_id}`,
            status: 'in_progress' as AlertStatus,
            riskLevel: payload.level === 'red' ? 'high' : 'medium',
            zone: `ANDÉN - CÁMARA #${payload.camera_id}`,
            description: payload.description,
            imageUrl: payload.image_url,
            detectedAt: new Date().toISOString(),
          };
          setAlerts(prev => [newAlert, ...prev]);
        }
      } catch (error) {
        console.error('[WS MÓVIL] Error interpretando payload entrante:', error);
      }
    };

    return () => {
      console.log('[WS MÓVIL] Desconectando canal para evitar fugas de subprocesos.');
      wsClient.close();
    };
  }, [user?.accessToken]);

  const resolveIncident = (id: string, status: AlertStatus) => {
    setAlerts(prev => prev.map(alert => alert.id === id ? { ...alert, status, respondedAt: new Date().toISOString() } : alert));
  };

  const activeAlerts = alerts.filter(alert => alert.status === 'in_progress' || alert.status === 'pending');
  const historicalAlerts = alerts.filter(alert => alert.status === 'completed' || alert.status === 'false_alarm');

  if (loading) {
    return (
      <View style={[styles.root, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color={Palette.red} />
        <Text style={{ color: Palette.textDim, marginTop: 10, fontSize: FontSize.sm }}>Sincronizando con el andén...</Text>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <TopBar right={activeAlerts.length > 0 ? (
        <View style={styles.badge}><Text style={styles.badgeText}>{activeAlerts.length}</Text></View>
      ) : undefined} />

      <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {activeAlerts.length > 0 && (
          <View style={styles.section}>
            <SectionLabel label="Alertas activas en andén" />
            {activeAlerts.map(alert => (
              <AlertCard key={alert.id} alert={alert} onComplete={id => resolveIncident(id, 'completed')} onFalseAlarm={id => resolveIncident(id, 'false_alarm')} />
            ))}
          </View>
        )}

        {activeAlerts.length === 0 && (
          <View style={styles.emptyState}>
            <Ionicons name="shield-checkmark-outline" size={40} color={Palette.greenDim} />
            <Text style={styles.emptyTitle}>Sin alertas activas</Text>
            <Text style={styles.emptySub}>El sistema está monitoreando el andén.</Text>
          </View>
        )}

        {historicalAlerts.length > 0 && (
          <View style={styles.section}>
            <SectionLabel label="Historial" subtitle="Últimas 24 horas" />
            {historicalAlerts.map(alert => (
              <AlertCard key={alert.id} alert={alert} onComplete={id => resolveIncident(id, 'completed')} onFalseAlarm={id => resolveIncident(id, 'false_alarm')} />
            ))}
          </View>
        )}
        <View style={styles.bottomPad} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.bg0 },
  scroll: { flex: 1 },
  content: { padding: Space[4], gap: Space[4] },
  badge: { minWidth: 20, height: 20, borderRadius: 10, backgroundColor: Palette.red, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 5 },
  badgeText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.white },
  section: { gap: Space[2] },
  card: { backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0, borderRadius: Radius.xl, overflow: 'hidden' },
  cardHeader: { flexDirection: 'row', alignItems: 'center', padding: 14, gap: 10 },
  riskBar: { width: 3, height: 40, borderRadius: 2, flexShrink: 0 },
  cardHeaderContent: { flex: 1, gap: 3 },
  cardTitleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  alertId: { fontSize: FontSize.xxs, color: Palette.textDim, letterSpacing: 1, fontWeight: FontWeight.bold },
  alertZone: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary },
  cardMeta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 1 },
  alertTime: { fontSize: FontSize.xxs, color: Palette.textDim },
  cardBody: { borderTopWidth: 1, borderTopColor: Palette.border0, padding: 14, gap: Space[3] },
  aiReportContainer: { backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1, borderRadius: Radius.lg, padding: 12, gap: 6 },
  aiHeader: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  aiTitle: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 0.8 },
  aiDescriptionText: { fontSize: FontSize.sm, color: Palette.textSecondary, fontWeight: FontWeight.medium, lineHeight: 18 },
  infoGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  fallbackCell: { flex: 1, backgroundColor: Palette.bg2, padding: 10, borderRadius: Radius.md },
  fallbackText: { fontSize: FontSize.xs, color: Palette.textDim, fontStyle: 'italic' },
  imageContainer: { backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1, borderRadius: Radius.lg, padding: 10, gap: 6 },
  imageHeader: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  imageTitle: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 0.8 },
  screenshotImage: { width: '100%', height: 160, borderRadius: Radius.md, backgroundColor: Palette.bg3 },
  feedPlaceholder: { height: 100, backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1, borderRadius: Radius.lg, alignItems: 'center', justifyContent: 'center', gap: 5 },
  feedText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 1.5 },
  feedSub: { fontSize: FontSize.xxs, color: Palette.textDim },
  actions: { flexDirection: 'row', gap: Space[2] },
  actionBtn: { flex: 1 },
  resolvedBanner: { flexDirection: 'row', alignItems: 'center', gap: 7, backgroundColor: Palette.bg2, borderRadius: Radius.md, padding: 10 },
  resolvedText: { fontSize: FontSize.xs, fontWeight: FontWeight.semibold },
  emptyState: { alignItems: 'center', justifyContent: 'center', paddingVertical: 60, gap: Space[2] },
  emptyTitle: { fontSize: FontSize.lg, fontWeight: FontWeight.bold, color: Palette.textMuted },
  emptySub: { fontSize: FontSize.sm, color: Palette.textDim },
  bottomPad: { height: 16 },
});