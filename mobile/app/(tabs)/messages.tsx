import { useState } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { TopBar } from '../../components/layout/TopBar';
import { StatusPill } from '../../components/ui/StatusPill';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { AppButton } from '../../components/ui/AppButton';
import { Palette, FontSize, FontWeight, Space, Radius, Shadow } from '../../constants/theme';
import type { IncidentAlert, AlertStatus } from '../../types/alert';

// Mock alerts
const INITIAL_ALERTS: IncidentAlert[] = [
  {
    id: 'ALT-001',
    status: 'in_progress',
    riskLevel: 'high',
    zone: 'ANDÉN - ESTACIÓN CENTRAL',
    suspect: {
      ageRange: '25 - 35 AÑOS',
      clothing: ['POLERA BLANCA', 'PANTALÓN OSCURO'],
      height: 'ALTO (APROX 1.65M)',
      sex: 'FEMENINO',
    },
    detectedAt: new Date().toISOString(),
  },
  {
    id: 'ALT-002',
    status: 'pending',
    riskLevel: 'medium',
    zone: 'ACCESO SUR - TORNIQUETE 3',
    suspect: {
      ageRange: '40 - 50 AÑOS',
      clothing: ['CHAQUETA NEGRA'],
      height: 'MEDIANO',
      sex: 'MASCULINO',
    },
    detectedAt: new Date(Date.now() - 12 * 60000).toISOString(),
  },
  {
    id: 'ALT-003',
    status: 'completed',
    riskLevel: 'low',
    zone: 'ANDÉN SUR - L2',
    suspect: {
      ageRange: '18 - 25 AÑOS',
      clothing: ['POLERA ROJA'],
      height: 'BAJO',
      sex: 'MASCULINO',
    },
    detectedAt: new Date(Date.now() - 60 * 60000).toISOString(),
  },
];

const STATUS_CONFIG: Record<AlertStatus, { label: string; pillStatus: 'active' | 'warning' | 'error' | 'neutral' | 'info' }> = {
  in_progress: { label: 'EN CURSO',      pillStatus: 'error'   },
  pending:     { label: 'PENDIENTE',     pillStatus: 'warning' },
  completed:   { label: 'COMPLETADO',    pillStatus: 'active'  },
  false_alarm: { label: 'FALSA ALARMA',  pillStatus: 'neutral' },
};

const RISK_COLORS: Record<string, string> = {
  high:   Palette.red,
  medium: Palette.amber,
  low:    Palette.green,
};

function timeAgo(iso: string) {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}min`;
  return `${Math.floor(diff / 3600)}h`;
}

function AlertCard({ alert, onComplete, onFalseAlarm }: {
  alert: IncidentAlert;
  onComplete: (id: string) => void;
  onFalseAlarm: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(alert.status === 'in_progress');
  const cfg = STATUS_CONFIG[alert.status];
  const isResolved = alert.status === 'completed' || alert.status === 'false_alarm';

  return (
    <View style={[styles.card, alert.status === 'in_progress' && Shadow.alert]}>
      {/* Header */}
      <TouchableOpacity
        style={styles.cardHeader}
        onPress={() => setExpanded(v => !v)}
        activeOpacity={0.8}
      >
        <View style={[styles.riskBar, { backgroundColor: RISK_COLORS[alert.riskLevel] }]} />
        <View style={styles.cardHeaderContent}>
          <View style={styles.cardTitleRow}>
            <Text style={styles.alertId}>{alert.id}</Text>
            <StatusPill label={cfg.label} status={cfg.pillStatus} />
          </View>
          <Text style={styles.alertZone}>{alert.zone}</Text>
          <View style={styles.cardMeta}>
            <Ionicons name="time-outline" size={11} color={Palette.textDim} />
            <Text style={styles.alertTime}>Hace {timeAgo(alert.detectedAt)}</Text>
          </View>
        </View>
        <Ionicons
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={16}
          color={Palette.textDim}
          style={{ marginLeft: 4 }}
        />
      </TouchableOpacity>

      {/* Expanded detail */}
      {expanded && (
        <View style={styles.cardBody}>
          <View style={styles.infoGrid}>
            <InfoCell icon="person-outline" label="Rango etario"  value={alert.suspect.ageRange} />
            <InfoCell icon="resize-outline" label="Estatura"      value={alert.suspect.height} />
            <InfoCell icon="male-female-outline" label="Sexo"     value={alert.suspect.sex} />
            <InfoCell
              icon="shirt-outline"
              label="Vestimenta"
              value={alert.suspect.clothing.join(' / ')}
            />
          </View>

          {/* Camera feed placeholder */}
          <View style={styles.feedPlaceholder}>
            <Ionicons name="videocam-outline" size={22} color={Palette.textDim} />
            <Text style={styles.feedText}>FEED EN VIVO</Text>
            <Text style={styles.feedSub}>Esperando transmisión de cámara…</Text>
          </View>

          {/* Actions — only if not resolved */}
          {!isResolved && (
            <View style={styles.actions}>
              <View style={styles.actionBtn}>
                <AppButton
                  label="Completado"
                  variant="primary"
                  icon="checkmark-circle-outline"
                  onPress={() => onComplete(alert.id)}
                />
              </View>
              <View style={styles.actionBtn}>
                <AppButton
                  label="Falsa alarma"
                  variant="warning"
                  icon="close-circle-outline"
                  onPress={() => onFalseAlarm(alert.id)}
                />
              </View>
            </View>
          )}

          {isResolved && (
            <View style={styles.resolvedBanner}>
              <Ionicons
                name={alert.status === 'completed' ? 'checkmark-circle' : 'close-circle'}
                size={15}
                color={alert.status === 'completed' ? Palette.green : Palette.amber}
              />
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

function InfoCell({ icon, label, value }: { icon: React.ComponentProps<typeof Ionicons>['name']; label: string; value: string }) {
  return (
    <View style={styles.infoCell}>
      <View style={styles.infoCellHeader}>
        <Ionicons name={icon} size={11} color={Palette.textDim} />
        <Text style={styles.infoCellLabel}>{label}</Text>
      </View>
      <Text style={styles.infoCellValue}>{value}</Text>
    </View>
  );
}

export default function MensajesScreen() {
  const [alerts, setAlerts] = useState<IncidentAlert[]>(INITIAL_ALERTS);

  const resolve = (id: string, status: AlertStatus) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status, respondedAt: new Date().toISOString() } : a));
  };

  const active = alerts.filter(a => a.status === 'in_progress' || a.status === 'pending');
  const history = alerts.filter(a => a.status === 'completed' || a.status === 'false_alarm');

  return (
    <View style={styles.root}>
      <TopBar
        right={
          active.length > 0 ? (
            <View style={styles.badge}>
              <Text style={styles.badgeText}>{active.length}</Text>
            </View>
          ) : undefined
        }
      />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Active alerts */}
        {active.length > 0 && (
          <View style={styles.section}>
            <SectionLabel label="Alertas activas" />
            {active.map(a => (
              <AlertCard
                key={a.id}
                alert={a}
                onComplete={id => resolve(id, 'completed')}
                onFalseAlarm={id => resolve(id, 'false_alarm')}
              />
            ))}
          </View>
        )}

        {/* Empty state */}
        {active.length === 0 && (
          <View style={styles.emptyState}>
            <Ionicons name="shield-checkmark-outline" size={40} color={Palette.greenDim} />
            <Text style={styles.emptyTitle}>Sin alertas activas</Text>
            <Text style={styles.emptySub}>El sistema está monitoreando el andén.</Text>
          </View>
        )}

        {/* History */}
        {history.length > 0 && (
          <View style={styles.section}>
            <SectionLabel label="Historial" subtitle="Últimas 24 horas" />
            {history.map(a => (
              <AlertCard
                key={a.id}
                alert={a}
                onComplete={id => resolve(id, 'completed')}
                onFalseAlarm={id => resolve(id, 'false_alarm')}
              />
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
  badge: {
    minWidth: 20, height: 20, borderRadius: 10,
    backgroundColor: Palette.red, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 5,
  },
  badgeText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.white },
  section: { gap: Space[2] },
  card: {
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.xl,
    overflow: 'hidden',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 10,
  },
  riskBar: { width: 3, height: 40, borderRadius: 2, flexShrink: 0 },
  cardHeaderContent: { flex: 1, gap: 3 },
  cardTitleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  alertId: { fontSize: FontSize.xxs, color: Palette.textDim, letterSpacing: 1, fontWeight: FontWeight.bold },
  alertZone: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary },
  cardMeta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 1 },
  alertTime: { fontSize: FontSize.xxs, color: Palette.textDim },
  cardBody: {
    borderTopWidth: 1,
    borderTopColor: Palette.border0,
    padding: 14,
    gap: Space[3],
  },
  infoGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  infoCell: {
    minWidth: '44%',
    flex: 1,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.md,
    padding: 10,
    gap: 3,
  },
  infoCellHeader: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  infoCellLabel: { fontSize: FontSize.xxs, color: Palette.textDim, textTransform: 'uppercase', letterSpacing: 0.6 },
  infoCellValue: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary },
  feedPlaceholder: {
    height: 100,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
  },
  feedText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 1.5 },
  feedSub:  { fontSize: FontSize.xxs, color: Palette.textDim },
  actions: { flexDirection: 'row', gap: Space[2] },
  actionBtn: { flex: 1 },
  resolvedBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: Palette.bg2,
    borderRadius: Radius.md,
    padding: 10,
  },
  resolvedText: { fontSize: FontSize.xs, fontWeight: FontWeight.semibold },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
    gap: Space[2],
  },
  emptyTitle: { fontSize: FontSize.lg, fontWeight: FontWeight.bold, color: Palette.textMuted },
  emptySub: { fontSize: FontSize.sm, color: Palette.textDim },
  bottomPad: { height: 16 },
});