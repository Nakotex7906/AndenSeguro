import { useState } from 'react';
import {
  Image, Modal, ScrollView, StyleSheet, Text,
  TouchableOpacity, View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ReactNativeZoomableView } from '@openspacelabs/react-native-zoomable-view';
import { TopBar } from '../../components/layout/TopBar';
import { StatusPill } from '../../components/ui/StatusPill';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { useAuth } from '../../store/auth';
import { Palette, FontSize, FontWeight, Space, Radius } from '../../constants/theme';

const PROTOCOLS = [
  {
    id: '1', code: 'PRO-001',
    title: 'Despliegue e Intervención Inmediata',
    level: 'high' as const, active: true,
    steps: [
      'Acudir de forma inmediata a la ubicación exacta reportada por el monitor.',
      'Delimitar un perímetro de seguridad para proteger a la persona y transeúntes.',
      'Identificar y alejar de forma segura objetos de riesgo (cortopunzantes, cuerdas, medicamentos) o alejar a la persona de zonas de peligro.',
    ],
  },
  {
    id: '2', code: 'PRO-002',
    title: 'Primeros Auxilios Psicológicos (ABCDE)',
    level: 'high' as const, active: false,
    steps: [
      'A – Escucha Activa: Mantener la calma, invitar a hablar en espacio privado, escuchar sin interrumpir.',
      'B – Reentrenamiento de Ventilación: Guiar respiración pausada (inhalar/exhalar en tiempos de 4 seg).',
      'C – Categorización de Necesidades: Ayudar a identificar problemas inmediatos y abordarlos paso a paso.',
      'D – Derivación a Redes: Facilitar contacto con red de apoyo o servicios de salud.',
      'E – Psicoeducación: Explicar que las reacciones (angustia, llanto, confusión) son normales ante una crisis.',
    ],
  },
  {
    id: '3', code: 'PRO-003',
    title: 'Resguardo y Acompañamiento Constante',
    level: 'medium' as const, active: false,
    steps: [
      'No dejar sola a la persona bajo ninguna circunstancia hasta que llegue personal de salud o fuerza pública.',
      'Si llega unidad de primeros auxilios, facilitar su ingreso y continuar apoyando en el perímetro.',
    ],
  },
  {
    id: '4', code: 'PRO-004',
    title: 'Gestión con Servicios de Emergencia Externa',
    level: 'medium' as const, active: false,
    steps: [
      'Al llegar Carabineros, SAMU o Bomberos, entregar responsabilidad informando lo observado y acciones realizadas.',
      'Si se requiere traslado, colaborar en la coordinación del acceso de la ambulancia al recinto.',
    ],
  },
  {
    id: '5', code: 'PRO-005',
    title: 'Registro y Documentación',
    level: 'low' as const, active: false,
    steps: [
      'Si la persona rechaza atención, asegurar firma del documento de "Delegación de Responsabilidades".',
      'Informar cierre de la intervención a la jefatura para los registros epidemiológicos y de seguridad interna.',
    ],
  },
];

const LEVEL_COLORS: Record<string, string> = {
  high: Palette.red, medium: Palette.amber, low: Palette.green,
};

const QUICK_STATS = [
  { icon: 'shield-checkmark-outline' as const, label: 'Turno activo', value: 'Mañana', color: Palette.green },
  { icon: 'time-outline' as const,             label: 'Inicio turno', value: '06:00',  color: Palette.textSecondary },
  { icon: 'alert-circle-outline' as const,     label: 'Alertas hoy',  value: '3',      color: Palette.amber },
  { icon: 'checkmark-circle-outline' as const, label: 'Completadas',  value: '2',      color: Palette.green },
];

function MapModal({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={modal.overlay}>
        <View style={modal.container}>
          <TouchableOpacity style={modal.closeBtn} onPress={onClose}>
            <Ionicons name="close" size={22} color={Palette.textPrimary} />
          </TouchableOpacity>
          <ReactNativeZoomableView
            maxZoom={4}
            minZoom={0.5}
            initialZoom={1}
            style={modal.zoomView}
          >
            <Image
              source={require('../../assets/images/mapa.png')}
              style={modal.mapImage}
              resizeMode="contain"
            />
          </ReactNativeZoomableView>
        </View>
      </View>
    </Modal>
  );
}

function ProtocolRow({ protocol, index }: { protocol: typeof PROTOCOLS[0]; index: number }) {
  return (
    <View style={styles.protocolRow}>
      <View style={[styles.protocolIndex, { borderColor: LEVEL_COLORS[protocol.level] }]}>
        <Text style={[styles.protocolIndexText, { color: LEVEL_COLORS[protocol.level] }]}>
          {index + 1}
        </Text>
      </View>
      <View style={styles.protocolInfo}>
        <Text style={styles.protocolCode}>{protocol.code}</Text>
        <Text style={styles.protocolTitle}>{protocol.title}</Text>
        {protocol.steps.map((step, i) => (
          <View key={i} style={styles.stepRow}>
            <View style={[styles.stepDot, { backgroundColor: LEVEL_COLORS[protocol.level] }]} />
            <Text style={styles.stepText}>{step}</Text>
          </View>
        ))}
      </View>
      {protocol.active && <StatusPill label="ACTIVO" status="active" />}
    </View>
  );
}

export default function InicioScreen() {
  const { user } = useAuth();
  const [mapVisible, setMapVisible] = useState(false);

  return (
    <View style={styles.root}>
      <TopBar right={<StatusPill label="EN SERVICIO" status="active" />} />
      <MapModal visible={mapVisible} onClose={() => setMapVisible(false)} />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[styles.content, { paddingBottom: 96 }]}
        showsVerticalScrollIndicator={false}
        onScroll={(e) => {
          const handler = (global as any).__tabScrollHandler;
          if (handler) handler(e.nativeEvent.contentOffset.y);
        }}
        scrollEventThrottle={16}
      >
        {/* Welcome */}
        <View style={styles.welcomeRow}>
          <View>
            <Text style={styles.welcomeLabel}>Bienvenido,</Text>
            <Text style={styles.welcomeName}>{user?.name ?? 'Agente'}</Text>
          </View>
          <View style={styles.badgeChip}>
            <Ionicons name="card-outline" size={12} color={Palette.textDim} />
            <Text style={styles.badgeText}>{user?.badge ?? '—'}</Text>
          </View>
        </View>

        {/* Stats */}
        <View style={styles.statsGrid}>
          {QUICK_STATS.map(s => (
            <View key={s.label} style={styles.statCell}>
              <Ionicons name={s.icon} size={18} color={s.color} />
              <Text style={[styles.statValue, { color: s.color }]}>{s.value}</Text>
              <Text style={styles.statLabel}>{s.label}</Text>
            </View>
          ))}
        </View>

        {/* Mapa — toca para zoom */}
        <View style={styles.section}>
          <SectionLabel label="Mapa de vías" subtitle="Red de Metro de Santiago" />
          <TouchableOpacity
            style={styles.mapWrap}
            onPress={() => setMapVisible(true)}
            activeOpacity={0.85}
          >
            <Image
              source={require('../../assets/images/mapa.png')}
              style={styles.mapImage}
              resizeMode="contain"
            />
            <View style={styles.mapHint}>
              <Ionicons name="expand-outline" size={13} color={Palette.textDim} />
              <Text style={styles.mapHintText}>Toca para ampliar</Text>
            </View>
          </TouchableOpacity>
        </View>

        {/* Protocolos */}
        <View style={styles.section}>
          <SectionLabel label="Protocolos operativos" subtitle="Estrategia Nacional de Prevención" />
          <View style={styles.protocolList}>
            {PROTOCOLS.map((p, idx) => (
              <ProtocolRow key={p.id} protocol={p} index={idx} />
            ))}
          </View>
        </View>

        {/* Zona asignada */}
        <View style={styles.assignCard}>
          <Ionicons name="navigate-circle-outline" size={18} color={Palette.green} />
          <View style={{ flex: 1 }}>
            <Text style={styles.assignLabel}>Zona asignada</Text>
            <Text style={styles.assignValue}>{user?.assignment ?? 'Sin asignación'}</Text>
          </View>
          <View style={styles.assignBadge}>
            <Text style={styles.assignBadgeText}>ACTIVO</Text>
          </View>
        </View>

        <View style={styles.bottomPad} />
      </ScrollView>
    </View>
  );
}

const modal = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.92)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    width: '95%',
    height: '85%',
    backgroundColor: Palette.bg1,
    borderRadius: Radius.xl,
    overflow: 'hidden',
  },
  closeBtn: {
    position: 'absolute', top: 12, right: 12, zIndex: 10,
    width: 34, height: 34, borderRadius: 17,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center', justifyContent: 'center',
  },
  zoomView: { flex: 1 },
  mapImage: { width: '100%', height: '100%' },
});

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.bg0 },
  scroll: { flex: 1 },
  content: { padding: Space[4], gap: Space[4] },
  welcomeRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  welcomeLabel: { fontSize: FontSize.xs, color: Palette.textDim },
  welcomeName: { fontSize: FontSize.xl, fontWeight: FontWeight.bold, color: Palette.textPrimary, marginTop: 2 },
  badgeChip: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1,
    borderRadius: Radius.full, paddingHorizontal: 10, paddingVertical: 5,
  },
  badgeText: { fontSize: FontSize.xxs, color: Palette.textDim, letterSpacing: 0.8 },
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: Space[2] },
  statCell: {
    flex: 1, minWidth: '44%',
    backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0,
    borderRadius: Radius.lg, padding: 12, gap: 4, alignItems: 'flex-start',
  },
  statValue: { fontSize: FontSize['2xl'], fontWeight: FontWeight.bold, letterSpacing: -0.5 },
  statLabel: { fontSize: FontSize.xxs, color: Palette.textDim, textTransform: 'uppercase', letterSpacing: 0.6 },
  section: { gap: Space[2] },
  mapWrap: {
    borderRadius: Radius.xl, overflow: 'hidden',
    backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0,
  },
  mapImage: { width: '100%', height: 220 },
  mapHint: {
    position: 'absolute', bottom: 8, right: 10,
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: Radius.sm,
    paddingHorizontal: 8, paddingVertical: 3,
  },
  mapHintText: { fontSize: FontSize.xxs, color: Palette.textDim },
  protocolList: { gap: Space[3] },
  protocolRow: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 12,
    backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0,
    borderRadius: Radius.lg, padding: 14,
  },
  protocolIndex: {
    width: 28, height: 28, borderRadius: Radius.sm, borderWidth: 1,
    alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 2,
  },
  protocolIndexText: { fontSize: FontSize.sm, fontWeight: FontWeight.bold },
  protocolInfo: { flex: 1, gap: 6 },
  protocolCode: { fontSize: FontSize.xxs, color: Palette.textDim, letterSpacing: 1, textTransform: 'uppercase' },
  protocolTitle: { fontSize: FontSize.sm, fontWeight: FontWeight.bold, color: Palette.textSecondary },
  stepRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  stepDot: { width: 5, height: 5, borderRadius: 3, marginTop: 5, flexShrink: 0 },
  stepText: { fontSize: FontSize.xs, color: Palette.textMuted, lineHeight: 18, flex: 1 },
  assignCard: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: Palette.greenBg, borderWidth: 1, borderColor: Palette.greenDim,
    borderRadius: Radius.lg, padding: 14,
  },
  assignLabel: { fontSize: FontSize.xxs, color: Palette.green, textTransform: 'uppercase', letterSpacing: 0.8 },
  assignValue: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textPrimary, marginTop: 2 },
  assignBadge: { backgroundColor: Palette.green, borderRadius: Radius.sm, paddingHorizontal: 8, paddingVertical: 3 },
  assignBadgeText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: '#0a0f0a', letterSpacing: 0.8 },
  bottomPad: { height: 16 },
});