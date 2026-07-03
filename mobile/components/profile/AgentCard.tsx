import { Image, StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius, Shadow } from '../../constants/theme';
import { StatusPill } from '../ui/StatusPill';

interface AgentCardProps {
  name: string;
  badge: string;
  assignment: string;
  photoUrl?: string;
  status: 'on_duty' | 'off_duty' | 'on_break';
}

const STATUS_LABELS: Record<AgentCardProps['status'], { label: string; status: 'active' | 'warning' | 'neutral' }> = {
  on_duty:   { label: 'EN SERVICIO', status: 'active'  },
  off_duty:  { label: 'FUERA DE SERVICIO', status: 'neutral' },
  on_break:  { label: 'EN PAUSA', status: 'warning' },
};

export function AgentCard({ name, badge, assignment, photoUrl, status }: AgentCardProps) {
  const s = STATUS_LABELS[status];

  return (
    <View style={[styles.card, Shadow.card]}>
      {/* Header row: ID badge + status pill */}
      <View style={styles.headerRow}>
        <View style={styles.idBadge}>
          <Text style={styles.idLabel}>IDENTIFICACIÓN OPERATIVA</Text>
        </View>
        <StatusPill label={s.label} status={s.status} />
      </View>

      {/* Agent name */}
      <Text style={styles.name}>{name}</Text>
      <Text style={styles.badgeText}>ID: {badge}</Text>

      {/* Photo */}
      <View style={styles.photoWrap}>
        {photoUrl ? (
          <Image source={{ uri: photoUrl }} style={styles.photo} resizeMode="cover" />
        ) : (
          <View style={styles.photoPlaceholder}>
            {/* Silhouette placeholder */}
            <Text style={styles.photoPlaceholderText}>👮</Text>
          </View>
        )}
      </View>

      {/* Assignment */}
      <View style={styles.assignmentRow}>
        <Text style={styles.assignmentLabel}>ASIGNACIÓN</Text>
        <Text style={styles.assignmentValue}>{assignment}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.xl,
    padding: 16,
    gap: 8,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  idBadge: {
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  idLabel: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    letterSpacing: 0.8,
    color: Palette.textDim,
    textTransform: 'uppercase',
  },
  name: {
    fontSize: FontSize.xl,
    fontWeight: FontWeight.bold,
    color: Palette.textPrimary,
    letterSpacing: -0.3,
    marginTop: 4,
  },
  badgeText: {
    fontSize: FontSize.xs,
    color: Palette.textDim,
    letterSpacing: 1,
  },
  photoWrap: {
    marginVertical: 8,
    borderRadius: Radius.lg,
    overflow: 'hidden',
    backgroundColor: Palette.bg2,
    height: 180,
    borderWidth: 1,
    borderColor: Palette.border1,
  },
  photo: {
    width: '100%',
    height: '100%',
  },
  photoPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  photoPlaceholderText: {
    fontSize: 64,
  },
  assignmentRow: {
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.md,
    padding: 12,
    gap: 3,
  },
  assignmentLabel: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    letterSpacing: 1.2,
    color: Palette.textDim,
    textTransform: 'uppercase',
  },
  assignmentValue: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Palette.textSecondary,
  },
});
