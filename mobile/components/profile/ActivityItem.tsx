import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

type ActivityType = 'alert' | 'patrol' | 'report';

interface ActivityItemProps {
  title: string;
  description: string;
  timestamp: string;
  type: ActivityType;
}

const TYPE_COLOR: Record<ActivityType, string> = {
  alert:   Palette.red,
  patrol:  Palette.green,
  report:  Palette.textMuted,
};

export function ActivityItem({ title, description, timestamp, type }: ActivityItemProps) {
  return (
    <View style={styles.row}>
      <View style={[styles.typeDot, { backgroundColor: TYPE_COLOR[type] }]} />
      <View style={styles.content}>
        <View style={styles.topRow}>
          <Text style={styles.title} numberOfLines={1}>{title}</Text>
          <Text style={styles.timestamp}>{timestamp}</Text>
        </View>
        <Text style={styles.description} numberOfLines={2}>{description}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    gap: 12,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: Palette.border0,
  },
  typeDot: {
    width: 6,
    height: 6,
    borderRadius: Radius.full,
    marginTop: 5,
    flexShrink: 0,
  },
  content: {
    flex: 1,
    gap: 3,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  title: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Palette.textSecondary,
    flex: 1,
    marginRight: 8,
  },
  timestamp: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
  },
  description: {
    fontSize: FontSize.xs,
    color: Palette.textMuted,
    lineHeight: 16,
  },
});
