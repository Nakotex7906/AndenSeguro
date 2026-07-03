import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

interface SettingNavRowProps {
  icon: string;
  title: string;
  subtitle?: string;
  onPress?: () => void;
}

export function SettingNavRow({ icon, title, subtitle, onPress }: SettingNavRowProps) {
  return (
    <TouchableOpacity
      onPress={onPress}
      activeOpacity={0.7}
      style={styles.row}
    >
      <View style={styles.iconWrap}>
        <Text style={styles.icon}>{icon}</Text>
      </View>
      <View style={styles.text}>
        <Text style={styles.title}>{title}</Text>
        {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
      </View>
      {/* Chevron */}
      <Text style={styles.chevron}>›</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.lg,
    padding: 14,
  },
  iconWrap: {
    width: 32,
    height: 32,
    borderRadius: Radius.md,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: { fontSize: 15 },
  text: {
    flex: 1,
    gap: 2,
  },
  title: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Palette.textSecondary,
  },
  subtitle: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
  },
  chevron: {
    fontSize: 20,
    color: Palette.textDim,
    lineHeight: 22,
  },
});