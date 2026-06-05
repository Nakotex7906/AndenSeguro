import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';
import { Toggle } from '../../components/ui/Toggle';

interface SettingRowProps {
  icon: string;
  title: string;
  subtitle?: string;
  value: boolean;
  onValueChange: (v: boolean) => void;
  toggleColor?: string;
}

export function SettingRow({
  icon,
  title,
  subtitle,
  value,
  onValueChange,
  toggleColor,
}: SettingRowProps) {
  return (
    <View style={styles.row}>
      <View style={styles.iconWrap}>
        <Text style={styles.icon}>{icon}</Text>
      </View>
      <View style={styles.text}>
        <Text style={styles.title}>{title}</Text>
        {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
      </View>
      <Toggle value={value} onValueChange={onValueChange} color={toggleColor} />
    </View>
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
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
});
