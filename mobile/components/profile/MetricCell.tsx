import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

interface MetricCellProps {
  value: string;
  label: string;
  sublabel?: string;
  valueColor?: string;
}

export function MetricCell({
  value,
  label,
  sublabel,
  valueColor = Palette.textPrimary,
}: MetricCellProps) {
  return (
    <View style={styles.cell}>
      <Text style={[styles.value, { color: valueColor }]}>{value}</Text>
      <Text style={styles.label}>{label}</Text>
      {sublabel && <Text style={styles.sublabel}>{sublabel}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  cell: {
    flex: 1,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.lg,
    padding: 12,
    gap: 2,
  },
  value: {
    fontSize: FontSize['2xl'],
    fontWeight: FontWeight.bold,
    letterSpacing: -0.5,
  },
  label: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.semibold,
    color: Palette.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  sublabel: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
    marginTop: 1,
  },
});
