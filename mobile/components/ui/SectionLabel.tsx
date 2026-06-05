import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, LetterSpacing } from '../../constants/theme';

interface SectionLabelProps {
  label: string;
  subtitle?: string;
  /** Right-side action slot */
  action?: React.ReactNode;
}

export function SectionLabel({ label, subtitle, action }: SectionLabelProps) {
  return (
    <View style={styles.row}>
      <View style={styles.left}>
        <View style={styles.bar} />
        <View>
          <Text style={styles.label}>{label}</Text>
          {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
        </View>
      </View>
      {action}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  bar: {
    width: 3,
    height: 16,
    borderRadius: 2,
    backgroundColor: Palette.green,
  },
  label: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.bold,
    letterSpacing: LetterSpacing.wider,
    color: Palette.textSecondary,
    textTransform: 'uppercase',
  },
  subtitle: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
    marginTop: 1,
  },
});
