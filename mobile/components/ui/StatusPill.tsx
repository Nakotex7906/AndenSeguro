import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, LetterSpacing, Radius } from '../../constants/theme';

type Status = 'active' | 'warning' | 'error' | 'neutral' | 'info';

interface StatusPillProps {
  label: string;
  status?: Status;
}

const CONFIG: Record<Status, { bg: string; border: string; text: string; dot: string }> = {
  active:  { bg: Palette.greenBg,  border: Palette.greenDim, text: Palette.green,      dot: Palette.green   },
  warning: { bg: Palette.amberBg,  border: Palette.amberDim, text: Palette.amber,      dot: Palette.amber   },
  error:   { bg: Palette.redBg,    border: Palette.redDim,   text: Palette.red,        dot: Palette.red     },
  info:    { bg: Palette.blueBg,   border: Palette.blueDim,  text: Palette.blue,       dot: Palette.blue    },
  neutral: { bg: Palette.bg2,      border: Palette.border1,  text: Palette.textMuted,  dot: Palette.textDim },
};

export function StatusPill({ label, status = 'neutral' }: { label: string; status?: Status }) {
  const c = CONFIG[status];
  return (
    <View style={[styles.pill, { backgroundColor: c.bg, borderColor: c.border }]}>
      <View style={[styles.dot, { backgroundColor: c.dot }]} />
      <Text style={[styles.label, { color: c.text }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: Radius.full,
    borderWidth: 1,
  },
  dot: { width: 5, height: 5, borderRadius: Radius.full },
  label: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    letterSpacing: LetterSpacing.wider,
    textTransform: 'uppercase',
  },
});