import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius, Shadow } from '../../constants/theme';

interface AlertBannerProps {
  status: string;       // "ESTADO: EN CURSO"
  title: string;        // "SUJETO EN RIESGO DETECTADO"
}

export function AlertBanner({ status, title }: AlertBannerProps) {
  return (
    <View style={[styles.banner, Shadow.alert]}>
      <Text style={styles.status}>{status}</Text>
      <Text style={styles.title}>{title}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: Palette.redBg,
    borderWidth: 1,
    borderColor: Palette.redDim,
    borderRadius: Radius.xl,
    padding: 16,
    gap: 4,
  },
  status: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    letterSpacing: 1.5,
    color: Palette.red,
    textTransform: 'uppercase',
  },
  title: {
    fontSize: FontSize.xl,
    fontWeight: FontWeight.extrabold,
    color: Palette.textPrimary,
    letterSpacing: -0.3,
    lineHeight: 26,
  },
});