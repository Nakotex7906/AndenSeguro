import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

type Variant = 'primary' | 'danger' | 'ghost' | 'outline' | 'warning';
type IoniconName = React.ComponentProps<typeof Ionicons>['name'];

interface AppButtonProps {
  label: string;
  onPress?: () => void;
  variant?: Variant;
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  icon?: IoniconName | React.ReactNode;
}

const VARIANTS: Record<Variant, { bg: string; border: string; text: string; disabledBg: string }> = {
  primary: { bg: Palette.green,  border: Palette.green,   text: '#0a0f0a',             disabledBg: Palette.bg3 },
  danger:  { bg: Palette.red,    border: Palette.red,     text: Palette.white,         disabledBg: Palette.bg3 },
  warning: { bg: Palette.amber,  border: Palette.amber,   text: '#0a0f0a',             disabledBg: Palette.bg3 },
  ghost:   { bg: 'transparent',  border: 'transparent',   text: Palette.textMuted,     disabledBg: 'transparent' },
  outline: { bg: 'transparent',  border: Palette.border1, text: Palette.textSecondary, disabledBg: 'transparent' },
};

export function AppButton({
  label, onPress, variant = 'primary', loading = false,
  disabled = false, fullWidth = true, icon,
}: AppButtonProps) {
  const v = VARIANTS[variant];
  const isDisabled = disabled || loading;
  const textColor = isDisabled ? Palette.textDim : v.text;

  const renderIcon = () => {
    if (!icon) return null;
    // Si es string, es un nombre de Ionicon
    if (typeof icon === 'string') {
      return <Ionicons name={icon as IoniconName} size={15} color={textColor} />;
    }
    // Si es ReactNode (compatibilidad con el uso antiguo)
    return <View style={styles.iconWrap}>{icon}</View>;
  };

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.75}
      style={[
        styles.btn,
        {
          backgroundColor: isDisabled ? v.disabledBg : v.bg,
          borderColor: isDisabled ? Palette.border0 : v.border,
          alignSelf: fullWidth ? 'stretch' : 'flex-start',
        },
      ]}
    >
      {loading ? (
        <ActivityIndicator size={14} color={textColor} />
      ) : (
        <View style={styles.inner}>
          {renderIcon()}
          <Text style={[styles.label, { color: textColor }]}>{label}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  btn: {
    height: 44,
    borderRadius: Radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  inner: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  iconWrap: { alignItems: 'center', justifyContent: 'center' },
  label: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.bold,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
});