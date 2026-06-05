import { ReactNode, useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';
import { Palette, FontSize, Radius, FontWeight } from '../../constants/theme';

interface AppInputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  icon?: ReactNode;
  rightIcon?: ReactNode;
  error?: string;
  editable?: boolean;
}

export function AppInput({
  label, placeholder, value, onChangeText,
  secureTextEntry = false, keyboardType = 'default',
  autoCapitalize = 'none', icon, rightIcon, error, editable = true,
}: AppInputProps) {
  const [focused, setFocused] = useState(false);

  return (
    <View style={styles.wrapper}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View
        style={[
          styles.row,
          focused && styles.rowFocused,
          !!error && styles.rowError,
          !editable && styles.rowDisabled,
        ]}
      >
        {icon && <View style={styles.iconSlot}>{icon}</View>}
        <TextInput
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          placeholderTextColor={Palette.textDim}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={editable}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={[styles.input, icon ? styles.inputWithLeft : undefined, rightIcon ? styles.inputWithRight : undefined]}
        />
        {rightIcon && <View style={styles.rightSlot}>{rightIcon}</View>}
      </View>
      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { gap: 4 },
  label: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    color: Palette.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 2,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.md,
    height: 46,
  },
  rowFocused:  { borderColor: Palette.border2 },
  rowError:    { borderColor: Palette.red },
  rowDisabled: { opacity: 0.6, backgroundColor: Palette.bg1 },
  iconSlot:    { paddingLeft: 12, justifyContent: 'center' },
  rightSlot:   { paddingRight: 12, justifyContent: 'center' },
  input: {
    flex: 1,
    color: Palette.textPrimary,
    fontSize: FontSize.sm,
    paddingHorizontal: 12,
    height: '100%',
  },
  inputWithLeft:  { paddingLeft: 8 },
  inputWithRight: { paddingRight: 8 },
  errorText: { fontSize: FontSize.xs, color: Palette.red, marginTop: 2 },
});