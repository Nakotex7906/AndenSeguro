import { useState } from 'react';
import {
  KeyboardAvoidingView, Platform, ScrollView,
  StyleSheet, Text, TouchableOpacity, View, ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../../store/auth';
import { Palette, FontSize, FontWeight, Space, Radius, LetterSpacing } from '../../constants/theme';
import { AppInput } from '../../components/ui/AppInput';

export default function LoginScreen() {
  const { login, isLoading, error, clearError } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) return;
    clearError();
    await login({ username: username.trim(), password });
  };

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        style={styles.kav}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Logo / Brand */}
          <View style={styles.brand}>
            <View style={styles.logoWrap}>
              <Ionicons name="shield-checkmark" size={36} color={Palette.green} />
            </View>
            <Text style={styles.appName}>ANDÉN SEGURO</Text>
            <Text style={styles.appSub}>Sistema de Vigilancia Operativa</Text>
          </View>

          {/* Card */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Acceso al sistema</Text>
            <Text style={styles.cardSub}>Ingresa tus credenciales operativas</Text>

            {/* Error */}
            {error && (
              <View style={styles.errorBanner}>
                <Ionicons name="alert-circle" size={14} color={Palette.red} />
                <Text style={styles.errorText}>{error}</Text>
              </View>
            )}

            {/* Username */}
            <AppInput
              label="Usuario"
              placeholder="ej: agente.essus"
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
              icon={<Ionicons name="person-outline" size={16} color={Palette.textDim} />}
            />

            {/* Password */}
            <AppInput
              label="Contraseña"
              placeholder="••••••••"
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPassword}
              icon={<Ionicons name="lock-closed-outline" size={16} color={Palette.textDim} />}
              rightIcon={
                <TouchableOpacity onPress={() => setShowPassword(v => !v)} hitSlop={8}>
                  <Ionicons
                    name={showPassword ? 'eye-off-outline' : 'eye-outline'}
                    size={16}
                    color={Palette.textDim}
                  />
                </TouchableOpacity>
              }
            />

            {/* Submit */}
            <TouchableOpacity
              style={[styles.btn, (!username || !password || isLoading) && styles.btnDisabled]}
              onPress={handleLogin}
              disabled={!username || !password || isLoading}
              activeOpacity={0.8}
            >
              {isLoading ? (
                <ActivityIndicator size="small" color="#0a0f0a" />
              ) : (
                <>
                  <Ionicons name="log-in-outline" size={16} color="#0a0f0a" />
                  <Text style={styles.btnText}>INGRESAR</Text>
                </>
              )}
            </TouchableOpacity>

            {/* Hint */}
            <Text style={styles.hint}>Demo: agente.essus / 1234</Text>
          </View>

          {/* Footer */}
          <Text style={styles.footer}>Andén Seguro © 2026 — v1.0.0</Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Palette.bg0 },
  kav:  { flex: 1 },
  scroll: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: Space[4],
    gap: Space[6],
  },
  brand: { alignItems: 'center', gap: Space[2] },
  logoWrap: {
    width: 72,
    height: 72,
    borderRadius: Radius.xl,
    backgroundColor: Palette.greenBg,
    borderWidth: 1,
    borderColor: Palette.greenDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Space[1],
  },
  appName: {
    fontSize: FontSize['2xl'],
    fontWeight: FontWeight.extrabold,
    color: Palette.textPrimary,
    letterSpacing: LetterSpacing.widest,
  },
  appSub: {
    fontSize: FontSize.xs,
    color: Palette.textDim,
    letterSpacing: LetterSpacing.wide,
    textTransform: 'uppercase',
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  card: {
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.xl,
    padding: Space[4],
    gap: Space[3],
  },
  cardTitle: {
    fontSize: FontSize.lg,
    fontWeight: FontWeight.bold,
    color: Palette.textPrimary,
  },
  cardSub: {
    fontSize: FontSize.xs,
    color: Palette.textDim,
    marginTop: -Space[1],
    marginBottom: Space[1],
  },
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Palette.redBg,
    borderWidth: 1,
    borderColor: Palette.redDim,
    borderRadius: Radius.md,
    padding: 10,
  },
  errorText: { fontSize: FontSize.xs, color: Palette.red, flex: 1 },
  btn: {
    height: 48,
    backgroundColor: Palette.green,
    borderRadius: Radius.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: Space[1],
  },
  btnDisabled: { opacity: 0.45 },
  btnText: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.bold,
    color: '#0a0f0a',
    letterSpacing: 1,
  },
  hint: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
    textAlign: 'center',
    letterSpacing: 0.5,
  },
  footer: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
    textAlign: 'center',
    letterSpacing: 0.5,
  },
});