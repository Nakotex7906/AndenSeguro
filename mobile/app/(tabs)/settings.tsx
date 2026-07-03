import { useState } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import Slider from '@react-native-community/slider';
import * as Brightness from 'expo-brightness';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { TopBar } from '../../components/layout/TopBar';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { Toggle } from '../../components/ui/Toggle';
import { AppButton } from '../../components/ui/AppButton';
import { useAuth } from '../../store/auth';
import { Palette, FontSize, FontWeight, Space, Radius } from '../../constants/theme';

function SettingRow({
  icon, title, subtitle, value, onValueChange, color,
}: {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  title: string; subtitle?: string;
  value: boolean; onValueChange: (v: boolean) => void; color?: string;
}) {
  return (
    <View style={styles.settingRow}>
      <View style={styles.settingIcon}>
        <Ionicons name={icon} size={16} color={Palette.textMuted} />
      </View>
      <View style={styles.settingText}>
        <Text style={styles.settingTitle}>{title}</Text>
        {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
      </View>
      <Toggle value={value} onValueChange={onValueChange} color={color} />
    </View>
  );
}

function NavRow({
  icon, title, subtitle, onPress,
}: {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  title: string; subtitle?: string; onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.settingRow} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.settingIcon}>
        <Ionicons name={icon} size={16} color={Palette.textMuted} />
      </View>
      <View style={styles.settingText}>
        <Text style={styles.settingTitle}>{title}</Text>
        {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
      </View>
      <Ionicons name="chevron-forward" size={16} color={Palette.textDim} />
    </TouchableOpacity>
  );
}

export default function AjustesScreen() {
  const { logout, user } = useAuth();
  const router = useRouter();
  const [notifications, setNotifications] = useState(true);
  const [gps, setGps] = useState(true);
  const [vibration, setVibration] = useState(true);
  const [sound, setSound] = useState(true);
  const [autoReport, setAutoReport] = useState(false);
  const [brightness, setBrightness] = useState(0.8);

  const handleBrightness = async (val: number) => {
    setBrightness(val);
    try {
      const { status } = await Brightness.requestPermissionsAsync();
      if (status === 'granted') {
        await Brightness.setSystemBrightnessAsync(val);
      }
    } catch (_) {}
  };

  return (
    <View style={styles.root}>
      <TopBar />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <SectionLabel label="Ajustes" subtitle="Configuración del sistema" />

        {/* Perfil — toca para abrir */}
        <TouchableOpacity
          style={styles.accountCard}
          onPress={() => router.push('/(tabs)/profile')}
          activeOpacity={0.75}
        >
          <View style={styles.accountAvatar}>
            <Ionicons name="person" size={22} color={Palette.textDim} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.accountName}>{user?.name}</Text>
            <Text style={styles.accountBadge}>{user?.badge} · {user?.role}</Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={Palette.textDim} />
        </TouchableOpacity>

        {/* Brillo */}
        <View style={styles.group}>
          <Text style={styles.groupTitle}>Pantalla</Text>
          <View style={styles.groupCard}>
            <View style={styles.brightnessRow}>
              <View style={styles.settingIcon}>
                <Ionicons name="sunny-outline" size={16} color={Palette.textMuted} />
              </View>
              <View style={styles.settingText}>
                <Text style={styles.settingTitle}>Brillo</Text>
                <Text style={styles.settingSubtitle}>{Math.round(brightness * 100)}%</Text>
              </View>
              <Ionicons name="sunny" size={16} color={Palette.amber} />
            </View>
            <Slider
              style={styles.slider}
              minimumValue={0.05}
              maximumValue={1}
              value={brightness}
              onValueChange={handleBrightness}
              minimumTrackTintColor={Palette.amber}
              maximumTrackTintColor={Palette.border1}
              thumbTintColor={Palette.amber}
            />
          </View>
        </View>

        {/* Notificaciones */}
        <View style={styles.group}>
          <Text style={styles.groupTitle}>Notificaciones</Text>
          <View style={styles.groupCard}>
            <SettingRow
              icon="notifications-outline" title="Alertas de intrusión"
              subtitle="Notificaciones de prioridad crítica"
              value={notifications} onValueChange={setNotifications} color={Palette.blue}
            />
            <View style={styles.divider} />
            <SettingRow
              icon="volume-high-outline" title="Sonido de alerta"
              subtitle="Reproducir audio al recibir alerta"
              value={sound} onValueChange={setSound} color={Palette.blue}
            />
            <View style={styles.divider} />
            <SettingRow
              icon="phone-portrait-outline" title="Vibración"
              subtitle="Vibrar al recibir notificación"
              value={vibration} onValueChange={setVibration} color={Palette.blue}
            />
          </View>
        </View>

        {/* Operativo */}
        <View style={styles.group}>
          <Text style={styles.groupTitle}>Operativo</Text>
          <View style={styles.groupCard}>
            <SettingRow
              icon="location-outline" title="GPS"
              subtitle="Mantener siempre activo"
              value={gps} onValueChange={setGps} color={Palette.amber}
            />
            <View style={styles.divider} />
            <SettingRow
              icon="document-text-outline" title="Reporte automático"
              subtitle="Generar reporte al cerrar turno"
              value={autoReport} onValueChange={setAutoReport} color={Palette.green}
            />
          </View>
        </View>

        {/* Sistema */}
        <View style={styles.group}>
          <Text style={styles.groupTitle}>Sistema</Text>
          <View style={styles.groupCard}>
            <NavRow
              icon="document-lock-outline" title="Protocolos"
              subtitle="Ver manual de procedimientos"
              onPress={() => {}}
            />
            <View style={styles.divider} />
            <NavRow
              icon="id-card-outline" title="Credenciales digitales"
              subtitle="Gestionar certificados operativos"
              onPress={() => {}}
            />
            <View style={styles.divider} />
            <NavRow icon="help-circle-outline" title="Soporte técnico" onPress={() => {}} />
            <View style={styles.divider} />
            <NavRow
              icon="information-circle-outline" title="Versión de la app"
              subtitle="v1.0.0 · Andén Seguro"
              onPress={() => {}}
            />
          </View>
        </View>

        <AppButton label="Cerrar sesión" variant="danger" icon="log-out-outline" onPress={logout} />
        <View style={styles.bottomPad} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.bg0 },
  scroll: { flex: 1 },
  content: { padding: Space[4], gap: Space[4] },
  accountCard: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0,
    borderRadius: Radius.lg, padding: 14,
  },
  accountAvatar: {
    width: 42, height: 42, borderRadius: 21,
    backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1,
    alignItems: 'center', justifyContent: 'center',
  },
  accountName: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textPrimary },
  accountBadge: { fontSize: FontSize.xxs, color: Palette.textDim, marginTop: 2, textTransform: 'uppercase', letterSpacing: 0.5 },
  group: { gap: Space[1] },
  groupTitle: {
    fontSize: FontSize.xxs, fontWeight: FontWeight.bold,
    color: Palette.textDim, textTransform: 'uppercase', letterSpacing: 1, paddingLeft: 4,
  },
  groupCard: {
    backgroundColor: Palette.bg1, borderWidth: 1, borderColor: Palette.border0,
    borderRadius: Radius.lg, overflow: 'hidden',
  },
  brightnessRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 14, paddingBottom: 4 },
  slider: { marginHorizontal: 14, marginBottom: 10 },
  settingRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 14 },
  settingIcon: {
    width: 30, height: 30, borderRadius: Radius.sm,
    backgroundColor: Palette.bg2, borderWidth: 1, borderColor: Palette.border1,
    alignItems: 'center', justifyContent: 'center',
  },
  settingText: { flex: 1, gap: 2 },
  settingTitle: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary },
  settingSubtitle: { fontSize: FontSize.xxs, color: Palette.textDim },
  divider: { height: 1, backgroundColor: Palette.border0, marginLeft: 14 },
  bottomPad: { height: 16 },
});