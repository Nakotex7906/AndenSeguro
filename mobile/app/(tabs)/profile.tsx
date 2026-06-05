import { useState } from 'react';
import {
  Alert, Image, ScrollView, StyleSheet, Text,
  TouchableOpacity, View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { TopBar } from '../../components/layout/TopBar';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { AppInput } from '../../components/ui/AppInput';
import { AppButton } from '../../components/ui/AppButton';
import { StatusPill } from '../../components/ui/StatusPill';
import { useAuth } from '../../store/auth';
import { Palette, FontSize, FontWeight, Space, Radius, Shadow } from '../../constants/theme';

const ACTIVITY = [
  { id: '1', type: 'alert'  as const, title: 'Alerta Crítica Atendida',  desc: 'Protocolo ejecutado con éxito en Estación Central.', time: '14:22' },
  { id: '2', type: 'patrol' as const, title: 'Patrullaje Preventivo',     desc: 'Recorrido perimetral completado en Andén Sur.',       time: '13:05' },
  { id: '3', type: 'report' as const, title: 'Reporte Enviado',           desc: 'Informe de turno transmitido al supervisor.',         time: '11:40' },
];

const TYPE_ICON: Record<string, React.ComponentProps<typeof Ionicons>['name']> = {
  alert:  'alert-circle',
  patrol: 'walk',
  report: 'document-text',
};
const TYPE_COLOR: Record<string, string> = {
  alert:  Palette.red,
  patrol: Palette.green,
  report: Palette.textMuted,
};

export default function PerfilScreen() {
  const { user, updateProfile, logout } = useAuth();

  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(user?.name ?? '');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [photoUri, setPhotoUri] = useState<string | undefined>(user?.photoUri);
  const [saving, setSaving] = useState(false);

  const pickPhoto = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permiso requerido', 'Se necesita acceso a la galería para cambiar la foto.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      setPhotoUri(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permiso requerido', 'Se necesita acceso a la cámara.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      setPhotoUri(result.assets[0].uri);
    }
  };

  const handlePhotoPress = () => {
    Alert.alert('Cambiar foto', 'Selecciona una opción', [
      { text: 'Cámara',   onPress: takePhoto  },
      { text: 'Galería',  onPress: pickPhoto  },
      { text: 'Cancelar', style: 'cancel'     },
    ]);
  };

  const handleSave = async () => {
    setSaving(true);
    await new Promise(r => setTimeout(r, 600));
    updateProfile({ name: name.trim(), photoUri });
    setSaving(false);
    setEditing(false);
  };

  const handleCancel = () => {
    setName(user?.name ?? '');
    setPhotoUri(user?.photoUri);
    setEditing(false);
  };

  return (
    <View style={styles.root}>
      <TopBar
        right={
          !editing ? (
            <TouchableOpacity
              onPress={() => setEditing(true)}
              style={styles.editBtn}
              hitSlop={8}
            >
              <Ionicons name="pencil-outline" size={15} color={Palette.green} />
              <Text style={styles.editBtnText}>EDITAR</Text>
            </TouchableOpacity>
          ) : undefined
        }
      />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <SectionLabel label="Perfil" subtitle="Identificación operativa" />

        {/* Agent card */}
        <View style={[styles.agentCard, Shadow.card]}>
          {/* ID + status */}
          <View style={styles.cardHeader}>
            <View style={styles.idChip}>
              <Ionicons name="id-card-outline" size={11} color={Palette.textDim} />
              <Text style={styles.idChipText}>IDENTIFICACIÓN OPERATIVA</Text>
            </View>
            <StatusPill label="EN SERVICIO" status="active" />
          </View>

          {/* Photo */}
          <TouchableOpacity
            style={styles.photoWrap}
            onPress={editing ? handlePhotoPress : undefined}
            activeOpacity={editing ? 0.75 : 1}
          >
            {photoUri ? (
              <Image source={{ uri: photoUri }} style={styles.photo} resizeMode="cover" />
            ) : (
              <View style={styles.photoPlaceholder}>
                <Ionicons name="person" size={52} color={Palette.textDim} />
              </View>
            )}
            {editing && (
              <View style={styles.photoOverlay}>
                <Ionicons name="camera" size={22} color={Palette.white} />
                <Text style={styles.photoOverlayText}>Cambiar foto</Text>
              </View>
            )}
          </TouchableOpacity>

          {/* Name / badge */}
          {editing ? (
            <AppInput
              label="Nombre completo"
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
              icon={<Ionicons name="person-outline" size={14} color={Palette.textDim} />}
            />
          ) : (
            <>
              <Text style={styles.agentName}>{user?.name}</Text>
              <Text style={styles.agentBadge}>ID: {user?.badge}</Text>
            </>
          )}

          {/* Assignment */}
          <View style={styles.assignmentRow}>
            <Ionicons name="navigate-outline" size={13} color={Palette.textDim} />
            <Text style={styles.assignmentLabel}>ASIGNACIÓN</Text>
            <Text style={styles.assignmentValue}>{user?.assignment}</Text>
          </View>
        </View>

        {/* Editable contact info */}
        {editing && (
          <View style={styles.contactGroup}>
            <SectionLabel label="Información de contacto" />
            <AppInput
              label="Teléfono"
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
              icon={<Ionicons name="call-outline" size={14} color={Palette.textDim} />}
            />
            <AppInput
              label="Correo electrónico"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              icon={<Ionicons name="mail-outline" size={14} color={Palette.textDim} />}
            />
          </View>
        )}

        {/* Save / cancel */}
        {editing && (
          <View style={styles.editActions}>
            <View style={styles.editActionBtn}>
              <AppButton label="Cancelar" variant="outline" onPress={handleCancel} />
            </View>
            <View style={styles.editActionBtn}>
              <AppButton label="Guardar" variant="primary" icon="checkmark-outline" onPress={handleSave} loading={saving} />
            </View>
          </View>
        )}

        {/* Metrics */}
        {!editing && (
          <>
            <View style={styles.metricsRow}>
              <View style={styles.metricCell}>
                <Text style={styles.metricValue}>142</Text>
                <Text style={styles.metricLabel}>Intervenciones</Text>
                <Text style={styles.metricSub}>+3 esta semana</Text>
              </View>
              <View style={styles.metricCell}>
                <Text style={[styles.metricValue, { color: Palette.amber }]}>01:45</Text>
                <Text style={styles.metricLabel}>Respuesta prom.</Text>
                <Text style={styles.metricSub}>Nivel óptimo</Text>
              </View>
            </View>

            {/* Activity */}
            <View style={styles.section}>
              <SectionLabel label="Actividad reciente" />
              <View style={styles.activityCard}>
                {ACTIVITY.map((item, idx) => (
                  <View key={item.id}>
                    <View style={styles.activityRow}>
                      <View style={[styles.activityIconWrap, { borderColor: TYPE_COLOR[item.type] }]}>
                        <Ionicons name={TYPE_ICON[item.type]} size={13} color={TYPE_COLOR[item.type]} />
                      </View>
                      <View style={styles.activityContent}>
                        <View style={styles.activityTitleRow}>
                          <Text style={styles.activityTitle} numberOfLines={1}>{item.title}</Text>
                          <Text style={styles.activityTime}>{item.time}</Text>
                        </View>
                        <Text style={styles.activityDesc} numberOfLines={2}>{item.desc}</Text>
                      </View>
                    </View>
                    {idx < ACTIVITY.length - 1 && <View style={styles.divider} />}
                  </View>
                ))}
              </View>
            </View>

            {/* Report button */}
            <AppButton
              label="Generar reporte de turno"
              variant="outline"
              icon="document-text-outline"
              onPress={() => {}}
            />
          </>
        )}

        <View style={styles.bottomPad} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.bg0 },
  scroll: { flex: 1 },
  content: { padding: Space[4], gap: Space[4] },
  editBtn: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  editBtnText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.green, letterSpacing: 0.8 },
  agentCard: {
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.xl,
    padding: 16,
    gap: 10,
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  idChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  idChipText: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 0.8, textTransform: 'uppercase' },
  photoWrap: {
    height: 180,
    borderRadius: Radius.lg,
    overflow: 'hidden',
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    position: 'relative',
  },
  photo: { width: '100%', height: '100%' },
  photoPlaceholder: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: Palette.bg2 },
  photoOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  photoOverlayText: { fontSize: FontSize.sm, fontWeight: FontWeight.bold, color: Palette.white, letterSpacing: 0.5 },
  agentName: { fontSize: FontSize.xl, fontWeight: FontWeight.bold, color: Palette.textPrimary, letterSpacing: -0.3 },
  agentBadge: { fontSize: FontSize.xs, color: Palette.textDim, letterSpacing: 1 },
  assignmentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.md,
    padding: 12,
  },
  assignmentLabel: { fontSize: FontSize.xxs, fontWeight: FontWeight.bold, color: Palette.textDim, letterSpacing: 1, textTransform: 'uppercase' },
  assignmentValue: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary },
  contactGroup: { gap: Space[3] },
  editActions: { flexDirection: 'row', gap: Space[2] },
  editActionBtn: { flex: 1 },
  metricsRow: { flexDirection: 'row', gap: Space[2] },
  metricCell: {
    flex: 1,
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.lg,
    padding: 12,
    gap: 2,
  },
  metricValue: { fontSize: FontSize['2xl'], fontWeight: FontWeight.bold, color: Palette.textPrimary, letterSpacing: -0.5 },
  metricLabel: { fontSize: FontSize.xxs, color: Palette.textMuted, textTransform: 'uppercase', letterSpacing: 0.8 },
  metricSub:   { fontSize: FontSize.xxs, color: Palette.textDim, marginTop: 1 },
  section: { gap: Space[2] },
  activityCard: {
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.xl,
    overflow: 'hidden',
  },
  activityRow: { flexDirection: 'row', gap: 12, padding: 14, alignItems: 'flex-start' },
  activityIconWrap: {
    width: 28, height: 28, borderRadius: Radius.sm,
    borderWidth: 1,
    backgroundColor: Palette.bg2,
    alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
  },
  activityContent: { flex: 1, gap: 3 },
  activityTitleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  activityTitle: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Palette.textSecondary, flex: 1, marginRight: 8 },
  activityTime:  { fontSize: FontSize.xxs, color: Palette.textDim },
  activityDesc:  { fontSize: FontSize.xs, color: Palette.textMuted, lineHeight: 16 },
  divider: { height: 1, backgroundColor: Palette.border0, marginLeft: 14 },
  bottomPad: { height: 16 },
});