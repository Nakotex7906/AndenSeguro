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

// Íconos y colores por tipo de actividad (estructura lista para conectar a API)
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

// Rol legible en español
const ROLE_LABEL: Record<string, string> = {
  admin:          'Administrador',
  jefe_estacion:  'Jefe de Estación',
  seguridad:      'Seguridad',
  operador:       'Operador',
};

export default function PerfilScreen() {
  const { user, updateProfile } = useAuth();

  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(user?.fullName ?? '');
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
    updateProfile({ fullName: name.trim(), photoUri });
    setSaving(false);
    setEditing(false);
  };

  const handleCancel = () => {
    setName(user?.fullName ?? '');
    setPhotoUri(user?.photoUri);
    setEditing(false);
  };

  const roleLabel = user?.role ? (ROLE_LABEL[user.role] ?? user.role) : '—';
  const userId = user?.id ? `#${String(user.id).padStart(4, '0')}` : '—';

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

          {/* Nombre / ID */}
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
              <Text style={styles.agentName}>{user?.fullName ?? '—'}</Text>
              <Text style={styles.agentBadge}>ID: {userId}</Text>
            </>
          )}

          {/* Rol */}
          <View style={styles.assignmentRow}>
            <Ionicons name="shield-outline" size={13} color={Palette.textDim} />
            <Text style={styles.assignmentLabel}>ROL</Text>
            <Text style={styles.assignmentValue}>{roleLabel}</Text>
          </View>
        </View>

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

        {/* Métricas — pendiente de endpoint real */}
        {!editing && (
          <>
            <View style={styles.placeholderCard}>
              <Ionicons name="stats-chart-outline" size={20} color={Palette.textDim} />
              <Text style={styles.placeholderText}>
                Las métricas de turno estarán disponibles cuando el backend exponga el endpoint de estadísticas del operador.
              </Text>
            </View>

            {/* Reporte */}
            <AppButton
              label="Generar reporte de turno"
              variant="outline"
              icon="document-text-outline"
              onPress={() => {
                Alert.alert('Próximamente', 'La generación de reportes estará disponible en la próxima versión.');
              }}
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
  editActions: { flexDirection: 'row', gap: Space[2] },
  editActionBtn: { flex: 1 },
  placeholderCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: Palette.bg1,
    borderWidth: 1,
    borderColor: Palette.border0,
    borderRadius: Radius.lg,
    padding: 14,
  },
  placeholderText: {
    flex: 1,
    fontSize: FontSize.xs,
    color: Palette.textDim,
    lineHeight: 18,
  },
  bottomPad: { height: 16 },
});