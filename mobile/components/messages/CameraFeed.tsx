import { StyleSheet, Text, View, Image } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

interface CameraFeedProps {
  /** URI of the camera frame / detection snapshot */
  frameUri?: string;
  /** Overlay bounding box annotation (optional) */
  showDetection?: boolean;
}

export function CameraFeed({ frameUri, showDetection = true }: CameraFeedProps) {
  return (
    <View style={styles.wrap}>
      {frameUri ? (
        <>
          <Image source={{ uri: frameUri }} style={styles.frame} resizeMode="cover" />
          {/* Detection overlay */}
          {showDetection && (
            <View style={styles.overlay}>
              {/* Corner marks */}
              <View style={[styles.corner, styles.tl]} />
              <View style={[styles.corner, styles.tr]} />
              <View style={[styles.corner, styles.bl]} />
              <View style={[styles.corner, styles.br]} />
              {/* Label */}
              <View style={styles.detectionLabel}>
                <View style={styles.detectionDot} />
                <Text style={styles.detectionText}>SUJETO DETECTADO</Text>
              </View>
            </View>
          )}
        </>
      ) : (
        /* Placeholder when no frame */
        <View style={styles.placeholder}>
          <View style={styles.placeholderIcon}>
            <Text style={styles.placeholderEmoji}>📷</Text>
          </View>
          <Text style={styles.placeholderText}>FEED EN VIVO</Text>
          <Text style={styles.placeholderSub}>Esperando transmisión…</Text>
        </View>
      )}

      {/* Live badge */}
      <View style={styles.liveBadge}>
        <View style={styles.liveDot} />
        <Text style={styles.liveText}>EN VIVO</Text>
      </View>
    </View>
  );
}

const CORNER_SIZE = 14;
const CORNER_WIDTH = 2;

const styles = StyleSheet.create({
  wrap: {
    borderRadius: Radius.xl,
    overflow: 'hidden',
    backgroundColor: Palette.bg2,
    height: 220,
    borderWidth: 1,
    borderColor: Palette.border1,
    position: 'relative',
  },
  frame: {
    width: '100%',
    height: '100%',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    borderWidth: 2,
    borderColor: 'rgba(239,68,68,0.6)',
    margin: 24,
    borderRadius: Radius.sm,
  },
  corner: {
    position: 'absolute',
    width: CORNER_SIZE,
    height: CORNER_SIZE,
    borderColor: Palette.red,
  },
  tl: { top: -1, left: -1, borderTopWidth: CORNER_WIDTH, borderLeftWidth: CORNER_WIDTH },
  tr: { top: -1, right: -1, borderTopWidth: CORNER_WIDTH, borderRightWidth: CORNER_WIDTH },
  bl: { bottom: -1, left: -1, borderBottomWidth: CORNER_WIDTH, borderLeftWidth: CORNER_WIDTH },
  br: { bottom: -1, right: -1, borderBottomWidth: CORNER_WIDTH, borderRightWidth: CORNER_WIDTH },
  detectionLabel: {
    position: 'absolute',
    bottom: -24,
    left: -1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: Palette.red,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: Radius.sm,
  },
  detectionDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: Palette.white,
  },
  detectionText: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    color: Palette.white,
    letterSpacing: 1,
  },
  placeholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  placeholderIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: Palette.bg3,
    borderWidth: 1,
    borderColor: Palette.border1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderEmoji: { fontSize: 24 },
  placeholderText: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.bold,
    color: Palette.textDim,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
  placeholderSub: {
    fontSize: FontSize.xxs,
    color: Palette.textDim,
  },
  liveBadge: {
    position: 'absolute',
    top: 10,
    left: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(0,0,0,0.65)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.sm,
    borderWidth: 1,
    borderColor: Palette.border1,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Palette.red,
  },
  liveText: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    color: Palette.textPrimary,
    letterSpacing: 1.2,
  },
});
