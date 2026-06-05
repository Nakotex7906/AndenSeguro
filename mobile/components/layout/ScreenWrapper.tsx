import { StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Palette } from '../../constants/theme';

interface ScreenWrapperProps {
  children: React.ReactNode;
  /** Extra horizontal padding (default 16) */
  px?: number;
  /** Remove top safe-area (e.g. when TopBar handles it) */
  noTopSafe?: boolean;
}

export function ScreenWrapper({ children, px = 16, noTopSafe = false }: ScreenWrapperProps) {
  return (
    <SafeAreaView
      style={[styles.safe, { backgroundColor: Palette.bg0 }]}
      edges={noTopSafe ? ['bottom', 'left', 'right'] : ['top', 'bottom', 'left', 'right']}
    >
      <View style={[styles.inner, { paddingHorizontal: px }]}>
        {children}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
  },
  inner: {
    flex: 1,
  },
});
