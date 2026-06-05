import { ReactNode } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Palette, FontSize, FontWeight, LetterSpacing } from '../../constants/theme';

interface TopBarProps {
  right?: ReactNode;
}

export function TopBar({ right }: TopBarProps) {
  return (
    <SafeAreaView edges={['top']} style={{ backgroundColor: Palette.bg1 }}>
      <View style={styles.bar}>
        <View style={styles.brand}>
          <Image
            source={require('../../assets/images/logo.png')}
            style={styles.logo}
            resizeMode="contain"
          />
          <Text style={styles.brandLabel}>ANDÉN SEGURO</Text>
        </View>
        {right && <View style={styles.right}>{right}</View>}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  bar: {
    height: 52,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: Palette.border0,
    backgroundColor: Palette.bg1,
  },
  brand: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  logo: { width: 30, height: 30 },
  brandLabel: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.bold,
    letterSpacing: LetterSpacing.widest,
    color: Palette.textSecondary,
    textTransform: 'uppercase',
  },
  right: { flexDirection: 'row', alignItems: 'center', gap: 8 },
});