export const Palette = {
  bg0:        '#0d0e10',
  bg1:        '#111214',
  bg2:        '#181a1d',
  bg3:        '#1c1e21',
  border0:    '#1f2023',
  border1:    '#2a2d31',
  border2:    '#3a3d41',
  textPrimary:   '#f0f0f0',
  textSecondary: '#d1d5db',
  textMuted:     '#6b7280',
  textDim:       '#4b4f56',
  green:       '#22c55e',
  greenDim:    '#1a3826',
  greenBg:     '#0e1a14',
  red:         '#ef4444',
  redDim:      '#3d1212',
  redBg:       '#1a0808',
  amber:       '#f59e0b',
  amberDim:    '#3d2f0a',
  amberBg:     '#1f1a0e',
  blue:        '#3b82f6',
  blueBg:      '#0f1829',
  blueDim:     '#1e3a5f',
  white:       '#ffffff',
  black:       '#000000',
} as const;

export const FontSize = {
  xxs:   10,
  xs:    11,
  sm:    12,
  base:  13,
  md:    14,
  lg:    16,
  xl:    18,
  '2xl': 22,
  '3xl': 28,
} as const;

export const FontWeight = {
  normal:    '400' as const,
  medium:    '500' as const,
  semibold:  '600' as const,
  bold:      '700' as const,
  extrabold: '800' as const,
};

export const LetterSpacing = {
  tight:  -0.5,
  normal:  0,
  wide:    0.8,
  wider:   1.5,
  widest:  2.5,
} as const;

export const Space = {
  1:   4,
  2:   8,
  3:  12,
  4:  16,
  5:  20,
  6:  24,
  8:  32,
  10: 40,
} as const;

export const Radius = {
  sm:    6,
  md:    8,
  lg:    12,
  xl:    16,
  '2xl': 20,
  full:  9999,
} as const;

export const Shadow = {
  card: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.35,
    shadowRadius: 8,
    elevation: 5,
  },
  alert: {
    shadowColor: '#991b1b',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.45,
    shadowRadius: 24,
    elevation: 8,
  },
} as const;

export const Colors = {
  light: { text: Palette.textPrimary, background: Palette.bg0, tint: Palette.green, icon: Palette.textMuted, tabIconDefault: Palette.textMuted, tabIconSelected: Palette.green },
  dark:  { text: Palette.textPrimary, background: Palette.bg0, tint: Palette.green, icon: Palette.textMuted, tabIconDefault: Palette.textMuted, tabIconSelected: Palette.green },
};