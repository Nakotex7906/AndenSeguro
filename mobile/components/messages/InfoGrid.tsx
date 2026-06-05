import { StyleSheet, Text, View } from 'react-native';
import { Palette, FontSize, FontWeight, Radius } from '../../constants/theme';

interface InfoCell {
  label: string;
  value: string | string[];
  icon?: string;
}

interface InfoGridProps {
  cells: InfoCell[];
  columns?: 2 | 3;
}

export function InfoGrid({ cells, columns = 2 }: InfoGridProps) {
  return (
    <View style={[styles.grid, { flexWrap: 'wrap', flexDirection: 'row', gap: 8 }]}>
      {cells.map((cell, i) => (
        <View
          key={i}
          style={[
            styles.cell,
            { width: columns === 2 ? '47%' : '30%' },
          ]}
        >
          {cell.icon && <Text style={styles.icon}>{cell.icon}</Text>}
          <Text style={styles.label}>{cell.label}</Text>
          {Array.isArray(cell.value) ? (
            cell.value.map((v, j) => (
              <Text key={j} style={styles.value}>{v}</Text>
            ))
          ) : (
            <Text style={styles.value}>{cell.value}</Text>
          )}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  grid: {
    // parent handles flex direction and gap
  },
  cell: {
    backgroundColor: Palette.bg2,
    borderWidth: 1,
    borderColor: Palette.border1,
    borderRadius: Radius.lg,
    padding: 12,
    gap: 3,
    flexGrow: 1,
  },
  icon: {
    fontSize: 14,
    marginBottom: 2,
  },
  label: {
    fontSize: FontSize.xxs,
    fontWeight: FontWeight.bold,
    color: Palette.textDim,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  value: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Palette.textSecondary,
  },
});