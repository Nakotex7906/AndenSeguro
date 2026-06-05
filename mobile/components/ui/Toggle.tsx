import { Switch } from 'react-native';
import { Palette } from '../../constants/theme';

interface ToggleProps {
  value: boolean;
  onValueChange: (v: boolean) => void;
  color?: string;
}

export function Toggle({ value, onValueChange, color = Palette.green }: ToggleProps) {
  return (
    <Switch
      value={value}
      onValueChange={onValueChange}
      trackColor={{ false: Palette.border1, true: color }}
      thumbColor={Palette.white}
      ios_backgroundColor={Palette.border1}
    />
  );
}
