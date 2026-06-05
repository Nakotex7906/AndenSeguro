import { useEffect } from 'react';
import { Tabs, useRouter } from 'expo-router';
import { useAuth } from '../../store/auth';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { HapticTab } from '../../components/HapticTab';
import { Palette, FontWeight } from '../../constants/theme';

type IoniconName = React.ComponentProps<typeof Ionicons>['name'];

export default function TabLayout() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/(auth)/login');
    }
  }, [isAuthenticated]);
  const bottomPad = Math.max(insets.bottom, 8);

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarShowLabel: true,
        tabBarActiveTintColor: Palette.green,
        tabBarInactiveTintColor: Palette.textDim,
        tabBarStyle: {
          backgroundColor: Palette.bg1,
          borderTopWidth: 1,
          borderTopColor: Palette.border0,
          height: 56 + bottomPad,
          paddingBottom: bottomPad,
          paddingTop: 8,
          elevation: 0,
        },
        tabBarLabelStyle: {
          fontSize: 9,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.6,
          textTransform: 'uppercase',
          marginTop: 2,
        },
        tabBarIconStyle: {
          marginBottom: 0,
        },
      }}
    >
      <Tabs.Screen
        name="home"
        options={{
          title: 'INICIO',
          tabBarIcon: ({ focused }) => (
            <Ionicons
              name={focused ? 'home' : 'home-outline'}
              size={22}
              color={focused ? Palette.green : Palette.textDim}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="messages"
        options={{
          title: 'ALERTAS',
          tabBarIcon: ({ focused }) => (
            <Ionicons
              name={focused ? 'alert-circle' : 'alert-circle-outline'}
              size={22}
              color={focused ? Palette.green : Palette.textDim}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'AJUSTES',
          tabBarIcon: ({ focused }) => (
            <Ionicons
              name={focused ? 'settings' : 'settings-outline'}
              size={22}
              color={focused ? Palette.green : Palette.textDim}
            />
          ),
          tabBarButton: (props) => (
            <TouchableOpacity
              {...(props as any)}
              style={props.style}
              onPress={() => router.push('/(tabs)/settings')}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{ href: null }}
      />
    </Tabs>
  );
}