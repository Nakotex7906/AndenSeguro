# Andén Seguro - Mobile (App móvil)

Aplicación móvil para personal operativo y dispositivos móviles del proyecto Andén Seguro. Permite recibir alertas en tiempo real, visualizar información resumida de incidentes y confirmar ack/instrucciones desde campo.

## Objetivo

- Entregar una interfaz ligera y ágil para que el personal de campo reciba y gestione alertas (Naranja/Rojo).
- Permitir confirmación/aceptación de protocolos y envío rápido de contexto al backend.

## Principios del módulo

- Alta disponibilidad y baja latencia en notificaciones.
- Privacidad por diseño: no mostrar ni persistir rostros ni información sensible.
- UX clara y priorización de acciones críticas (ack, contacto, escalado).

## Stack principal

- Expo (Router) + React Native + TypeScript
- Eslint + Prettier
- Estado: Context/Zustand (según módulo) y hooks compartidos

## Estructura recomendada

```text
mobile/
├── app/             # Rutas y pantallas (file-based routing)
├── components/      # Componentes reutilizables
├── assets/          # Imágenes y recursos estáticos
├── hooks/           # Hooks específicos de la app
├── constants/       # Temas, valores por defecto
├── scripts/         # Scripts útiles (reset, builds locales)
└── README.md
```

## Dependencias y reglas de uso

| Dependencia | Uso | Regla clave |
| --- | --- | --- |
| `expo` / `expo-router` | Router y runtime multiplataforma | Mantener la versión de Expo consistente con docs del equipo |
| `react-native` | UI nativa multiplataforma | Evitar uso de APIs de plataforma sin feature-flag |
| `typescript` | Tipado estático | Evitar `any` en módulos compartidos |
| `eslint` + `prettier` | Calidad y formato | Ejecutar `npm run lint` antes de PR |

### Privacidad

- No subir ni almacenar imágenes o video con rostros. La app debe mostrar solo metadatos de incidentes (nivel, estación, timestamp, id).

## Estándares de ingeniería

- TypeScript estricto donde sea posible.
- Componentes pequeños y con props tipadas.
- Hooks con tests unitarios cuando su lógica es compleja.
- Cero errores de lint en PRs.

## Commits (Convencional + ID)

Formato del repo:

```text
<tipo>(<alcance>): <descripción> #<ID_Tarea>
```

Tipos habituales:

- `feat`, `fix`, `docs`, `refactor`, `test`.

Ejemplos:

```text
feat(mobile): pantalla de alerta en vivo #86e0p0mbl
fix(mobile): corregir bug de navegación en tabs #86e0p0mbl
```

## Puesta en marcha local

Recomendado: usar Node 18+ y npm o yarn.

```bash
cd mobile
npm install
# iniciar Metro / Expo
npx expo start
```

Opciones de ejecución:

- Abrir en Expo Go (dispositivo físico)
- Usar emulador Android / iOS desde el output de `expo start`
- Crear un development build para funciones nativas con `eas build --profile development` (si configurado)

## Build de producción

Si se requiere build nativo, usar EAS Build (configurar `eas.json`):

```bash
# ejemplo (requiere configurar cuenta Expo + EAS)
eas build --platform all --profile production
```

## Testing y Quality

- Ejecutar linters y formateo antes de PR:

```bash
npm run lint
npx prettier --write .
```

