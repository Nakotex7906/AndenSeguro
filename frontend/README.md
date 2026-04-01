# Anden Seguro - Frontend (Dashboard de Monitoreo)

Frontend del sistema Anden Seguro para visualizacion de alertas y supervision operativa en tiempo real.

## Objetivo

Esta aplicacion permite al personal de seguridad:

- Visualizar estado de camaras y eventos relevantes.
- Recibir alertas escalonadas en tiempo real desde el backend.
- Consultar historial de incidentes para seguimiento operativo.

## Principios del modulo

- Baja latencia en visualizacion y notificacion.
- Interfaz clara para decisiones rapidas.
- Privacidad por diseno (Ley 19.628): sin exposicion de rostros.

## Stack principal

- React 19 + TypeScript.
- Vite para desarrollo y build.
- Tailwind CSS v4 para estilos.
- ESLint + Prettier para calidad y formato.

## Estructura recomendada

```text
frontend/src/
├── api/            # Clientes HTTP y servicios
├── assets/         # Recursos estaticos
├── components/     # Componentes reutilizables
│   ├── ui/         # Botones, cards, badges
│   └── layout/     # Header, sidebar, contenedores
├── features/       # Modulos por dominio
│   ├── monitoring/ # Camaras y streaming
│   ├── alerts/     # Notificaciones y WebSockets
│   └── incidents/  # Historial de incidentes
├── hooks/          # Hooks compartidos
├── pages/          # Pantallas principales
├── routes/         # Configuracion de rutas
├── store/          # Estado global (Zustand)
├── types/          # Tipos e interfaces
└── utils/          # Utilidades puras
```

## Dependencias y reglas de uso

| Dependencia | Uso | Regla clave |
| --- | --- | --- |
| react-use-websocket | Alertas en tiempo real | Evitar polling HTTP para alertas criticas |
| zustand | Estado global ligero | Store pequeno y orientado a dominio |
| @tanstack/react-query | Fetching y cache | Toda peticion centralizada en queries/mutations |
| react-router-dom | Navegacion SPA | Rutas definidas en un modulo central |
| clsx + tailwind-merge | Clases condicionales | Usar helper `cn()` para evitar colisiones |
| @phosphor-icons/react | Iconografia | Importar solo iconos usados |

## Estandares de ingenieria

- TypeScript estricto: evitar `any`.
- TSDoc en hooks, stores y utilidades complejas.
- Nombres descriptivos para estado y propiedades.
- Cero errores de lint antes de abrir PR.

Comandos recomendados:

```bash
npm run lint
npx prettier --write .
```

## Commits (Conventional Commits + ID)

Formato:

```text
<tipo>(<alcance>): <descripcion> <palabra_clave_opcional> #<ID_Tarea>
```

Tipos:

- `feat`: nueva funcionalidad.
- `fix`: correccion de errores.
- `docs`: cambios de documentacion/TSDoc.
- `refactor`: mejora interna sin cambiar funcionalidad externa.
- `test`: pruebas.

Ejemplos:

```text
feat(monitoring): integracion de vista en tiempo real #86e0p0a66
refactor(alerts): extraccion de logica websocket fix #86e0p0a66
```

## Puesta en marcha local

```bash
cd frontend
npm install
npm run dev
```

## Build de produccion

```bash
npm run build
npm run preview
```
