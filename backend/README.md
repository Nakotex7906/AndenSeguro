# Anden Seguro - Backend (Core IA + API)

Backend del sistema Anden Seguro para monitoreo inteligente y deteccion temprana de conductas de riesgo en estaciones de Metro.

## Objetivo

Este servicio orquesta:

- Captura y procesamiento de video en tiempo real.
- Pipeline de vision artificial (deteccion, tracking, pose y riesgo).
- Gestion de incidentes y alertas en tiempo real por WebSockets.

## Principios del modulo

- Asincronia primero: evitar bloqueos del event loop de FastAPI.
- Privacidad por diseno (Ley 19.628): no procesar ni almacenar rostros.
- Trazabilidad operativa: incidentes con historial y niveles de alerta.

## Stack principal

- API: FastAPI + Uvicorn.
- IA: PyTorch + Ultralytics (YOLO) + MediaPipe.
- Video: OpenCV.
- Persistencia: SQLModel.
- Tiempo real: WebSockets.

## Estructura recomendada

```text
backend/
├── app/
│   ├── api/            # Endpoints (routes/cameras.py, routes/incidents.py)
│   ├── core/           # Configuracion (settings, entorno, websockets)
│   ├── db/             # Conexion y sesiones de base de datos
│   ├── models/         # SQLModel + esquemas Pydantic
│   ├── services/       # Logica de negocio (CRUD, alertas, reglas)
│   ├── vision/         # Pipeline IA
│   │   ├── detector.py # Deteccion de personas
│   │   ├── pose.py     # Skeletonization
│   │   └── risk.py     # Analisis temporal y riesgo
│   └── main.py         # Entrada FastAPI
├── .env.example
├── requirements.txt
└── README.md
```

## Dependencias y reglas de uso

| Dependencia | Uso | Regla clave |
| --- | --- | --- |
| fastapi + uvicorn[standard] | API y servidor ASGI | Definir endpoints con `async def` |
| ultralytics | Deteccion de personas | Cargar modelo una sola vez al iniciar la app |
| opencv-python | Captura y preprocesamiento de frames | Preprocesar antes de inferencia |
| torch | Motor de redes neuronales | Ejecutar inferencia de riesgo eficientemente |
| mediapipe | Extraccion de puntos esqueleticos | No usar ni persistir rostro |
| sqlmodel | Acceso a datos | Modelos claros y tablas descriptivas |
| websockets | Alertas de baja latencia | Usar para notificacion Naranja/Rojo |
| httpx | Cliente HTTP asincrono | No usar `requests` en rutas async |
| python-dotenv | Variables de entorno | Nunca hardcodear credenciales |

## Regla critica de negocio

El sistema activa alertas criticas cuando la probabilidad de riesgo supera el umbral:

$$
\hat{y}_{t} > 0.85
$$

## Estandares de ingenieria

- Docstrings obligatorios en servicios, endpoints y funciones complejas.
- Metodos pequenos y con una sola responsabilidad.
- Variables descriptivas (ejemplo: `risk_threshold`).
- Evitar code smells: anidaciones profundas, estado global innecesario, duplicacion.

## Commits (Conventional Commits + ID)

Formato:

```text
<tipo>(<alcance>): <descripcion> <palabra_clave_opcional> #<ID_Tarea>
```

Tipos:

- `feat`: nueva funcionalidad.
- `fix`: correccion de errores.
- `docs`: cambios de documentacion.
- `refactor`: mejora interna sin cambiar comportamiento externo.
- `test`: pruebas.

Cierre automatico de tarea:

- Usar `fix`, `close` o `resolve` antes del `#ID` cuando corresponda.

Ejemplos:

```text
feat(vision): pipeline base de deteccion y tracking #86e0p0jnh
refactor(api): migracion de requests a httpx fix #86e0p0jnh
```

## Puesta en marcha local

```bash
cd backend
python -m venv .venv
```

### Activar entorno virtual

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar servidor

```bash
uvicorn app.main:app --reload
```
