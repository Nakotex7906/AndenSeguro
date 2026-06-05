# Andén Seguro — Contrato Técnico de API

> **Actividad 4 — Contrato API | Entregable 2: api_contract.md**
>
> **Versión:** `1.0.0` | **Fecha:** `2026-05-29` | **Estado:** `Vigente`

---

## Tabla de Contenidos

1. [Justificación y Estandarización](#1-justificación-y-estandarización)
2. [Estructuras Estándar de Mensajes](#2-estructuras-estándar-de-mensajes)
3. [Catálogo Completo de Endpoints](#3-catálogo-completo-de-endpoints)
4. [Subsistema de Tiempo Real — WebSocket](#4-subsistema-de-tiempo-real--websocket)

---

## 1. Justificación y Estandarización

### 1.1 Adopción de RESTful API

La arquitectura **REST (Representational State Transfer)** fue seleccionada como paradigma de comunicación cliente-servidor por las siguientes razones técnicas y de proyecto:

| Criterio | Justificación |
|---|---|
| **Sin estado (Stateless)** | Cada petición HTTP contiene toda la información necesaria para ser procesada de forma autónoma. Esto permite escalar horizontalmente el backend FastAPI sin necesidad de gestión de sesiones compartidas. |
| **Interfaz uniforme** | La semántica de los verbos HTTP (`GET`, `POST`, `PUT`, `DELETE`) proporciona un contrato predecible para el equipo de frontend, reduciendo la curva de aprendizaje y el acoplamiento entre módulos. |
| **Recursos nombrados** | La representación de entidades como recursos en URL en plural (`/api/incidents`, `/api/personnel`) es autoexplicativa, cohesionada y alineada con el dominio de negocio. |
| **Compatibilidad con FastAPI** | FastAPI está optimizado para la definición de rutas RESTful con validación automática de schemas vía Pydantic, generación de documentación OpenAPI 3.x y soporte nativo para respuestas asíncronas. |
| **Consumo estándar desde React** | El cliente `fetch` nativo del navegador la biblioteca o `tanstack/react-query` consumen endpoints REST de forma natural, simplificando la capa de integración en el dashboard. |

### 1.2 Formato de Intercambio JSON en `snake_case`

Todas las cargas útiles (payloads) de la API, tanto de entrada como de salida, utilizan el formato **JSON** con nomenclatura **`snake_case`** estricta para todos los nombres de campos. Esta decisión se justifica porque:

- `snake_case` es la convención nativa de Python (PEP 8) y de Pydantic, eliminando la necesidad de transformaciones entre la capa de modelo y la capa de serialización.
- Garantiza consistencia entre los modelos ORM (SQLModel), los schemas de validación (Pydantic) y los objetos JSON transmitidos, reduciendo la superficie de errores por transformaciones de nomenclatura.
- Evita ambigüedades de parsing (e.g., `camelCase` en JSON puede colisionar con reservas de palabras en ciertos entornos JavaScript).

**Ejemplo de campo correcto:**

```json
{
  "risk_score": 0.91,
  "camera_id": "CAM-01-ANDEN-3",
  "automated_description": "Individuo ID 7 con inclinación severa hacia las vías."
}
```

### 1.3 Estándar Cronológico ISO 8601

Todos los campos de fecha y hora en la API siguen el estándar **ISO 8601** con timezone UTC explícito. El formato canónico utilizado es:

```
YYYY-MM-DDTHH:MM:SS.mmmZ
```

**Ejemplo:**

```
2025-07-14T03:45:12.334Z
```

**Justificación técnica:**

- Elimina ambigüedades de timezone en un sistema que podría ser monitoreado desde distintas zonas horarias.
- Es el formato nativo de `datetime` de Python al serializar con Pydantic (`datetime.isoformat()`).
- Es directamente parseable por `new Date(string)` en JavaScript sin configuración adicional.
- Facilita la ordenación cronológica lexicográfica directa en consultas de base de datos.

### 1.4 Justificación Técnica de WebSockets para el Módulo de Alertas

La comunicación de alertas de emergencia **no puede depender del modelo de solicitud-respuesta HTTP (polling)** por las siguientes razones técnicas en relación directa con los requerimientos no funcionales del proyecto:

| RNF | Exigencia | Por qué HTTP Polling falla | Por qué WebSocket resuelve |
|---|---|---|---|
| **RNF-01** | Latencia máxima de **1 segundo** desde captura hasta alerta. | El polling introduce una latencia mínima igual al intervalo entre solicitudes (típicamente 1-5 s). Reducirlo a 100 ms genera una carga de red y CPU inasumible. | El servidor **empuja (push)** el evento al cliente en el mismo instante en que se detecta el riesgo. La latencia de red es el único factor de demora, típicamente < 50 ms en LAN. |
| **RNF-06** | Notificación completada en un máximo de **30 segundos**. | Con polling agresivo, la latencia acumulada entre detección, procesamiento y próxima solicitud del cliente puede superar el umbral crítico en escenarios de carga. | La conexión WebSocket persiste, eliminando el overhead de establecimiento de conexión TCP/TLS en cada evento. |

**Protocolo seleccionado:** `WebSocket` nativo del estándar W3C (RFC 6455), soportado de forma nativa por FastAPI mediante el módulo `websockets` y por todos los navegadores modernos.

**Modelo de operación:** El servidor mantiene una lista de conexiones activas. Cuando `CameraService` detecta que `risk_score >= 0.60`, el módulo `AlertService` serializa un objeto `AlertEvent` y lo difunde (broadcast) a todos los clientes WebSocket conectados al canal `ws://.../api/alerts/ws`.

---

## 2. Estructuras Estándar de Mensajes

### 2.1 Patrón de Envoltura para Respuestas Exitosas (Envelope Pattern)

Todas las respuestas exitosas de la API REST siguen una estructura de envoltura unificada que encapsula los datos junto con metainformación de control. Esto garantiza que el cliente pueda procesar la respuesta de forma genérica sin conocer el tipo de dato contenido.

**Estructura base (código HTTP `200 OK` o `201 Created`):**

```json
{
  "success": true,
  "status_code": 200,
  "message": "Operación completada exitosamente.",
  "data": { },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

**Descripción de campos del envelope:**

| Campo | Tipo | Descripción |
|---|---|---|
| `success` | `boolean` | Siempre `true` en respuestas exitosas. Facilita el manejo condicional en el cliente. |
| `status_code` | `integer` | Refleja el código HTTP de la respuesta para facilitar el logging del lado cliente. |
| `message` | `string` | Descripción legible en lenguaje natural del resultado de la operación. |
| `data` | `object` o `array` | Payload principal. Puede ser un objeto único o un arreglo de objetos según el endpoint. |
| `meta` | `object` | Metainformación de la respuesta: timestamp de generación y versión de API. |

**Variante para listas paginadas** (endpoint `GET /api/incidents`):

```json
{
  "success": true,
  "status_code": 200,
  "message": "Incidentes recuperados exitosamente.",
  "data": [ ],
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 154,
      "total_pages": 8
    }
  }
}
```

### 2.2 Estructura Estándar para Respuestas de Error

Todas las respuestas de error siguen un formato unificado independientemente del tipo de error, asociado a los códigos de estado HTTP semánticos correspondientes.

**Estructura base de error:**

```json
{
  "success": false,
  "status_code": 422,
  "error": {
    "error_code": "VALIDATION_ERROR",
    "message": "El cuerpo de la solicitud contiene campos inválidos.",
    "details": [
      {
        "field": "yellow_points",
        "issue": "Se esperaba una lista de tuplas de coordenadas flotantes, se recibió null."
      },
      {
        "field": "red_points",
        "issue": "La lista no puede estar vacía. Debe contener al menos 3 puntos para formar una zona válida."
      }
    ]
  },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

**Descripción de campos del objeto `error`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `error_code` | `string` | Código interno de error en `SCREAMING_SNAKE_CASE`. Identifica el tipo de error de forma programática. |
| `message` | `string` | Descripción legible del error orientada al desarrollador cliente. |
| `details` | `array<object>` | Matriz de objetos con el detalle técnico de cada problema encontrado. Puede estar vacía (`[]`) si el error no tiene campos específicos. |

**Tabla de códigos de error internos y su mapeo HTTP:**

| `error_code` | Código HTTP | Escenario de uso |
|---|---|---|
| `VALIDATION_ERROR` | `422 Unprocessable Entity` | El cuerpo de la solicitud no cumple con el schema Pydantic definido. |
| `BAD_REQUEST` | `400 Bad Request` | Parámetros de query string con valores fuera del rango permitido. |
| `RESOURCE_NOT_FOUND` | `404 Not Found` | El recurso solicitado (e.g., incidente con ID específico) no existe. |
| `CAMERA_UNAVAILABLE` | `503 Service Unavailable` | El servicio de cámara EZVIZ no está accesible o el token de sesión expiró. |
| `MODEL_INFERENCE_ERROR` | `500 Internal Server Error` | Error durante la inferencia del modelo YOLO o LSTM. |
| `INTERNAL_SERVER_ERROR` | `500 Internal Server Error` | Error genérico no clasificado en el servidor. |
| `ASSIGNMENT_CONFLICT` | `409 Conflict` | No hay personal disponible con el criterio de género requerido para asignar. |

---

## 3. Catálogo Completo de Endpoints

### Convención de Rutas

- **Prefijo base de la API:** `/api`
- **Recursos en plural:** `/incidents`, `/personnel`
- **Recursos de streaming agrupados:** `/api/stream/`
- **Parámetros de ruta:** `{id}` en `snake_case`
- **Parámetros de query:** en `snake_case` y `snake_case`

---

### 3.1 `GET /api/stream/video_feed`

**Descripción:** Inicia y transmite un flujo de video continuo en formato MJPEG (Motion JPEG). Cada frame del stream incluye las anotaciones visuales superpuestas por el pipeline de IA: bounding boxes de personas detectadas, ID de rastreo, indicadores de nivel de riesgo por color (verde/naranja/rojo) y el esqueleto de postura estimado. Este endpoint es consumido directamente por el elemento `<img>` del dashboard React.

**Método:** `GET`

**Ruta:** `/api/stream/video_feed`

**Autenticación:** No requerida en entorno local de desarrollo. Se recomienda protección por API Key en despliegue productivo.

**Parámetros de entrada:** Ninguno.

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `200 OK` | El stream se inicia correctamente. El `Content-Type` de respuesta es `multipart/x-mixed-replace; boundary=frame`. |
| `503 Service Unavailable` | La cámara IP EZVIZ no está disponible o el token de autenticación con la plataforma expiró. |
| `500 Internal Server Error` | Error irrecuperable en el pipeline de procesamiento de frames. |

**Headers de respuesta:**

```
Content-Type: multipart/x-mixed-replace; boundary=frame
Cache-Control: no-cache
Connection: keep-alive
```

**Tipo de respuesta:** Streaming binario MJPEG. No aplica el Envelope Pattern JSON para este endpoint por su naturaleza de streaming multimedia.

**Ejemplo de consumo en el frontend (React):**

```jsx
<img
  src={`${import.meta.env.VITE_API_BASE_URL}/api/stream/video_feed`}
  alt="Feed de cámara en tiempo real"
/>
```

---

### 3.2 `GET /api/stream/config`

**Descripción:** Recupera la configuración actual de las zonas de riesgo definidas sobre el plano de la cámara. Retorna las coordenadas de los polígonos que delimitan la **zona de advertencia amarilla** (intervención preventiva) y la **zona de peligro roja** (peligro inmediato de caída a las vías). Las coordenadas están normalizadas en el rango `[0.0, 1.0]` relativas al ancho y alto del frame de video.

**Método:** `GET`

**Ruta:** `/api/stream/config`

**Autenticación:** No requerida en entorno de desarrollo.

**Parámetros de entrada:** Ninguno.

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `200 OK` | Configuración de zonas recuperada exitosamente. |
| `500 Internal Server Error` | Error al leer el archivo de configuración persistido. |

**Payload de respuesta exitosa (`200 OK`):**

```json
{
  "success": true,
  "status_code": 200,
  "message": "Configuración de zonas recuperada exitosamente.",
  "data": {
    "yellow_points": [
      [0.10, 0.65],
      [0.90, 0.65],
      [0.90, 0.80],
      [0.10, 0.80]
    ],
    "red_points": [
      [0.15, 0.80],
      [0.85, 0.80],
      [0.85, 1.00],
      [0.15, 1.00]
    ]
  },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

**Descripción de campos de `data`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `yellow_points` | `array<array<float>>` | Lista de coordenadas `[x, y]` normalizadas que definen el polígono de la **zona amarilla** de advertencia. |
| `red_points` | `array<array<float>>` | Lista de coordenadas `[x, y]` normalizadas que definen el polígono de la **zona roja** de peligro inmediato. |

---

### 3.3 `POST /api/stream/config`

**Descripción:** Actualiza la configuración de los polígonos de zonas de riesgo en el sistema. Las nuevas coordenadas son persistidas en el archivo de configuración del servicio para sobrevivir reinicios del servidor. Esta operación es típicamente realizada por el administrador del sistema desde el dashboard al ajustar las zonas a la geometría real del andén en la imagen de la cámara.

**Método:** `POST`

**Ruta:** `/api/stream/config`

**Autenticación:** Se recomienda rol de administrador en despliegue productivo.

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `200 OK` | Configuración actualizada y persistida correctamente. |
| `422 Unprocessable Entity` | El payload no cumple el schema. Coordenadas fuera del rango `[0.0, 1.0]` o listas vacías. |
| `500 Internal Server Error` | Error al persistir la configuración en el sistema de archivos. |

**Payload de entrada (`Content-Type: application/json`):**

```json
{
  "yellow_points": [
    [0.10, 0.65],
    [0.90, 0.65],
    [0.90, 0.80],
    [0.10, 0.80]
  ],
  "red_points": [
    [0.15, 0.80],
    [0.85, 0.80],
    [0.85, 1.00],
    [0.15, 1.00]
  ]
}
```

**Descripción de campos del payload:**

| Campo | Tipo | Requerido | Validación |
|---|---|---|---|
| `yellow_points` | `array<array<float>>` | Sí | Lista no vacía. Mínimo 3 pares `[x, y]`. Valores en rango `[0.0, 1.0]`. |
| `red_points` | `array<array<float>>` | Sí | Lista no vacía. Mínimo 3 pares `[x, y]`. Valores en rango `[0.0, 1.0]`. |

**Payload de respuesta exitosa (`200 OK`):**

```json
{
  "success": true,
  "status_code": 200,
  "message": "Zonas de riesgo actualizadas correctamente.",
  "data": {
    "updated_at": "2025-07-14T03:45:12.334Z",
    "yellow_points_count": 4,
    "red_points_count": 4
  },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

**Payload de respuesta de error de validación (`422 Unprocessable Entity`):**

```json
{
  "success": false,
  "status_code": 422,
  "error": {
    "error_code": "VALIDATION_ERROR",
    "message": "El cuerpo de la solicitud contiene campos inválidos.",
    "details": [
      {
        "field": "red_points",
        "issue": "La lista no puede estar vacía. Se requieren al menos 3 puntos para definir un polígono."
      }
    ]
  },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

---

### 3.4 `GET /api/stream/stats`

**Descripción:** Retorna un snapshot de las estadísticas operativas del pipeline de visión artificial en el instante de la consulta. Incluye el conteo de personas detectadas actualmente, el número de alertas activas por nivel y la carga de procesamiento de la GPU. Este endpoint es consumido periódicamente por el panel de estadísticas del dashboard.

**Método:** `GET`

**Ruta:** `/api/stream/stats`

**Autenticación:** No requerida en entorno de desarrollo.

**Parámetros de entrada:** Ninguno.

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `200 OK` | Estadísticas del pipeline recuperadas exitosamente. |
| `503 Service Unavailable` | El servicio de cámara o el pipeline de IA no están inicializados. |

**Payload de respuesta exitosa (`200 OK`):**

```json
{
  "success": true,
  "status_code": 200,
  "message": "Estadísticas del pipeline recuperadas exitosamente.",
  "data": {
    "pipeline_status": "running",
    "persons_detected": 12,
    "active_tracks": 12,
    "alerts_active": {
      "orange_count": 1,
      "red_count": 0
    },
    "processing_metrics": {
      "avg_inference_latency_ms": 87.4,
      "fps": 24.1,
      "gpu_utilization_percent": 63.2,
      "gpu_memory_used_mb": 1842
    },
    "camera_info": {
      "camera_id": "CAM-01-ANDEN-3",
      "serial": "F12345678",
      "stream_status": "connected"
    },
    "last_updated_at": "2025-07-14T03:45:12.334Z"
  },
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0"
  }
}
```

**Descripción de campos de `data`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `pipeline_status` | `string` | Estado del pipeline: `"running"`, `"paused"`, `"error"`. |
| `persons_detected` | `integer` | Número total de personas actualmente en el frame. |
| `active_tracks` | `integer` | Número de tracks activos en el módulo DeepSORT. |
| `alerts_active.orange_count` | `integer` | Alertas de nivel naranja activas (score 0.60-0.84). |
| `alerts_active.red_count` | `integer` | Alertas de nivel rojo activas (score >= 0.85). |
| `processing_metrics.avg_inference_latency_ms` | `float` | Latencia promedio de inferencia YOLO+LSTM en milisegundos. |
| `processing_metrics.fps` | `float` | Frames por segundo procesados actualmente. |
| `processing_metrics.gpu_utilization_percent` | `float` | Porcentaje de utilización de GPU NVIDIA. `null` si opera en CPU. |
| `processing_metrics.gpu_memory_used_mb` | `integer` | Memoria GPU utilizada en MB. `null` si opera en CPU. |
| `camera_info.camera_id` | `string` | Identificador lógico de la cámara. |
| `camera_info.stream_status` | `string` | Estado de la conexión: `"connected"`, `"disconnected"`, `"reconnecting"`. |
| `last_updated_at` | `string` | Timestamp ISO 8601 de la última actualización de las estadísticas. |

---

### 3.5 `GET /api/incidents`

**Descripción:** Retorna el listado paginado y filtrable del registro histórico de incidentes auditados. Cumple el **RF-09**, proveyendo a los administradores y al equipo de auditoría acceso cronológico a todos los eventos de riesgo detectados, con los metadatos no identificables de cada uno: timestamp, cámara, nivel de riesgo, score numérico y acción tomada. Este endpoint es el punto de entrada principal para la generación de reportes de seguridad.

**Método:** `GET`

**Ruta:** `/api/incidents`

**Autenticación:** Rol de administrador u operador autorizado recomendado en producción.

**Parámetros de query string:**

| Parámetro | Tipo | Requerido | Valor por defecto | Descripción |
|---|---|---|---|---|
| `page` | `integer` | No | `1` | Número de página para la paginación. |
| `page_size` | `integer` | No | `20` | Cantidad de resultados por página. Máximo `100`. |
| `risk_level` | `string` | No | `null` | Filtrar por nivel: `"ORANGE"` o `"RED"`. |
| `camera_id` | `string` | No | `null` | Filtrar por identificador de cámara. |
| `date_from` | `string` | No | `null` | Fecha de inicio del rango (ISO 8601). Ej: `2025-07-01T00:00:00Z`. |
| `date_to` | `string` | No | `null` | Fecha de fin del rango (ISO 8601). Ej: `2025-07-14T23:59:59Z`. |
| `status` | `string` | No | `null` | Filtrar por estado: `"active"`, `"resolved"`, `"false_positive"`. |

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `200 OK` | Listado de incidentes recuperado exitosamente. |
| `400 Bad Request` | Parámetros de query con formato inválido (e.g., fechas mal formateadas). |
| `500 Internal Server Error` | Error en la consulta a la base de datos. |

**Payload de respuesta exitosa (`200 OK`):**

```json
{
  "success": true,
  "status_code": 200,
  "message": "Incidentes recuperados exitosamente.",
  "data": [
    {
      "id": 47,
      "timestamp": "2025-07-14T03:41:05.211Z",
      "camera_id": "CAM-01-ANDEN-3",
      "track_id": 7,
      "risk_level": "RED",
      "risk_score": 0.93,
      "frame_capture_url": "/api/incidents/47/capture",
      "automated_description": "Individuo ID 7 detectado con inclinación corporal severa hacia las vías. Postura de riesgo sostenida por 3.2 segundos consecutivos.",
      "action_taken": "Protocolo de emergencia activado. Personal asignado: Operadora González.",
      "status": "resolved",
      "resolved_at": "2025-07-14T03:41:31.004Z"
    },
    {
      "id": 46,
      "timestamp": "2025-07-14T02:18:43.779Z",
      "camera_id": "CAM-02-ANDEN-1",
      "track_id": 15,
      "risk_level": "ORANGE",
      "risk_score": 0.71,
      "frame_capture_url": "/api/incidents/46/capture",
      "automated_description": "Individuo ID 15 detectado en zona amarilla con comportamiento oscilante. Score de riesgo en ascenso.",
      "action_taken": "Alerta preventiva enviada. Monitoreo intensificado.",
      "status": "resolved",
      "resolved_at": "2025-07-14T02:19:15.002Z"
    }
  ],
  "meta": {
    "timestamp": "2025-07-14T03:45:12.334Z",
    "api_version": "1.0.0",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 47,
      "total_pages": 3
    }
  }
}
```

**Descripción de campos de cada objeto en `data`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | `integer` | Identificador único autoincremental del incidente. |
| `timestamp` | `string` | Momento exacto de detección del incidente (ISO 8601 UTC). |
| `camera_id` | `string` | Identificador lógico de la cámara que detectó el incidente. |
| `track_id` | `integer` | ID de rastreo DeepSORT del individuo (no biométrico, efímero por sesión). |
| `risk_level` | `string` | Nivel de escalamiento: `"ORANGE"` o `"RED"`. |
| `risk_score` | `float` | Valor de probabilidad de riesgo numérico en el momento de activación (`0.0` - `1.0`). |
| `frame_capture_url` | `string` | URL relativa para recuperar la captura visual del frame del incidente. |
| `automated_description` | `string` | Descripción generada automáticamente por el sistema sobre el comportamiento detectado. |
| `action_taken` | `string` | Descripción de la acción ejecutada en respuesta al incidente. |
| `status` | `string` | Estado del incidente: `"active"`, `"resolved"`, `"false_positive"`. |
| `resolved_at` | `string` \| `null` | Timestamp de resolución del incidente. `null` si aún está activo (ISO 8601 UTC). |

---

### 3.6 `POST /api/personnel/assign`

**Descripción:** Ejecuta la lógica de asignación y despacho de personal operativo para atender un incidente activo. Cumple el **RF-11**, implementando una asignación inteligente que considera el género aparente de la persona en riesgo (para respetar protocolos de intervención sensibles al género) y la disponibilidad operativa actual del personal. Si no hay personal disponible con el criterio prioritario, el sistema aplica escalamiento progresivo asignando al operador disponible más cercano.

**Método:** `POST`

**Ruta:** `/api/personnel/assign`

**Autenticación:** Rol de operador o sistema automatizado requerido en producción.

**Códigos de respuesta HTTP:**

| Código | Descripción |
|---|---|
| `201 Created` | Asignación creada y operador notificado exitosamente. |
| `409 Conflict` | No hay personal disponible con el criterio solicitado ni para escalamiento. |
| `404 Not Found` | El `incident_id` proporcionado no existe en la base de datos. |
| `422 Unprocessable Entity` | El payload no cumple el schema de validación. |
| `500 Internal Server Error` | Error al ejecutar la lógica de asignación o al notificar al operador. |

**Payload de entrada (`Content-Type: application/json`):**

```json
{
  "incident_id": 47,
  "preferred_gender": "female",
  "priority_level": "RED",
  "requester_notes": "La persona en riesgo presenta signos de angustia severa. Se solicita operadora femenina con capacitación en crisis."
}
```

**Descripción de campos del payload de entrada:**

| Campo | Tipo | Requerido | Validación |
|---|---|---|---|
| `incident_id` | `integer` | Sí | Debe corresponder a un incidente existente con `status = "active"`. |
| `preferred_gender` | `string` | Sí | Valores aceptados: `"female"`, `"male"`, `"any"`. Criterio de género preferido para el protocolo de intervención. |
| `priority_level` | `string` | Sí | Valores aceptados: `"ORANGE"`, `"RED"`. Determina la urgencia del despacho. |
| `requester_notes` | `string` | No | Notas adicionales del operador o sistema que solicita la asignación. Máximo 500 caracteres. |

**Payload de respuesta exitosa (`201 Created`):**

```json
{
  "success": true,
  "status_code": 201,
  "message": "Personal asignado y notificado exitosamente.",
  "data": {
    "assignment_id": 38,
    "incident_id": 47,
    "assigned_at": "2025-07-14T03:41:08.990Z",
    "assigned_operator": {
      "operator_id": 12,
      "display_name": "Operadora González",
      "gender": "female",
      "current_zone": "Andén 3 - Nivel Superior"
    },
    "preferred_gender_met": true,
    "escalation_applied": false,
    "assignment_status": "dispatched",
    "estimated_response_time_seconds": 22
  },
  "meta": {
    "timestamp": "2025-07-14T03:41:09.001Z",
    "api_version": "1.0.0"
  }
}
```

**Payload de respuesta de conflicto (`409 Conflict`):**

```json
{
  "success": false,
  "status_code": 409,
  "error": {
    "error_code": "ASSIGNMENT_CONFLICT",
    "message": "No hay personal operativo disponible en este momento para atender el incidente.",
    "details": [
      {
        "field": "preferred_gender",
        "issue": "No hay operadoras femeninas disponibles. Todos los operadores disponibles son masculinos."
      },
      {
        "field": "escalation",
        "issue": "Se intentó escalamiento progresivo, pero no hay ningún operador disponible en la estación."
      }
    ]
  },
  "meta": {
    "timestamp": "2025-07-14T03:41:09.001Z",
    "api_version": "1.0.0"
  }
}
```

**Descripción de campos de `data` en respuesta exitosa:**

| Campo | Tipo | Descripción |
|---|---|---|
| `assignment_id` | `integer` | Identificador único del registro de asignación creado. |
| `incident_id` | `integer` | ID del incidente al que responde esta asignación. |
| `assigned_at` | `string` | Timestamp ISO 8601 del momento exacto de la asignación. |
| `assigned_operator.operator_id` | `integer` | ID interno del operador asignado. |
| `assigned_operator.display_name` | `string` | Nombre de pantalla del operador (sin apellido completo por privacidad). |
| `assigned_operator.gender` | `string` | Género del operador asignado: `"female"` o `"male"`. |
| `assigned_operator.current_zone` | `string` | Zona de la estación donde se encuentra el operador al momento de la asignación. |
| `preferred_gender_met` | `boolean` | `true` si el operador asignado cumple el criterio de género solicitado. |
| `escalation_applied` | `boolean` | `true` si fue necesario ignorar el criterio de género por falta de disponibilidad. |
| `assignment_status` | `string` | Estado del despacho: `"dispatched"`, `"pending_acknowledgment"`. |
| `estimated_response_time_seconds` | `integer` | Tiempo estimado de llegada al punto de incidente en segundos. |

---

## 4. Subsistema de Tiempo Real — WebSocket

### 4.1 Especificación del Canal de Alertas

El sistema expone un canal WebSocket dedicado exclusivamente a la difusión de eventos de alerta en tiempo real. Este canal es la columna vertebral del cumplimiento de los **RNF-01** (latencia < 1 s) y **RNF-06** (notificación en < 30 s).

**URI del canal:**

```
ws://{host}:{port}/api/alerts/ws
```

**Ejemplo en entorno de desarrollo:**

```
ws://localhost:8000/api/alerts/ws
```

**Ejemplo en entorno de producción (con TLS):**

```
wss://anden-seguro.ejemplo.cl/api/alerts/ws
```

**Protocolo:** WebSocket (RFC 6455). El servidor no requiere subprotocolo específico.

**Ciclo de vida de la conexión:**

```
Cliente                          Servidor
   |                                |
   |--- HTTP GET /api/alerts/ws --->|  (Handshake HTTP Upgrade)
   |<-- 101 Switching Protocols ----|
   |                                |
   |<-- {"event_type": "connection_established", ...} --|  (Confirmación de conexión)
   |                                |
   |     [Conexión WebSocket activa]|
   |                                |
   |<-- AlertEvent JSON (push) -----|  (Cuando risk_score >= 0.60)
   |<-- AlertEvent JSON (push) -----|  (Cuando risk_score >= 0.85)
   |                                |
   |--- CLOSE frame --------------->|  (Al cerrar el dashboard)
   |<-- CLOSE frame ----------------|
```

### 4.2 Evento de Confirmación de Conexión

Inmediatamente tras establecer la conexión, el servidor emite un mensaje de confirmación:

```json
{
  "event_type": "connection_established",
  "timestamp": "2025-07-14T03:45:00.000Z",
  "message": "Conexión WebSocket establecida. Escuchando alertas del canal Andén Seguro."
}
```

### 4.3 Evento Push de Alerta — `AlertEvent`

Este es el mensaje principal que el backend emite al frontend en tiempo real cuando el pipeline de IA detecta que la probabilidad de riesgo de un individuo supera los umbrales definidos en **RF-07**.

**Condición de emisión:**

- **Nivel NARANJA:** `risk_score >= 0.60` y `risk_score < 0.85`
- **Nivel ROJO:** `risk_score >= 0.85`

**Estructura del objeto `AlertEvent` (JSON completo):**

```json
{
  "event_type": "risk_alert",
  "alert_id": "alert_20250714_034512_track7",
  "timestamp": "2025-07-14T03:45:12.334Z",
  "risk_assessment": {
    "track_id": 7,
    "risk_score": 0.93,
    "risk_level": "RED",
    "risk_level_label": "EMERGENCIA — Protocolo Inmediato",
    "score_trend": "ascending",
    "frames_in_risk_zone": 14
  },
  "location": {
    "camera_id": "CAM-01-ANDEN-3",
    "camera_display_name": "Cámara 01 — Andén 3, Línea 2",
    "station_name": "Estación Baquedano",
    "zone_triggered": "red_zone",
    "position_in_frame": {
      "x_normalized": 0.52,
      "y_normalized": 0.87
    }
  },
  "visual_evidence": {
    "frame_capture_url": "/api/incidents/47/capture",
    "capture_available": true,
    "skeleton_overlay_available": true
  },
  "automated_description": "Individuo ID 7 detectado con inclinación corporal severa hacia las vías en zona roja. La postura de riesgo ha sido sostenida de forma continua durante 14 frames consecutivos (aproximadamente 0.58 segundos). El vector de movimiento indica desplazamiento activo hacia el borde del andén.",
  "recommended_action": {
    "action_code": "DISPATCH_EMERGENCY_PERSONNEL",
    "action_description": "Activar protocolo de emergencia inmediato. Asignar personal operativo al andén. Considerar detención preventiva del tren.",
    "auto_assignment_triggered": true,
    "assignment_endpoint": "/api/personnel/assign"
  },
  "notification": {
    "alert_sound": "critical",
    "ui_color": "#EF4444",
    "ui_priority": 1,
    "requires_acknowledgment": true
  }
}
```

**Descripción detallada de los campos del `AlertEvent`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `event_type` | `string` | Tipo de evento WebSocket. Para alertas siempre es `"risk_alert"`. |
| `alert_id` | `string` | Identificador único del evento de alerta. Formato: `alert_{YYYYMMDD}_{HHmmss}_{trackN}`. |
| `timestamp` | `string` | Momento exacto de emisión del evento (ISO 8601 UTC). |
| `risk_assessment.track_id` | `integer` | ID de rastreo DeepSORT del individuo (efímero, no biométrico). |
| `risk_assessment.risk_score` | `float` | Probabilidad de riesgo calculada por el LSTM en el instante de la alerta (`0.0` - `1.0`). |
| `risk_assessment.risk_level` | `string` | Nivel de escalamiento: `"ORANGE"` o `"RED"`. |
| `risk_assessment.risk_level_label` | `string` | Etiqueta descriptiva legible para el dashboard. |
| `risk_assessment.score_trend` | `string` | Tendencia del score: `"ascending"`, `"stable"`, `"descending"`. |
| `risk_assessment.frames_in_risk_zone` | `integer` | Cantidad de frames consecutivos en que el individuo ha superado el umbral de riesgo. |
| `location.camera_id` | `string` | Identificador lógico de la cámara origen. |
| `location.camera_display_name` | `string` | Nombre descriptivo de la cámara para mostrar en el dashboard. |
| `location.station_name` | `string` | Nombre de la estación de metro. |
| `location.zone_triggered` | `string` | Zona de riesgo activada: `"yellow_zone"` o `"red_zone"`. |
| `location.position_in_frame` | `object` | Coordenadas normalizadas `[0.0, 1.0]` de la posición del individuo en el frame. |
| `visual_evidence.frame_capture_url` | `string` | URL relativa para recuperar la imagen capturada del frame del incidente. |
| `visual_evidence.capture_available` | `boolean` | `true` si la captura visual está disponible para consulta inmediata. |
| `visual_evidence.skeleton_overlay_available` | `boolean` | `true` si la captura incluye el esqueleto de postura superpuesto. |
| `automated_description` | `string` | Descripción narrativa generada automáticamente por el sistema sobre el comportamiento detectado. Incluye duración, dirección de movimiento y contexto de riesgo. |
| `recommended_action.action_code` | `string` | Código de acción recomendada: `"MONITOR_PREVENTIVE"` (naranja) o `"DISPATCH_EMERGENCY_PERSONNEL"` (rojo). |
| `recommended_action.action_description` | `string` | Descripción textual de la acción recomendada para el operador. |
| `recommended_action.auto_assignment_triggered` | `boolean` | `true` si el sistema ya invocó automáticamente `POST /api/personnel/assign`. |
| `recommended_action.assignment_endpoint` | `string` | URL del endpoint de asignación para invocación manual desde el dashboard. |
| `notification.alert_sound` | `string` | Tipo de alerta sonora para el dashboard: `"warning"` (naranja) o `"critical"` (rojo). |
| `notification.ui_color` | `string` | Color hexadecimal para la UI del dashboard: `"#F97316"` (naranja) o `"#EF4444"` (rojo). |
| `notification.ui_priority` | `integer` | Prioridad de visualización en la cola de alertas. `1` = máxima prioridad. |
| `notification.requires_acknowledgment` | `boolean` | `true` si el dashboard debe solicitar confirmación explícita del operador para cerrar la alerta. |

### 4.4 Ejemplo de Alerta de Nivel NARANJA

```json
{
  "event_type": "risk_alert",
  "alert_id": "alert_20250714_021843_track15",
  "timestamp": "2025-07-14T02:18:43.779Z",
  "risk_assessment": {
    "track_id": 15,
    "risk_score": 0.71,
    "risk_level": "ORANGE",
    "risk_level_label": "ADVERTENCIA — Intervención Preventiva",
    "score_trend": "ascending",
    "frames_in_risk_zone": 6
  },
  "location": {
    "camera_id": "CAM-02-ANDEN-1",
    "camera_display_name": "Cámara 02 — Andén 1, Línea 2",
    "station_name": "Estación Baquedano",
    "zone_triggered": "yellow_zone",
    "position_in_frame": {
      "x_normalized": 0.34,
      "y_normalized": 0.73
    }
  },
  "visual_evidence": {
    "frame_capture_url": "/api/incidents/46/capture",
    "capture_available": true,
    "skeleton_overlay_available": true
  },
  "automated_description": "Individuo ID 15 detectado en zona de advertencia con comportamiento oscilante y postura inestable. El score de riesgo presenta tendencia ascendente. Situación bajo monitoreo activo.",
  "recommended_action": {
    "action_code": "MONITOR_PREVENTIVE",
    "action_description": "Incrementar la frecuencia de monitoreo. Alertar al personal de andén para acercamiento preventivo no invasivo.",
    "auto_assignment_triggered": false,
    "assignment_endpoint": "/api/personnel/assign"
  },
  "notification": {
    "alert_sound": "warning",
    "ui_color": "#F97316",
    "ui_priority": 2,
    "requires_acknowledgment": false
  }
}
```

### 4.5 Consumo del WebSocket en el Frontend (React)

El siguiente fragmento ilustra el patrón de consumo del canal WebSocket en el dashboard React, usando un hook personalizado:

```javascript
// src/hooks/useAlertWebSocket.js

import { useEffect, useRef, useState } from "react";

export function useAlertWebSocket() {
  const [latest_alert, set_latest_alert] = useState(null);
  const [connection_status, set_connection_status] = useState("disconnected");
  const ws_ref = useRef(null);

  useEffect(() => {
    const ws_url = import.meta.env.VITE_WS_ALERTS_URL;
    const socket = new WebSocket(ws_url);
    ws_ref.current = socket;

    socket.onopen = () => {
      set_connection_status("connected");
    };

    socket.onmessage = (event) => {
      const alert_event = JSON.parse(event.data);
      if (alert_event.event_type === "risk_alert") {
        set_latest_alert(alert_event);
      }
    };

    socket.onclose = () => {
      set_connection_status("disconnected");
    };

    socket.onerror = () => {
      set_connection_status("error");
    };

    return () => {
      socket.close();
    };
  }, []);

  return { latest_alert, connection_status };
}
```

### 4.6 Resumen del Contrato WebSocket

| Atributo | Valor |
|---|---|
| **URI** | `ws://{host}:{port}/api/alerts/ws` |
| **Protocolo** | WebSocket (RFC 6455) |
| **Dirección del flujo** | Unidireccional: Servidor → Cliente (Push) |
| **Formato de mensajes** | JSON en `snake_case` |
| **Codificación** | UTF-8 |
| **Condición de emisión** | `risk_score >= 0.60` (Naranja) o `risk_score >= 0.85` (Rojo) |
| **Latencia objetivo** | < 1 segundo desde detección hasta recepción en cliente (RNF-01) |
| **Reconexión** | Responsabilidad del cliente. Se recomienda backoff exponencial con máximo de 5 reintentos. |
| **Heartbeat (ping/pong)** | Gestionado automáticamente por FastAPI/Starlette con intervalo de 30 segundos. |

---

*Documento generado para el hito universitario **Actividad 4: Contrato API** — Proyecto Andén Seguro.*
*Versión `1.0.0` — Vigente desde `24-05-2026`.*
