# Andén Seguro — Historias de Usuario

> **Versión:** `1.0.0` | **Fecha:** `03-06-2026` 

Historias de usuario organizadas por rol. Cada historia referencia el requerimiento funcional o no funcional que la origina para mantener trazabilidad directa con la especificación del sistema.

---

## Tabla de Contenidos

1. [Operador de Estación](#1-operador-de-estación)
3. [Personal de Respuesta en Campo](#2-personal-de-respuesta-en-campo)

---

## 1. Operador de Estación

El operador de estación es el usuario que monitorea el dashboard web en tiempo real desde la sala de control. Su objetivo principal es detectar situaciones de riesgo y coordinar la respuesta oportuna.

---

### HU-OE-01 — Visualizar el feed de cámara en tiempo real

> **Referencia:** RF-08

**Como** operador de estación,  
**quiero** ver en el dashboard el feed de video de todas las cámaras activas con las detecciones superpuestas,  
**para** poder identificar visualmente el nivel de riesgo de cada persona en el andén sin necesidad de interpretar datos técnicos.

---

### HU-OE-02 — Recibir alerta de nivel Naranja

> **Referencia:** RF-07, RNF-01

**Como** operador de estación,  
**quiero** recibir una notificación visual y sonora de nivel amarillo cuando una persona presente un score de riesgo entre 0.60 y 0.84,  
**para** poder iniciar una intervención preventiva antes de que la situación escale a una emergencia.

---

### HU-OE-03 — Recibir alerta de nivel Rojo con activación automática de protocolo

> **Referencia:** RF-06, RF-07, RF-10, RNF-01, RNF-06

**Como** operador de estación,  
**quiero** recibir una alerta crítica de nivel Rojo con captura visual del evento, descripción automática de la situación y ubicación exacta dentro de la red cuando el score de riesgo supere 0.85,  
**para** poder confirmar el incidente y coordinar la respuesta de emergencia.

---

### HU-OE-04 — Consultar la descripción automática del incidente

> **Referencia:** RF-10

**Como** operador de estación,  
**quiero** leer una descripción generada automáticamente que explique el comportamiento detectado (duración, postura, dirección de movimiento),  
**para** entender el contexto del incidente sin necesidad de revisar el video frame a frame.

---

### HU-OE-05 — Ver el nivel de riesgo por persona sin datos biométricos

> **Referencia:** RF-08, RNF-04

**Como** operador de estación,  
**quiero** ver el nivel de riesgo asignado a cada persona detectada en el feed mediante indicadores de color (verde, naranja, rojo) y un ID de rastreo anónimo,  
**para** tomar decisiones informadas sin que el sistema revele ni almacene datos biométricos de los usuarios.

---

### HU-OE-06 — Confirmar o desestimar una alerta activa

> **Referencia:** RF-07, RF-09

**Como** operador de estación,  
**quiero** poder marcar una alerta activa como resuelta o como falso positivo desde el dashboard,  
**para** que el historial de incidencias quede correctamente auditado y el sistema pueda medir su tasa de precisión.

---

### HU-OE-07 — Verificar el estado operativo del pipeline de IA

> **Referencia:** RF-08, RNF-01

**Como** operador de estación,  
**quiero** ver en el dashboard métricas como FPS, latencia de inferencia y estado de la conexión con la cámara,  
**para** detectar degradación del sistema antes de que afecte la capacidad de detección en tiempo real.

---

## 2. Personal de Respuesta en Campo

El personal de respuesta en campo opera desde dispositivos móviles (app Expo/React Native). Su necesidad central es recibir instrucciones claras y confirmables en el menor tiempo posible para actuar ante una emergencia.

---

### HU-PC-01 — Recibir notificación push de alerta en el dispositivo móvil

> **Referencia:** RF-10, RNF-06

**Como** personal de respuesta en campo,  
**quiero** recibir una notificación inmediata en mi dispositivo móvil en el momento en que se detecte un incidente activo,  
**para** poder reaccionar dentro de la ventana crítica de 30 segundos establecida por el protocolo de emergencia.

---

### HU-PC-02 — Ver el detalle del incidente asignado

> **Referencia:** RF-10, RF-11

**Como** personal de respuesta en campo,  
**quiero** acceder desde mi dispositivo móvil al nivel de alerta, la descripción del comportamiento detectado y la ubicación exacta del incidente dentro del anden,  
**para** llegar al lugar correcto con el contexto necesario para intervenir de forma efectiva.

---

### HU-PC-03 — Confirmar la recepción y atención del incidente

> **Referencia:** RF-09, RF-11

**Como** personal de respuesta en campo,  
**quiero** poder confirmar desde la app móvil que recibí la asignación y que estoy en camino al punto del incidente,  
**para** que el operador de estación y el administrador puedan monitorear el estado de respuesta en tiempo real.

---

### HU-PC-04 — Acceder a recursos de apoyo disponibles para la intervención

> **Referencia:** RF-11

**Como** personal de respuesta en campo,  
**quiero** ver en la app la lista de redes de apoyo y recursos disponibles asociados al incidente (contactos, protocolos específicos),  
**para** contar con el respaldo necesario durante la intervención sin depender de comunicaciones externas adicionales.

---

### HU-PC-05 — Marcar el incidente como resuelto desde campo

> **Referencia:** RF-09, RF-11

**Como** personal de respuesta en campo,  
**quiero** registrar desde mi dispositivo móvil que el incidente fue atendido y resuelto,  
**para** que el sistema actualice el historial auditado y libere mi disponibilidad operativa para futuras asignaciones.

---

### HU-PC-06 — Recibir asignación considerando mi disponibilidad actual

> **Referencia:** RF-11

**Como** personal de respuesta en campo,  
**quiero** que el sistema solo me asigne incidentes cuando estoy disponible operativamente y que aplique escalamiento progresivo si no puedo responder a tiempo,  
**para** garantizar que siempre haya una respuesta efectiva al incidente sin sobrecargar a un solo operador.

---