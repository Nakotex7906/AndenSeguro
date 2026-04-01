# Sistema de Monitoreo de Incidencias con IA

## Descripción del Proyecto
Andén Seguro es un sistema de monitoreo inteligente basado en el análisis de cámaras de seguridad. Su propósito principal es la detección temprana y prevención de posibles intentos de suicidio en las estaciones de metro.

Mediante el uso de algoritmos de visión artificial, el sistema evalúa el flujo de video en tiempo real para identificar comportamientos de riesgo, permitiendo alertar de manera oportuna e inmediata a los equipos de emergencia. El enfoque central es salvaguardar la vida de los transeúntes, garantizando al mismo tiempo el estricto cumplimiento de las normativas de privacidad de los pasajeros.

### Objetivo General
Diseñar e implementar un sistema de monitoreo inteligente basado en cámaras de seguridad para la detección temprana y prevención de posibles intentos de suicidio, con el fin de alertar oportunamente a los equipos de emergencia y salvaguardar la vida de los transeúntes.

---

## Equipo y Roles

| Nombre | Rol | Responsabilidades Principales |
| :--- | :--- | :--- |
| **Ignacio Essus** | **Líder de Proyecto** | Gestión de hitos, integración de módulos. |
| **Guillermo Salgado** | **Backend Developer** | Lógica de IA (Python), gestión de alertas y API. |
| **Waldo Alonso Chavez** | **Frontend Developer** | Dashboard de monitoreo (React), UI/UX y WebSockets. |

---

El Estándar de Commits (Conventional Commits + ID)
Utilizaremos el formato <tipo>(<alcance>): <descripción> #<ID_Tarea>. Esto permite que, al hacer el push, la actividad aparezca automáticamente en la tarea de ClickUp.

Tipos de Commits sugeridos:

feat: Una nueva funcionalidad (ej. el algoritmo de detección). 

fix: Corrección de un error o bug. 

docs: Cambios solo en la documentación o Javadoc.

refactor: Cambio en el código que no corrige un error ni añade una función (limpieza de code smells).

test: Añadir o corregir pruebas.

Ejemplos prácticos para el equipo:

Ignacio (IA/Core): feat(ia): implementacion de logica LSTM para deteccion de crisis #8642abc 
Guillermo (Backend): feat(db): crear tabla IncidentLog para historial de incidencias #8642def 
Alonso (Frontend): feat(ui): diseño de interfaz para Nivel Rojo con alerta sonora #8642ghi

Diagrama De Clases 

classDiagram
    class PedestrianDetector {
        -String model_path = "yolov5s.pt"
        +detect(frame: Image) List~Detection~
    }
    class DeepSortTracker {
        +update(detections: List) List~TrackedPerson~
        -assign_id(person: Detection) int
    }
    class Skeletonizer {
        +extract_keypoints(person: TrackedPerson) PoseVector
    }
    class RiskAnalyzer {
        -float risk_threshold = 0.85
        -LSTM_Model model
        +predict_risk(sequence: List~PoseVector~) float
    }
    class AlertManager {
        +process_risk(score: float)
        -notify_security(level: String)
        -trigger_automated_brake()
    }
    class Incident {
        +String incident_id
        +DateTime timestamp
        +float risk_score
        +save_to_db()
    }

    PedestrianDetector ..> DeepSortTracker : detecta [cite: 28-29]
    DeepSortTracker ..> Skeletonizer : rastrea [cite: 29-30]
    Skeletonizer ..> RiskAnalyzer : alimenta [cite: 30-31]
    RiskAnalyzer --> AlertManager : notifica si > 0.85 
    AlertManager --> Incident : registra evento [cite: 69-71]
