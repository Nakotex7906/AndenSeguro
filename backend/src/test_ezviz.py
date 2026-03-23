import os
import requests
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
 
APP_KEY    = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
SERIAL     = os.getenv("SERIAL")
BASE_URL   = os.getenv("BASE_URL") 

# Umbral de confianza y zona de peligro (% del ancho de imagen)
CONF_THRESHOLD  = 0.5
DANGER_ZONE_PCT = 0.15   # 15% inferior de la imagen = zona cerca del borde

def get_stream_url():
    resp = requests.post(
        f"{BASE_URL}/api/lapp/token/get",
        data={"appKey": APP_KEY, "appSecret": APP_SECRET}
    )
    access_token = resp.json()["data"]["accessToken"]
    resp = requests.post(
        f"{BASE_URL}/api/lapp/v2/live/address/get",
        data={
            "accessToken": access_token,
            "deviceSerial": SERIAL,
            "channelNo": 1,
            "protocol": 3,
            "quality": 1
        }
    )
    return resp.json()["data"]["url"]

def draw_danger_zone(frame):
    """Dibuja la zona de peligro (borde inferior)."""
    h, w = frame.shape[:2]
    danger_y = int(h * (1 - DANGER_ZONE_PCT))
    cv2.rectangle(frame, (0, danger_y), (w, h), (0, 0, 255), 2)
    cv2.putText(frame, "ZONA DE RIESGO", (10, danger_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return danger_y

def is_in_danger_zone(box, danger_y):
    """Verifica si el pie de la persona está en la zona de riesgo."""
    x1, y1, x2, y2 = box
    foot_y = y2   # parte inferior del bounding box
    return foot_y >= danger_y

def main():
    print("⬇️  Cargando modelo YOLOv8...")
    model = YOLO("yolov8n.pt")   # nano = más rápido, ideal para tiempo real
    print("✅ Modelo cargado")

    url = get_stream_url()
    print(f"🎥 Conectando al stream...")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ No se pudo abrir el stream")
        return

    alert_frames = 0   # contador para evitar falsas alarmas

    print("▶️  Detectando personas... 'q' para salir")

    while True:
        cap.grab()
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            cap.release()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Dibujar zona de riesgo
        danger_y = draw_danger_zone(frame)

        # Detección con YOLO (solo clase 0 = personas)
        results = model(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)

        alert = False
        person_count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                person_count += 1

                in_danger = is_in_danger_zone((x1, y1, x2, y2), danger_y)

                # Color: rojo si está en zona de riesgo, verde si no
                color = (0, 0, 255) if in_danger else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'⚠ RIESGO' if in_danger else 'OK'} {conf:.0%}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if in_danger:
                    alert = True

        # Sistema anti-falsas alarmas: alerta solo si persiste 10 frames
        if alert:
            alert_frames += 1
        else:
            alert_frames = max(0, alert_frames - 1)

        if alert_frames >= 10:
            cv2.putText(frame, "!!! ALERTA !!!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            print(f"🚨 ALERTA: persona en zona de riesgo ({alert_frames} frames)")

        # Info en pantalla
        cv2.putText(frame, f"Personas: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AndenSeguro - Deteccion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()