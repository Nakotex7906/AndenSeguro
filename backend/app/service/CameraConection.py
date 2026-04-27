import os
import cv2
import json
import numpy as np
import requests
from ultralytics import YOLO
from dotenv import load_dotenv

# Cargar variables de entorno (.env debe estar en la misma carpeta o raíz)
load_dotenv()

# Configuración de la API de Ezviz
APP_KEY    = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
SERIAL     = os.getenv("SERIAL")
BASE_URL   = os.getenv("BASE_URL", "https://isaopen.ezvizlife.com")

CONF_THRESHOLD = 0.5
CONFIG_FILE = "config.json"

# Estado de la configuración de zonas
config_mode = 'YELLOW'
yellow_points = []
red_points = []
mouse_pos = (0, 0)

def get_stream_url():
    """Obtiene la URL de transmisión usando la API de Ezviz."""
    if not all([APP_KEY, APP_SECRET, SERIAL]):
        raise Exception("Faltan credenciales (APP_KEY, APP_SECRET o SERIAL) en el archivo .env")

    # 1. Obtener Token
    resp_token = requests.post(f"{BASE_URL}/api/lapp/token/get",
        data={"appKey": APP_KEY, "appSecret": APP_SECRET})
    token_data = resp_token.json()
    
    if token_data.get("code") != "200":
        raise Exception(f"Error de autenticación API: {token_data.get('msg')}")
    
    token = token_data["data"]["accessToken"]
    
    # 2. Obtener URL del Stream
    resp_url = requests.post(f"{BASE_URL}/api/lapp/v2/live/address/get",
        data={
            "accessToken": token, 
            "deviceSerial": SERIAL,
            "channelNo": 1, 
            "protocol": 3, 
            "quality": 1   
        })
    url_data = resp_url.json()
    
    if url_data.get("code") != "200":
        raise Exception(f"Error al obtener dirección de stream: {url_data.get('msg')}")
        
    return url_data["data"]["url"]

def load_config():
    """Carga los polígonos guardados."""
    global yellow_points, red_points, config_mode
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                y_pts = data.get("yellow_points", [])
                r_pts = data.get("red_points", [])
                
                if y_pts:
                    yellow_points = [tuple(p) for p in y_pts]
                if r_pts:
                    red_points = [tuple(p) for p in r_pts]
                    config_mode = 'DONE'
                elif y_pts:
                    config_mode = 'RED'
        except Exception as e:
            print(f"[ERROR] No se pudo leer la configuración: {e}")

def save_config():
    """Guarda ambos polígonos en el archivo JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "yellow_points": yellow_points,
            "red_points": red_points
        }, f)

def mouse_click(event, x, y, flags, param):
    """Maneja el dibujo secuencial de los dos polígonos."""
    global yellow_points, red_points, config_mode, mouse_pos

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if config_mode == 'YELLOW':
            yellow_points.append((x, y))
        elif config_mode == 'RED':
            red_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if config_mode == 'YELLOW' and len(yellow_points) >= 3:
            config_mode = 'RED'
            print("[INFO] Zona Amarilla definida. Ahora define la Zona Roja.")
        elif config_mode == 'RED' and len(red_points) >= 3:
            config_mode = 'DONE'
            save_config()
            print("[INFO] Configuración completada y guardada.")

def draw_polygon_ui(frame, points, color, is_closed):
    """Dibuja el polígono en proceso de creación."""
    if len(points) > 0:
        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_array], is_closed, color, 2)
        if not is_closed:
            cv2.line(frame, points[-1], mouse_pos, color, 1)
            for p in points:
                cv2.circle(frame, p, 3, color, -1)

def main():
    global yellow_points, red_points, config_mode, mouse_pos

    print("[INFO] Cargando modelo YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    load_config()

    print("[INFO] Conectando con Ezviz Open Platform...")
    try:
        stream_url = get_stream_url()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Nombres de ventana sin tilde para evitar el bug de clics de OpenCV en Windows
    cv2.namedWindow("AndenSeguro - Deteccion")
    cv2.setMouseCallback("AndenSeguro - Deteccion", mouse_click)

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            print("[ADVERTENCIA] Error al recuperar frame. Reintentando...")
            continue

        # 1. VISUALIZACIÓN DE ZONAS
        if config_mode == 'YELLOW':
            draw_polygon_ui(frame, yellow_points, (0, 215, 255), False)
        else:
            overlay = frame.copy()
            pts_y = np.array(yellow_points, np.int32)
            cv2.fillPoly(overlay, [pts_y], (0, 215, 255))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [pts_y], True, (0, 215, 255), 2)

        if config_mode == 'RED':
            draw_polygon_ui(frame, red_points, (0, 0, 255), False)
        elif config_mode == 'DONE':
            overlay = frame.copy()
            pts_r = np.array(red_points, np.int32)
            cv2.fillPoly(overlay, [pts_r], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [pts_r], True, (0, 0, 255), 2)

        # 2. DETECCIÓN Y EVALUACIÓN
        if config_mode == 'DONE':
            results = model(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    feet = ( (x1 + x2) // 2, y2 )
                    
                    in_red = cv2.pointPolygonTest(np.array(red_points, np.int32), feet, False) >= 0
                    in_yellow = cv2.pointPolygonTest(np.array(yellow_points, np.int32), feet, False) >= 0
                    
                    if in_red:
                        color, label = (0, 0, 255), "PELIGRO"
                    elif in_yellow:
                        color, label = (0, 215, 255), "PRECAUCION"
                    else:
                        color, label = (0, 255, 0), "SEGURO"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("AndenSeguro - Deteccion", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            config_mode = 'YELLOW'
            yellow_points, red_points = [], []
            if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
            print("[INFO] Configuración borrada.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()