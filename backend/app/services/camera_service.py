import os
import cv2
import json
import numpy as np
import requests
from ultralytics import YOLO

class CameraStreamer:
    def __init__(self, app_key, app_secret, serial, base_url):
        self.app_key = app_key
        self.app_secret = app_secret
        self.serial = serial
        self.base_url = base_url
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = 0.5
        self.config_file = "config.json"
        
        self.yellow_points = []
        self.red_points = []
        self.config_mode = 'DONE' # Asumimos que la config viene dada por ahora
        self.load_config()

    def load_config(self):
        """Carga los polígonos desde el archivo JSON o base de datos."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    if data.get("yellow_points"):
                        self.yellow_points = [tuple(p) for p in data["yellow_points"]]
                    if data.get("red_points"):
                        self.red_points = [tuple(p) for p in data["red_points"]]
            except Exception as e:
                print(f"[ERROR] No se pudo leer la configuración: {e}")

    def get_stream_url(self):
        """Obtiene la URL de transmisión de Ezviz."""
        resp_token = requests.post(f"{self.base_url}/api/lapp/token/get",
            data={"appKey": self.app_key, "appSecret": self.app_secret})
        token_data = resp_token.json()
        
        if token_data.get("code") != "200":
            raise Exception(f"Error de autenticación: {token_data.get('msg')}")
        
        token = token_data["data"]["accessToken"]
        
        resp_url = requests.post(f"{self.base_url}/api/lapp/v2/live/address/get",
            data={
                "accessToken": token, 
                "deviceSerial": self.serial,
                "channelNo": 1, 
                "protocol": 3, 
                "quality": 1   
            })
        url_data = resp_url.json()
        return url_data["data"]["url"]

    def generate_frames(self):
        """Generador que captura, procesa y emite fotogramas."""
        use_mock = os.getenv("USE_MOCK_CAMERA", "False") == "True"

        if use_mock:
            print("[INFO] MODO PRUEBA: Usando webcam local (0)...")
            # Usa 0 para tu webcam. Si prefieres probar con un video del andén, 
            # cambia el 0 por la ruta del archivo: cv2.VideoCapture("ruta/al/video.mp4")
            cap = cv2.VideoCapture(0) 
        else:
            try:
                print("[INFO] Conectando con Ezviz Open Platform...")
                stream_url = self.get_stream_url()
                cap = cv2.VideoCapture(stream_url)
            except Exception as e:
                print(f"[ERROR] {e}")
                return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while True:
            success, frame = cap.read()
            if not success:
                # Si usas un video local (.mp4) y termina, puedes reiniciar el loop aquí
                if use_mock and isinstance(cap.get(cv2.CAP_PROP_POS_FRAMES), float):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Dibujar polígonos si existen
            if self.red_points and self.yellow_points:
                overlay = frame.copy()
                pts_r = np.array(self.red_points, np.int32)
                pts_y = np.array(self.yellow_points, np.int32)
                
                cv2.fillPoly(overlay, [pts_r], (0, 0, 255))
                cv2.fillPoly(overlay, [pts_y], (0, 215, 255))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                cv2.polylines(frame, [pts_r], True, (0, 0, 255), 2)
                cv2.polylines(frame, [pts_y], True, (0, 215, 255), 2)

            # Detección YOLOv8
            results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    feet = ((x1 + x2) // 2, y2)
                    
                    in_red = cv2.pointPolygonTest(np.array(self.red_points, np.int32), feet, False) >= 0 if self.red_points else False
                    in_yellow = cv2.pointPolygonTest(np.array(self.yellow_points, np.int32), feet, False) >= 0 if self.yellow_points else False
                    
                    if in_red:
                        color, label = (0, 0, 255), "PELIGRO"
                    elif in_yellow:
                        color, label = (0, 215, 255), "PRECAUCION"
                    else:
                        color, label = (0, 255, 0), "SEGURO"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Codificar a JPEG para el streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Formato Multipart para streaming en navegadores
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')