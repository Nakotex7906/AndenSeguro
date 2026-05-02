import os
import cv2
import json
import numpy as np
import requests

class CameraService:
    def __init__(self, model, app_key, app_secret, serial, base_url):
        self.model = model
        self.app_key = app_key
        self.app_secret = app_secret
        self.serial = serial
        self.base_url = base_url
        self.config_file = "config.json"
        self.conf_threshold = 0.5
        
        # Estado de zonas (Ahora se guardan como porcentajes de dimensiones 0.0 a 1.0)
        self.yellow_points = []
        self.red_points = []
        self.load_config()

    def load_config(self):
        """Carga los polígonos guardados en el archivo de configuración."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.yellow_points = [tuple(p) for p in data.get("yellow_points", [])]
                    self.red_points = [tuple(p) for p in data.get("red_points", [])]
            except Exception as e:
                print(f"[ERROR] No se pudo leer la configuración: {e}")

    def get_stream_url(self):
        """Obtiene la URL de transmisión usando la API de Ezviz."""
        if not all([self.app_key, self.app_secret, self.serial]):
            raise Exception("Faltan credenciales en el archivo .env")

        resp_token = requests.post(f"{self.base_url}/api/lapp/token/get",
            data={"appKey": self.app_key, "appSecret": self.app_secret})
        token_data = resp_token.json()
        
        if token_data.get("code") != "200":
            raise Exception(f"Error de autenticación API: {token_data.get('msg')}")
        
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
        """Generador de frames procesados para el streaming MJPEG."""
        use_mock = os.getenv("USE_MOCK_CAMERA", "False") == "True"
        
        try:
            source = 0 if use_mock else self.get_stream_url()
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"[ERROR] Fallo al iniciar captura: {e}")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break
                
            h, w = frame.shape[:2]

            # Convertir porcentajes guardados a píxeles absolutos reales del frame
            abs_yellow = [(int(p[0]*w), int(p[1]*h)) for p in self.yellow_points] if self.yellow_points else []
            abs_red = [(int(p[0]*w), int(p[1]*h)) for p in self.red_points] if self.red_points else []

            # 1. Dibujar Zonas (Polígonos)
            if abs_yellow and abs_red:
                overlay = frame.copy()
                pts_y = np.array(abs_yellow, np.int32)
                pts_r = np.array(abs_red, np.int32)
                
                cv2.fillPoly(overlay, [pts_y], (0, 215, 255))
                cv2.fillPoly(overlay, [pts_r], (0, 0, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts_y], True, (0, 215, 255), 2)
                cv2.polylines(frame, [pts_r], True, (0, 0, 255), 2)

            # 2. Detección con el modelo inyectado
            results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    feet = ( (x1 + x2) // 2, y2 )
                    
                    # Evaluación de posición
                    in_red = cv2.pointPolygonTest(np.array(abs_red, np.int32), feet, False) >= 0 if abs_red else False
                    in_yellow = cv2.pointPolygonTest(np.array(abs_yellow, np.int32), feet, False) >= 0 if abs_yellow else False
                    
                    color, label = (0, 255, 0), "SEGURO"
                    if in_red:
                        color, label = (0, 0, 255), "PELIGRO"
                    elif in_yellow:
                        color, label = (0, 215, 255), "PRECAUCION"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Codificación para streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()