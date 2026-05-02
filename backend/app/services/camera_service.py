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
        
        # Umbral configurable por entorno para equilibrar exigencia vs hardware (por defecto 0.25 para caza de lejos)
        self.conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
        # Resolución nativa de entrada para YOLO configurable (.env). 640 es rápido pero ciego de lejos. 1280 es exigente pero nítido
        self.imgsz = int(os.getenv("YOLO_IMGSZ", "1280"))
        
        # Estado de zonas (Ahora se guardan como porcentajes de dimensiones 0.0 a 1.0)
        self.yellow_points = []
        self.red_points = []
        self.load_config()
        
        # Estadísticas en tiempo real
        self.current_stats = {
            "total_persons": 0,
            "risk_persons": 0,
            "danger_persons": 0
        }
        
        import time
        self.track_history = {} # Guardaremos {id: entry_time} para riesgo por tiempo

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
        debug_stream_url = os.getenv("DEBUG_STREAM_URL", "")
        
        try:
            if debug_stream_url:
                source = debug_stream_url
                print(f"[INFO] Usando stream de debug: {source}")
                # Extraer URL directa si es un enlace de YouTube
                if "youtube.com" in source or "youtu.be" in source:
                    try:
                        import yt_dlp
                        print("[INFO] Extrayendo URL cruda de YouTube con yt-dlp...")
                        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(source, download=False)
                            source = info['url']
                    except ImportError:
                        print("[ERROR] Falta instalar yt-dlp. Ejecuta: pip install yt-dlp")
            elif use_mock:
                source = 0
                print("[INFO] Usando cámara web local (mock)")
            else:
                source = self.get_stream_url()
                print("[INFO] Usando stream de Ezviz")
                
            cap = cv2.VideoCapture(source)
            if not debug_stream_url:
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

            # 2. Detección, Pose y Tracking (Detectará la VRAM automáticamente si es ONNX)
            results = self.model.track(frame, classes=[0], conf=self.conf_threshold, persist=True, verbose=False, imgsz=self.imgsz, half=True, device='0')
            
            total_personas = 0
            personas_riesgo = 0
            personas_peligro = 0
            import time
            current_time = time.time()
            debug_pose = os.getenv("DEBUG_POSE", "False") == "True"
            
            for result in results:
                # Extraemos cajas y keypoints si es YOLOv8-pose
                boxes = result.boxes
                keypoints = result.keypoints if hasattr(result, 'keypoints') else None
                
                for i, box in enumerate(boxes):
                    total_personas += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    feet = ( (x1 + x2) // 2, y2 )
                    box_id = int(box.id[0]) if box.id is not None else "N/A"
                    
                    # Evaluación de posición en base a polígonos
                    in_red = cv2.pointPolygonTest(np.array(abs_red, np.int32), feet, False) >= 0 if abs_red else False
                    in_yellow = cv2.pointPolygonTest(np.array(abs_yellow, np.int32), feet, False) >= 0 if abs_yellow else False
                    
                    color, label = (0, 255, 0), "SEGURO"
                    is_risk = False
                    is_danger = False

                    # 3. Factor Postural y Lógica (Bounding Box + Pose Keypoints)
                    bad_posture = False
                    if keypoints and keypoints.xy is not None and len(keypoints.xy) > i:
                        kpts = keypoints.xy[i]
                        if len(kpts) > 12:  # Asegurar que hay suficientes keypoints (0=nariz, 11/12=caderas)
                            # Evaluamos inclinación o cabeza gacha comparando Y de la nariz con cadera
                            head_y = kpts[0][1]
                            hip_y = (kpts[11][1] + kpts[12][1]) / 2 if kpts[11][1] > 0 else y2
                            # Si la cabeza está más abajo o igual que la cadera
                            if head_y > 0 and hip_y > 0 and head_y >= (hip_y - 20):
                                bad_posture = True
                                
                            # DEBUG: Visualización estructurada del esqueleto sobre la persona
                            if debug_pose:
                                for kp in kpts:
                                    px, py = int(kp[0]), int(kp[1])
                                    if px > 0 and py > 0:
                                        cv2.circle(frame, (px, py), 3, (255, 0, 255), -1) # Puntos fucsia

                                # Eje de tracking postural: Línea visible desde la cabeza hasta la cadera
                                head_x, head_y_int = int(kpts[0][0]), int(kpts[0][1])
                                hip_x = int((kpts[11][0] + kpts[12][0]) / 2)
                                hip_y_int = int((kpts[11][1] + kpts[12][1]) / 2)
                                
                                if head_x > 0 and head_y_int > 0 and hip_y_int > 0:
                                    color_eje = (0, 165, 255) if bad_posture else (0, 255, 255) # Naranja alerta, Amarillo normal
                                    cv2.line(frame, (head_x, head_y_int), (hip_x, hip_y_int), color_eje, 2)
                                    if bad_posture:
                                        cv2.putText(frame, "POSTURA CRITICA", (head_x - 30, head_y_int - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                    # 4. Factor Temporal y Merodeo
                    if in_yellow or in_red:
                        if box_id not in self.track_history:
                            self.track_history[box_id] = current_time
                        time_in_zone = current_time - self.track_history[box_id]
                    else:
                        # Reseteamos su historial si sale de la zona
                        if box_id in self.track_history:
                            del self.track_history[box_id]
                        time_in_zone = 0

                    if in_red:
                        color, label = (0, 0, 255), "PELIGRO"
                        is_danger = True
                    elif in_yellow:
                        if time_in_zone > 5.0 or bad_posture: # Más de 5 segundos inmóvil en línea amarilla o mala postura
                            color, label = (0, 0, 255), "ALTO RIESGO"
                            is_danger = True
                        else:
                            color, label = (0, 215, 255), "PRECAUCION"
                            is_risk = True

                    if is_danger: personas_peligro += 1
                    if is_risk: personas_riesgo += 1

                    # Dibujamos bounding box, postura (opcional), e ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{box_id} {label}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.current_stats["total_persons"] = total_personas
            self.current_stats["risk_persons"] = personas_riesgo
            self.current_stats["danger_persons"] = personas_peligro

            # 3. Codificación para streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()