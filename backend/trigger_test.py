"""
Script de simulación de incidentes mediante HTTP — Andén Seguro.

Efectúa una llamada REST hacia el servidor local para inyectar una
alerta crítica sin violar los límites de aislamiento de procesos.
"""

import requests
import sys

def trigger_server_incident():
    """
    Realiza una petición POST al endpoint de pruebas del servidor FastAPI.
    """
    target_ip = "192.168.1.86"  # Tu IP local de Windows
    target_port = "8000"
    url = f"http://{target_ip}:{target_port}/api/alerts/inject-mock"

    print(f"[CLIENTE] Tocando puerta del servidor en: {url} ...")
    
    try:
        response = requests.post(url, timeout=5.0)
        
        if response.status_code == 200:
            print("\n[OK] ¡Petición aceptada con éxito por el servidor!")
            print(f"[RESPUESTA] {response.json().get('message')}")
            print("\nRevisa ahora la pantalla de alertas activas en tu aplicación móvil.")
        else:
            print(f"\n[ERROR] El servidor respondió con código: {response.status_code}")
            print(f"[DETALLE] {response.text}")

    except requests.exceptions.RequestException as error:
        print(f"\n[FALLO DE RED] No se pudo conectar con FastAPI: {error}")
        print("Asegúrate de que uvicorn esté corriendo con '--host 0.0.0.0'.")

if __name__ == "__main__":
    trigger_server_incident()