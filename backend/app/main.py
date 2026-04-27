from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. CARGAR VARIABLES DE ENTORNO PRIMERO
load_dotenv()

# 2. AHORA SÍ IMPORTAR LAS RUTAS
from app.api.routes import stream 

app = FastAPI(title="Andén Seguro API", version="1.0.0")

# Configuración estricta de CORS para permitir la conexión desde tu frontend en React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cambiar por el puerto de Vite (ej. http://localhost:5173) en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
app.include_router(stream.router, prefix="/api/stream", tags=["Streaming"])

@app.get("/")
def health_check():
    return {"status": "Operativo", "project": "Andén Seguro"}