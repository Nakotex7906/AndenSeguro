from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. Cargar variables de entorno al inicio absoluto
load_dotenv()

# 2. Importar rutas después de cargar el entorno
from app.api.routes import stream

app = FastAPI(
    title="Andén Seguro API",
    description="Sistema de monitoreo de estaciones mediante visión artificial",
    version="1.0.0"
)

# Configuración de CORS para el frontend en React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de rutas
app.include_router(stream.router, prefix="/api/stream", tags=["Streaming"])

@app.get("/")
def health_check():
    return {"status": "Operativo", "project": "Andén Seguro"}