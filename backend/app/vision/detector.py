"""
Detector de personas — Pipeline de IA.

Encapsula la inicialización del modelo YOLOv8-pose y la inferencia
con tracking integrado. Retorna detecciones tipadas.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Resultado de una detección individual."""

    box_xyxy: tuple[int, int, int, int]
    track_id: Optional[int]
    confidence: float
    keypoints_xy: Optional[np.ndarray]  # Shape: (17, 2) para COCO keypoints


class PersonDetector:
    """Inicializa el modelo YOLO y ejecuta la detección con tracking."""

    def __init__(self, model_name: str, imgsz: int, conf_threshold: float):
        logger.info(f"Cargando modelo {model_name}...")
        self.model = YOLO(model_name)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

        # Selección automática de hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"NVIDIA CUDA DETECTADO: Usando GPU ({gpu_name})")
            if model_name.endswith(".pt"):
                self.model.to("cuda")
        else:
            logger.warning(
                "CUDA NO DETECTADO: Usando CPU. "
                "El video se lagueará con modelos grandes."
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Ejecuta detección + tracking sobre un frame y retorna las detecciones."""
        results = self.model.track(
            frame,
            classes=[0],
            conf=self.conf_threshold,
            persist=True,
            verbose=False,
            imgsz=self.imgsz,
            half=True,
            device="0",
            tracker="bytetrack.yaml",
        )

        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints if hasattr(result, "keypoints") else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                conf = float(box.conf[0])

                kpts = None
                if (
                    keypoints is not None
                    and keypoints.xy is not None
                    and len(keypoints.xy) > i
                ):
                    raw = keypoints.xy[i]
                    kpts = (
                        raw.cpu().numpy()
                        if hasattr(raw, "cpu")
                        else np.array(raw)
                    )

                detections.append(
                    Detection(
                        box_xyxy=(x1, y1, x2, y2),
                        track_id=track_id,
                        confidence=conf,
                        keypoints_xy=kpts,
                    )
                )

        return detections
