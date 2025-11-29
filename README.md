# Ground Point Predictor

Ein neuronales Netzwerk zur Vorhersage von Bodenkontaktpunkten für Objekte (z.B. Straßenschilder) basierend auf der YOLO-Pose Architektur.

## Übersicht

Dieses Projekt extrahiert die Keypoint-Vorhersage-Logik aus YOLO-Pose und wendet sie auf die Aufgabe der Bodenpunkt-Vorhersage an. Das Modell kann mit **jedem beliebigen YOLO-Detector** kombiniert werden.

### Anwendungsfall

```
┌─────────────────┐          ┌─────────────────────────┐
│  Beliebiges     │  Boxes   │   GroundPointPredictor  │
│  YOLO Modell    │ ───────► │   (dieses Projekt)      │ ──► Bodenpunkte
│  (Detection)    │          │                         │
└─────────────────┘          └─────────────────────────┘
```

## Architektur

### 1. Backbone - Feature Extraction
```
Eingabebild (640×640×3) → ResNet34 → Feature Map (20×20×512)
```
Das **gesamte Bild** wird verarbeitet. Das Netzwerk sieht:
- Das Objekt (Schild)
- Den Boden
- Den Horizont
- Perspektivische Information

### 2. ROI-Align - Box-spezifische Features
```
Feature Map + Bounding Boxes → ROI-Align → Features pro Box (7×7×512)
```
Für jede Bounding Box werden die Features **an dieser Stelle** extrahiert.
Dies ist effizienter als separate Crops, da das Backbone nur einmal läuft.

### 3. Keypoint Head - Offset Vorhersage
```python
# Direkt aus YOLO-Pose (ultralytics/nn/modules/head.py, Zeile 353)
Conv(512→256, 3×3) → Conv(256→256, 3×3) → Conv(256→2, 1×1)
```
Der Keypoint Head gibt **2 Werte** pro Box aus: `(offset_x, offset_y)`

### 4. Decode - Absolute Koordinaten
```python
# YOLO-Pose Style (head.py, Zeile 382-383)
ground_point_x = box_center_x + offset_x * 2.0 * box_width
ground_point_y = box_center_y + offset_y * 2.0 * box_height
```

Der Offset wird relativ zur Box interpretiert:
- `offset = 0` → Bodenpunkt liegt im Box-Zentrum
- `offset = 0.5` → Bodenpunkt liegt 1 Boxbreite/-höhe entfernt
- `offset = 1.0` → Bodenpunkt liegt 2 Boxbreiten/-höhen entfernt

### 5. Loss Funktion - OKS-Style
```python
# Aus YOLO-Pose (ultralytics/utils/loss.py, Zeile 187-191)
d = (pred_x - gt_x)² + (pred_y - gt_y)²      # Euklidische Distanz
e = d / (2σ² × box_area × 2)                  # Normalisiert auf Box-Größe
loss = 1 - exp(-e)                            # OKS-Style Loss
```

Größere Boxen dürfen größere absolute Fehler haben.

## Projektstruktur

```
ground_point_pipeline/
├── config.py                    # Konfiguration
├── train.py                     # Training starten
├── predict.py                   # Inferenz starten
├── models/
│   ├── __init__.py
│   └── ground_point_predictor.py   # Hauptmodell
├── data/
│   ├── __init__.py
│   └── dataset.py               # Dataset-Klasse
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training-Loop
│   └── losses.py                # Loss-Funktionen
├── inference/
│   ├── __init__.py
│   └── predictor.py             # Inferenz-Pipeline
├── utils/
│   ├── __init__.py
│   └── visualization.py         # Visualisierung
├── weights/
│   └── best_model.pt            # Trainierte Gewichte
└── README.md                    # Diese Datei
```

## Installation

```bash
# Abhängigkeiten (in bestehendem Environment)
pip install torch torchvision opencv-python matplotlib tqdm numpy
```

## Verwendung

### Training

```bash
# Training mit Standardkonfiguration
python train.py

# Mit angepassten Parametern
python train.py --epochs 100 --backbone resnet50 --batch-size 16

# Alle Optionen anzeigen
python train.py --help
```

### Inferenz

```bash
# Auf einzelnem Bild mit Bounding Boxes
python predict.py --image bild.jpg --boxes "100,200,300,400;150,100,250,350"

# Auf Test-Datensatz mit Visualisierung
python predict.py --test-data --visualize

# Mit eigenen Gewichten
python predict.py --weights pfad/zu/model.pt --image bild.jpg
```

### Python API

```python
from inference import GroundPointInference

# Predictor laden
predictor = GroundPointInference(
    weights_path="weights/best_model.pt",
    backbone="resnet34",
    device="cuda"
)

# Mit beliebigen Bounding Boxes (z.B. von YOLO)
boxes = [[100, 200, 300, 400], [150, 100, 250, 350]]  # xyxy Format
ground_points = predictor.predict(image, boxes)

# Mit YOLO Integration
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")
results = yolo(image)
boxes, ground_points = predictor.predict_with_yolo(image, results[0])
```

## Ergebnisse

Trainiert auf Mapillary Traffic Sign Dataset:

| Metrik | Wert |
|--------|------|
| Mean Error | 20.3 px |
| **Median Error** | **9.8 px** |
| 90th Percentile | 49.2 px |
| 95th Percentile | 79.7 px |

Bei 640×640 Bildern entspricht der Median-Fehler von ~10px etwa **1.5% der Bildgröße**.

### Vergleich mit Baselines

| Methode | Median Error | Beschreibung |
|---------|--------------|--------------|
| Geometry-MLP | ~30 px | Nur Box-Koordinaten, kein Bild |
| **GroundPointPredictor** | **9.8 px** | Mit Bild-Features |

Das Modell ist **3× genauer** als reine Geometrie-Ansätze, da es:
- Die Pfostenlänge aus dem Bild erkennt
- Perspektivische Information nutzt
- Den Boden im Kontext sieht

## Datenformat

Die Labels müssen im YOLO-Pose Format vorliegen:

```
# class x_center y_center width height keypoint_x keypoint_y [visibility]
0 0.5 0.3 0.1 0.15 0.5 0.8
```

Alle Koordinaten sind normalisiert (0-1).

## Referenzen

- YOLO-Pose: `ultralytics/nn/modules/head.py` (Pose class)
- Keypoint Loss: `ultralytics/utils/loss.py` (KeypointLoss)
- ROI-Align: `torchvision.ops.roi_align`

## Lizenz

Dieses Projekt basiert auf Code von Ultralytics (AGPL-3.0).


