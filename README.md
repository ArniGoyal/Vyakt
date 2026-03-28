# Vyakt — व्यक्त

[![GitHub license](https://img.shields.io/github/license/ArniGoyal/Vyakt)](./LICENSE)
> *Vyakt* (व्यक्त) means **expressed** or **made visible** in Sanskrit.  
> This project makes the ancient language of Kathak hand gestures visible and understandable through AI.

---

## Overview

**Vyakt** is a real-time computer vision system that detects and classifies Kathak hasta mudras (hand gestures) using a webcam. It identifies whether a mudra is **Asamyukta** (single-hand) or **Sanyukta** (both-hands) and displays its classical meaning on screen.

Built using YOLOv8s and trained on the [Dance Mudra dataset](https://universe.roboflow.com/mudras/dance-mudra) from Roboflow Universe, Vyakt bridges the gap between classical Indian dance and modern AI.

---

## Model Performance

| Metric | Score |
|---|---|
| **mAP50** | **0.950** |
| **mAP50-95** | **0.709** |
| **Precision** | **0.906** |
| **Recall** | **0.916** |

### Per Class Results

| Mudra | Type | mAP50 | Status |
|---|---|---|---|
| alapadmam | Asamyukta | 0.995 | ✅ |
| anjali | Sanyukta | 0.995 | ✅ |
| ardhachandran | Asamyukta | 0.995 | ✅ |
| berunda | Sanyukta | 0.995 | ✅ |
| chakra | Sanyukta | 0.995 | ✅ |
| chandrakala | Asamyukta | 0.995 | ✅ |
| chaturam | Asamyukta | 0.995 | ✅ |
| garuda | Sanyukta | 0.995 | ✅ |
| katrimukha | Asamyukta | 0.995 | ✅ |
| kilaka | Sanyukta | 0.995 | ✅ |
| shivalinga | Sanyukta | 0.995 | ✅ |
| swastikam | Sanyukta | 0.995 | ✅ |
| mukulam | Asamyukta | 0.989 | ✅ |
| suchi | Asamyukta | 0.989 | ✅ |
| mushti | Asamyukta | 0.980 | ✅ |
| hamsapaksha | Asamyukta | 0.972 | ✅ |
| hamsasyam | Asamyukta | 0.955 | ✅ |
| pathaka | Asamyukta | 0.948 | ✅ |
| bramaram | Asamyukta | 0.940 | ✅ |
| tamarachudam | Asamyukta | 0.937 | ✅ |
| sikharam | Asamyukta | 0.933 | ✅ |
| katakamukha1 | Asamyukta | 0.892 | 🟡 |
| kapith | Asamyukta | 0.883 | 🟡 |
| tripathaka | Asamyukta | 0.746 | 🟡 |
| ardhapathaka | Asamyukta | 0.638 | 🟡 |

---

## Mudra Classes

### Asamyukta Hastas — Single Hand (18 mudras)

| Mudra | Sanskrit | Meaning |
|---|---|---|
| Pathaka | पताका | Flag — fingers extended, thumb bent. Denotes clouds, forest, sea, night |
| Tripathaka | त्रिपताका | Three-part flag — ring finger bent. Represents crown, tree, flame |
| Ardhapathaka | अर्धपताका | Half flag — last two fingers bent. Symbolises a knife or banner |
| Katrimukha | कर्तरीमुख | Scissors — index & little finger extended. Means separation |
| Mushti | मुष्टि | Fist — all fingers closed. Means holding or strength |
| Ardhachandran | अर्धचन्द्र | Half moon — thumb extended outward. Used to hold or grasp |
| Chandrakala | चन्द्रकला | Moon digit — little finger raised. Represents the moon or beauty |
| Suchi | सूची | Needle — index finger extended. Denotes oneness or the sun |
| Kapith | कपित्थ | Wood apple — thumb touches middle & index fingers. Offering to gods |
| Katakamukha | कटकामुख | Bracelet — curved fingers. Used to hold a garland or flower |
| Sikharam | शिखरम् | Peak — fist with thumb raised. Represents a bow or lover |
| Mukulam | मुकुलम् | Flower bud — all fingertips together. Represents a lotus bud |
| Hamsasyam | हंसास्य | Swan face — thumb, index & middle tips together |
| Hamsapaksha | हंसपक्ष | Swan wing — four fingers spread, thumb bent |
| Bramaram | भ्रमर | Bee — index finger on thumb, others spread |
| Chaturam | चतुरम् | Clever — ring finger bent on thumb. Represents a serpent or bee |
| Tamarachudam | ताम्रचूड | Rooster — thumb on middle finger |
| Alapadmam | अलपद्म | Full bloom lotus — all fingers spread wide |

### Sanyukta Hastas — Both Hands (7 mudras)

| Mudra | Sanskrit | Meaning |
|---|---|---|
| Garuda | गरुड | Eagle — thumbs interlocked, hands spread. Represents the eagle of Vishnu |
| Anjali | अञ्जलि | Salutation — both palms joined. Represents greeting and prayer |
| Berunda | बेरुण्ड | Two-headed bird — both hands spread as wings |
| Chakra | चक्र | Wheel — both index fingers and thumbs form a circle. Vishnu's discus |
| Shivalinga | शिवलिङ्ग | Symbol of Shiva — right fist placed on left palm |
| Swastikam | स्वस्तिकम् | Auspicious cross — wrists crossed at the chest |
| Kilaka | किलक | Bond — little fingers interlinked. Represents friendship |

---

## Project Structure

```
Vyakt/
  best.pt                ← trained YOLOv8s weights
  mudra_info.json    ← mudra descriptions with Asamyukta/Sanyukta labels
  detect.py              ← real-time webcam detection script
  README.md
  LICENSE
```

---

## Setup

### Requirements

- Python 3.8+
- Mac with M1/M2/M3 chip (recommended) or any system with a webcam
- Webcam (built-in or external)

### Install Dependencies

```bash
pip3 install ultralytics opencv-python torch torchvision
```

### Run Real-Time Detection

```bash
cd Vyakt/
python3 detect.py
```

Press **Q** to quit the webcam window.

---

## How It Works

```
Webcam frame
      ↓
YOLOv8s detects hand region + predicts mudra class
      ↓
Smoothing buffer — majority vote over last 10 frames
      ↓
Stable label looked up in mudra_info_new.json
      ↓
Overlay: Type badge + Mudra name + Confidence + Description
```

---

## What You See on Screen

```
┌──────────────────────────────── Device: MPS ─┐
│ ASAMYUKTA   ← green                          │
│                                              │
│         [hand with bounding box]             │
│                                              │
├──────────────────────────────────────────────┤
│ pathaka   94%                ← green         │
│ Pataka — Flag. Fingers extended, thumb       │
│ bent. Denotes clouds, forest, sea.           │
└──────────────────────────────────────────────┘
```

- **Green label** → Asamyukta (single hand)
- **Orange label** → Sanyukta (both hands)
- Smoothing buffer prevents flickering between frames

---

## Training Details

**Dataset:** [Dance Mudra by Mudras — Roboflow Universe](https://universe.roboflow.com/mudras/dance-mudra)  
**Training platform:** Google Colab (Tesla T4 GPU)  
**Base model:** YOLOv8s (small)

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data='data.yaml',
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,
    hsv_h=0.05, hsv_s=0.7, hsv_v=0.5,
    degrees=20, translate=0.1, scale=0.5,
    fliplr=0.5, mosaic=1.0,
    mixup=0.15, copy_paste=0.1, erasing=0.3,
)
```

### Dataset Split

| Split | Images |
|---|---|
| Train | 983 |
| Valid | 282 |
| Test | 282 |

### Augmentations Used

- HSV colour jitter (hue, saturation, brightness)
- Random rotation ±20°
- Translation and scaling
- Horizontal flip
- Mosaic augmentation
- Mixup
- Copy-paste
- Random erasing

---

## Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv8s — Ultralytics | Object detection and mudra classification |
| OpenCV | Webcam capture and frame annotation |
| PyTorch + MPS | Inference on Apple Silicon Neural Engine |
| Google Colab + T4 GPU | Model training |
| Roboflow Universe | Dataset source |

---

## Limitations

- Sanyukta mudras require both hands clearly visible in frame simultaneously
- Works best in good lighting with a plain or uncluttered background
- Ardhapathaka (0.638) and Tripathaka (0.746) have lower accuracy and may occasionally be misclassified
- Dataset covers mudras from Kathak and allied classical dance traditions

---

## Acknowledgements

- Classical mudra descriptions based on Natyashastra traditions
- Dataset: [Dance Mudra by Mudras on Roboflow Universe](https://universe.roboflow.com/mudras/dance-mudra)
- Detection framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## License

This project is for educational and research purposes.  
Classical mudra descriptions are based on traditional Kathak pedagogy.

---

*Vyakt — because every gesture has a meaning waiting to be expressed.*