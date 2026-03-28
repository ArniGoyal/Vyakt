# save as detect.py on local system
# pip3 install ultralytics opencv-python torch torchvision
from ultralytics import YOLO
import cv2
import json
import torch
from collections import deque, Counter

# ─── Device Setup ───────────────────────────────────────
if torch.backends.mps.is_available():
    device = 'mps'
    print("✓ Using Apple Neural Engine (MPS)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("✓ Using NVIDIA GPU")
else:
    device = 'cpu'
    print("⚠️ Using CPU")

# ─── Load Model ─────────────────────────────────────────
model = YOLO("best.pt")
model.to(device)

# ─── Load Descriptions ──────────────────────────────────
with open("mudra_info.json", "r") as f:
    MUDRA_INFO = json.load(f)

print(f"✓ {len(model.names)} mudras loaded")
print("✓ Press Q to quit")

# ─── Smoothing Buffer ────────────────────────────────────
buffer       = deque(maxlen=10)
stable_label = ""
description  = ""
confidence   = 0.0

# ─── Webcam ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    results   = model(frame, conf=0.35, verbose=False, device=device)
    annotated = results[0].plot()
    boxes     = results[0].boxes

    if boxes and len(boxes) > 0:
        best_idx   = boxes.conf.argmax().item()
        cls_id     = int(boxes.cls[best_idx].item())
        confidence = float(boxes.conf[best_idx].item())
        label      = model.names[cls_id]

        buffer.append(label)
        most_common, freq = Counter(buffer).most_common(1)[0]

        if freq >= 6:
            stable_label = most_common
            description  = MUDRA_INFO.get(
                stable_label.lower(),
                MUDRA_INFO.get(stable_label, "")
            )
    else:
        buffer.clear()
        stable_label = ""
        description  = ""
        confidence   = 0.0

    # ─── UI Overlay ─────────────────────────────────────
    h, w = annotated.shape[:2]

    # Black bar at bottom
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    if stable_label and description:
        # ── Determine type and color ──
        if "Sanyukta" in description:
            type_color = (255, 165, 0)   # orange for Sanyukta
            type_label = "SANYUKTA"
        else:
            type_color = (0, 255, 180)   # green for Asamyukta
            type_label = "ASAMYUKTA"

        # ── Type badge at top left ──
        cv2.putText(
            annotated, type_label,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, type_color, 2
        )

        # ── Mudra name + confidence ──
        cv2.putText(
            annotated,
            f"{stable_label}   {confidence*100:.0f}%",
            (15, h-72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, type_color, 2
        )

        # ── Description (strip the Sanyukta/Asamyukta prefix) ──
        clean_desc = description.split('|')[-1].strip()
        words      = clean_desc.split()
        line1      = " ".join(words[:10])
        line2      = " ".join(words[10:])

        cv2.putText(
            annotated, line1,
            (15, h-42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )
        if line2:
            cv2.putText(
                annotated, line2,
                (15, h-16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1
            )

    else:
        # ── No hand detected ──
        cv2.putText(
            annotated, "Show a mudra...",
            (15, h-50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (150, 150, 150), 1
        )

    # ── Device indicator top right ──
    cv2.putText(
        annotated,
        f"Device: {device.upper()}",
        (w-190, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (150, 150, 150), 1
    )

    cv2.imshow("Kathak Mudra Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()