import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_FILE = "src/models/qr_yolo_model/weights/best.pt"
TEST_IMAGES_DIR = Path("QR_Dataset/test_images")
ANNOTATED_DIR = Path("outputs/annotated_image")
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

JSON_DETECTION = "outputs/submission_detection_1.json"
JSON_DECODING = "outputs/submission_decoding_2.json"


detector = YOLO(MODEL_FILE)


def try_qr_decode(cropped):

    if cropped.size == 0:
        return ""

    h, w = cropped.shape[:2]
    if min(h, w) < 400:
        scale = 400 / min(h, w)
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    preprocess_variants = [
        gray,
        cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2),
        cv2.morphologyEx(cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                         cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)),
        cv2.filter2D(gray, -1, np.array([[-1, -1, -1],
                                         [-1, 9, -1],
                                         [-1, -1, -1]]))
    ]

    qr_engine = cv2.QRCodeDetector()
    for img_variant in preprocess_variants:
        try:
            decoded, pts, _ = qr_engine.detectAndDecode(img_variant)
            if decoded.strip():
                return decoded.strip()
        except Exception:
            continue
    return ""



def categorize_qr(content: str) -> str:

    if not content:
        return "undecoded"

    text = content.upper()

    if any(k in text for k in ["BATCH", "LOT", "BATCH NO", "LOT NO", "BATCH#", "LOT#"]):
        return "batch"
    if text.startswith(("B", "L")) and len(content) >= 5 and any(c.isdigit() for c in content):
        return "batch"

    if any(k in text for k in ["EXP", "EXPIRY", "EXPIRE", "EXPY", "VALID UNTIL"]):
        return "expiry"
    if re.search(r'(20[2-9][0-9])(0[1-9]|1[0-2])', text) or \
       re.search(r'(0[1-9]|1[0-2])/(20[2-9][0-9])', text) or \
       re.search(r'\d{2}-\d{2}-\d{4}', text):
        return "expiry"

    if any(k in text for k in ["MRP", "PRICE", "RS", "₹", "RUPEES", "COST"]) or \
       re.search(r'RS\s*\d+\.?\d*', text, re.IGNORECASE):
        return "price"

    if any(k in text for k in ["MFR", "MANUFACTURER", "MFG", "MADE BY", "SIG", "PRODUCED BY"]):
        return "manufacturer"

    if (8 <= len(content) <= 20 and content.isalnum()
            and any(c.isalpha() for c in content)
            and any(c.isdigit() for c in content)):
        return "serial"

    if (len(content) >= 6 and any(c.isupper() for c in content)
            and any(c.islower() for c in content)
            and any(c.isdigit() for c in content)):
        return "product_code"

    return "unknown"



det_records, dec_records = [], []
count_detected = count_decoded = 0

all_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
print(f"🔍 Found {len(all_images)} images for inference.")

for img_path in all_images:
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    outputs = detector.predict(str(img_path), save=False, verbose=False)
    boxes = outputs[0].boxes.xyxy.cpu().numpy()

    det_entry, dec_entry = [], []

    for (x1, y1, x2, y2) in boxes.astype(int):
        count_detected += 1
        box_coords = [int(x1), int(y1), int(x2), int(y2)]
        det_entry.append({"bbox": box_coords})
        pad = 30
        h, w = frame.shape[:2]
        cropped = frame[max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad)]

        qr_text = try_qr_decode(cropped)
        if qr_text:
            count_decoded += 1
        qr_type = categorize_qr(qr_text)
        dec_entry.append({"bbox": box_coords, "value": qr_text, "type": qr_type})
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{qr_text[:15]} ({qr_type})"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    det_records.append({"image_id": img_path.stem, "qrs": det_entry})
    dec_records.append({"image_id": img_path.stem, "qrs": dec_entry})

    cv2.imwrite(str(ANNOTATED_DIR / img_path.name), frame)

with open(JSON_DETECTION, "w") as f:
    json.dump(det_records, f, indent=2)

with open(JSON_DECODING, "w") as f:
    json.dump(dec_records, f, indent=2)

rate = (count_decoded / count_detected * 100) if count_detected else 0
print(f"\n✅ Done! Annotated images: {ANNOTATED_DIR}")
print(f"📑 Detection results → {JSON_DETECTION}")
print(f"📑 Decoding results → {JSON_DECODING}")
print(f"📊 Boxes: {count_detected}, Decoded: {count_decoded}, Success: {rate:.1f}%")
