import json
from pathlib import Path
from ultralytics import YOLO
import cv2


MODEL_PATH = "src/models/qr_yolo_model/weights/best.pt"

# pointing to the folder that contains images to test
IMAGES_FOLDER = "QR_Dataset/test_images"  # or any folder with images

# folder output for annotated image
OUTPUT_IMAGE_FOLDER = "outputs/annotated_image"
Path(OUTPUT_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

# Output of the JSON file
OUTPUT_JSON = "outputs/submission_detection_1.json"

# loading the pretrained model from the specified folder path of model
model = YOLO(MODEL_PATH)


image_paths = list(Path(IMAGES_FOLDER).glob("*.jpg")) + list(Path(IMAGES_FOLDER).glob("*.png"))

results_json = []

for img_path in image_paths:
    results = model.predict(str(img_path), save=False, verbose=False)
    image_result = results[0]
    img = cv2.imread(str(img_path))
    bboxes = []
    if image_result.boxes is not None:
        for box in image_result.boxes.xyxy.tolist(): 
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            bboxes.append({"bbox": [x_min, y_min, x_max, y_max]})
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    results_json.append({
        "image_id": img_path.stem,
        "qrs": bboxes
    })

    output_img_path = Path(OUTPUT_IMAGE_FOLDER) / img_path.name
    cv2.imwrite(str(output_img_path), img)

with open(OUTPUT_JSON, "w") as f:
    json.dump(results_json, f, indent=4)

print(f"Annotated images saved in '{OUTPUT_IMAGE_FOLDER}' and JSON saved as '{OUTPUT_JSON}'")
