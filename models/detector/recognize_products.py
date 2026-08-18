import os
import csv
import cv2
import easyocr
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\AI\infilect_assignment\models\detector\candidate_v2\candidate_v2.pt"

IMAGE_DIR = r"C:\AI\infilect_assignment\runs\detect\outputs\detector_validation"

OUTPUT_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "product_recognition.csv")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading Candidate 2...")

model = YOLO(MODEL_PATH)

print("Candidate 2 loaded.")

print("\nLoading EasyOCR...")

reader = easyocr.Reader(["en"], gpu=True)

print("EasyOCR loaded.")


# ============================================================
# IMAGE LIST
# ============================================================

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nFound {len(files)} original images.")


# ============================================================
# CSV
# ============================================================

rows = []

global_object_id = 0


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_number, filename in enumerate(files, 1):

    print("\n" + "=" * 70)
    print(f"{image_number}. {filename}")
    print("=" * 70)

    image_path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        continue

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    result = model.predict(
        source=image,
        conf=0.50,
        iou=0.50,
        device="cpu",
        verbose=False
    )[0]

    boxes = result.boxes

    print(f"YOLO detections: {len(boxes)}")


    # --------------------------------------------------------
    # OCR ON ORIGINAL IMAGE
    # --------------------------------------------------------

    print("Running OCR...")

    ocr_results = reader.readtext(image, detail=1)

    print(f"OCR text regions: {len(ocr_results)}")


    # --------------------------------------------------------
    # CONVERT OCR RESULTS
    # --------------------------------------------------------

    ocr_items = []

    for detection in ocr_results:

        polygon, text, confidence = detection

        text = text.strip()

        if len(text) < 2:
            continue

        if confidence < 0.30:
            continue

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        ocr_items.append({
            "text": text,
            "confidence": float(confidence),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": center_x,
            "cy": center_y
        })


    # --------------------------------------------------------
    # PROCESS EACH YOLO OBJECT
    # --------------------------------------------------------

    for object_number, box in enumerate(boxes, 1):

        global_object_id += 1

        coords = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = coords

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # FIND OCR TEXT INSIDE THIS OBJECT
        # ----------------------------------------------------

        matched_text = []

        for item in ocr_items:

            cx = item["cx"]
            cy = item["cy"]

            # Text center inside YOLO box
            if x1 <= cx <= x2 and y1 <= cy <= y2:

                matched_text.append(item)


        # ----------------------------------------------------
        # SORT OCR BY CONFIDENCE
        # ----------------------------------------------------

        matched_text.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )


        # ----------------------------------------------------
        # BUILD LABEL
        # ----------------------------------------------------

        if matched_text:

            # Remove duplicate text
            unique_text = []

            for item in matched_text:

                text = item["text"]

                if text.lower() not in [
                    t.lower() for t in unique_text
                ]:
                    unique_text.append(text)

            product_text = " | ".join(unique_text[:5])

            best_confidence = matched_text[0]["confidence"]

        else:

            product_text = "UNKNOWN"
            best_confidence = 0.0


        # ----------------------------------------------------
        # SAVE CROP
        # ----------------------------------------------------

        crop_filename = (
            f"{os.path.splitext(filename)[0]}"
            f"-{object_number}.jpg"
        )

        crop_path = os.path.join(
            OUTPUT_DIR,
            crop_filename
        )

        cv2.imwrite(crop_path, crop)


        # ----------------------------------------------------
        # SAVE CSV ROW
        # ----------------------------------------------------

        rows.append({
            "global_object_id": global_object_id,
            "image": filename,
            "object_number": object_number,
            "product_text": product_text,
            "ocr_confidence": round(best_confidence, 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "crop": crop_filename
        })


        print(
            f"Object {object_number:03d} → "
            f"{product_text} "
            f"(OCR {best_confidence:.2f})"
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "global_object_id",
            "image",
            "object_number",
            "product_text",
            "ocr_confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "crop"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT RECOGNITION COMPLETE")
print("=" * 70)

print(f"Total objects processed : {len(rows)}")
print(f"CSV saved               : {CSV_PATH}")
print(f"Crops saved             : {OUTPUT_DIR}")

print("\nDone.")