import os
import cv2
import easyocr

IMAGE_DIR = r"C:\AI\infilect_assignment\runs\detect\outputs\detector_validation"

reader = easyocr.Reader(["en"], gpu=True)

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("\n========== ORIGINAL IMAGE OCR ==========\n")

for i, filename in enumerate(files, 1):

    path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    # Resize large images down while maintaining enough detail
    h, w = img.shape[:2]

    max_dim = 1600

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    results = reader.readtext(img, detail=1)

    print(f"\n{i}. {filename}")

    if not results:
        print("  No text detected")
        continue

    for _, text, confidence in results:

        text = text.strip()

        if confidence >= 0.30 and len(text) >= 2:
            print(f"  {text} | {confidence:.2f}")

print("\n========== OCR COMPLETE ==========\n")