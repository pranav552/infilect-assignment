import os
import cv2
import easyocr

CROP_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_crops\crops\object"

reader = easyocr.Reader(["en"], gpu=True)

files = [
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:10]

print("\n========== IMPROVED OCR TEST ==========\n")

for i, filename in enumerate(files, 1):

    path = os.path.join(CROP_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    # Upscale
    img = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # OCR on original/upscaled image
    result_color = reader.readtext(img, detail=1)

    # OCR on enhanced grayscale image
    result_gray = reader.readtext(enhanced, detail=1)

    print(f"\n{i}. {filename}")

    print("  --- Color OCR ---")

    if result_color:
        for _, text, confidence in result_color:
            if confidence >= 0.20:
                print(f"  {text} | {confidence:.2f}")
    else:
        print("  No text")

    print("  --- Enhanced OCR ---")

    if result_gray:
        for _, text, confidence in result_gray:
            if confidence >= 0.20:
                print(f"  {text} | {confidence:.2f}")
    else:
        print("  No text")

print("\n========== TEST COMPLETE ==========\n")