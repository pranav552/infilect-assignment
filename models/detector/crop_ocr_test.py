import os
import cv2
import easyocr


CROP_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition"

reader = easyocr.Reader(["en"], gpu=False)


files = sorted([
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])[:20]


print("\n==============================================")
print("TARGETED PRODUCT-CROP OCR TEST")
print("==============================================")


for i, filename in enumerate(files, 1):

    path = os.path.join(CROP_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        continue


    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    upscaled = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)


    # --------------------------------------------------------
    # OCR — COLOR
    # --------------------------------------------------------

    color_results = reader.readtext(
        upscaled,
        detail=1
    )


    # --------------------------------------------------------
    # OCR — ENHANCED
    # --------------------------------------------------------

    enhanced_results = reader.readtext(
        enhanced,
        detail=1
    )


    print("\n" + "-" * 60)

    print(f"{i}. {filename}")


    print("\nCOLOR:")

    if color_results:

        for _, text, confidence in color_results:

            text = text.strip()

            if confidence >= 0.30:

                print(
                    f"  {text} | {confidence:.2f}"
                )

    else:

        print("  No text")


    print("\nENHANCED:")

    if enhanced_results:

        for _, text, confidence in enhanced_results:

            text = text.strip()

            if confidence >= 0.30:

                print(
                    f"  {text} | {confidence:.2f}"
                )

    else:

        print("  No text")


print("\n==============================================")
print("TEST COMPLETE")
print("==============================================")