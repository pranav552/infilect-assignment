import pandas as pd
import re

INPUT = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition\product_recognition.csv"

OUTPUT = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition\product_recognition_clean.csv"


df = pd.read_csv(INPUT)


def clean_text(text):

    if pd.isna(text):
        return "UNKNOWN"

    text = str(text).strip()

    # Remove common YOLO/OCR garbage
    text = re.sub(r'\bobject\b', ' ', text, flags=re.IGNORECASE)

    # Remove decimal confidence-like values
    text = re.sub(r'\b\d+\.\d+\b', ' ', text)

    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^A-Za-z\s|]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return "UNKNOWN"

    # Split OCR fragments
    parts = [p.strip() for p in text.split("|")]

    useful = []

    for part in parts:

        part = part.strip()

        # Ignore very short fragments
        if len(part) < 3:
            continue

        # Ignore fragments containing no letters
        if not re.search(r'[A-Za-z]', part):
            continue

        useful.append(part)

    if not useful:
        return "UNKNOWN"

    # Keep the useful OCR evidence
    result = " | ".join(useful[:3])

    return result.upper()


df["clean_product_text"] = df["product_text"].apply(clean_text)


# ------------------------------------------------------------
# Add a simple reliability flag
# ------------------------------------------------------------

def reliability(row):

    text = row["clean_product_text"]
    confidence = row["ocr_confidence"]

    if text == "UNKNOWN":
        return "UNKNOWN"

    if confidence >= 0.85 and len(text) >= 4:
        return "HIGH"

    if confidence >= 0.60 and len(text) >= 4:
        return "MEDIUM"

    return "LOW"


df["recognition_reliability"] = df.apply(
    reliability,
    axis=1
)


df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)


print("\n==============================================")
print("CLEANING COMPLETE")
print("==============================================")

print(f"Input rows  : {len(df)}")
print(f"Output file : {OUTPUT}")

print("\nReliability:")
print(df["recognition_reliability"].value_counts())

print("\nFirst 20 cleaned results:")
print(
    df[
        [
            "global_object_id",
            "image",
            "object_number",
            "product_text",
            "clean_product_text",
            "ocr_confidence",
            "recognition_reliability"
        ]
    ].head(20).to_string(index=False)
)