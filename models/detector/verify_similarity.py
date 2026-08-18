import os
import re
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


CROP_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition"

CSV_PATH = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition\product_recognition_clean.csv"


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[df["crop"].notna()].copy()

df["crop"] = df["crop"].astype(str)


# ============================================================
# FIRST 20 CROPS
# ============================================================

df = df.head(20).reset_index(drop=True)


# ============================================================
# LOAD RESNET50
# ============================================================

device = torch.device("cpu")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

transform = weights.transforms()


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = []

valid_rows = []

for index, row in df.iterrows():

    path = os.path.join(CROP_DIR, row["crop"])

    if not os.path.exists(path):
        continue

    image = Image.open(path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    embedding = embedding.squeeze().numpy()

    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-8
    )

    embeddings.append(embedding)

    valid_rows.append(row)


df = pd.DataFrame(valid_rows).reset_index(drop=True)

embeddings = np.array(embeddings)


# ============================================================
# SIMILARITY MATRIX
# ============================================================

similarity = embeddings @ embeddings.T


# ============================================================
# NORMALIZE OCR LABEL
# ============================================================

def normalize_label(text):

    if pd.isna(text):
        return "UNKNOWN"

    text = str(text).upper()

    if text == "UNKNOWN":
        return "UNKNOWN"

    # Keep only alphabetic words
    words = re.findall(r"[A-Z]{3,}", text)

    if not words:
        return "UNKNOWN"

    return " ".join(words)


df["normalized_label"] = df["clean_product_text"].apply(
    normalize_label
)


# ============================================================
# COMPARE PAIRS
# ============================================================

same_pairs = []
different_pairs = []

for i in range(len(df)):

    for j in range(i + 1, len(df)):

        label_i = df.loc[i, "normalized_label"]
        label_j = df.loc[j, "normalized_label"]

        score = float(similarity[i, j])

        if (
            label_i != "UNKNOWN"
            and label_j != "UNKNOWN"
            and label_i == label_j
        ):
            same_pairs.append(
                (
                    score,
                    df.loc[i, "crop"],
                    df.loc[j, "crop"],
                    label_i
                )
            )

        elif (
            label_i != "UNKNOWN"
            and label_j != "UNKNOWN"
            and label_i != label_j
        ):
            different_pairs.append(
                (
                    score,
                    df.loc[i, "crop"],
                    df.loc[j, "crop"],
                    label_i,
                    label_j
                )
            )


# ============================================================
# RESULTS
# ============================================================

same_pairs.sort(reverse=True)

different_pairs.sort(reverse=True)


print("\n==============================================")
print("SAME-LABEL PAIRS")
print("==============================================")

for score, crop1, crop2, label in same_pairs[:10]:

    print(
        f"{score:.4f} | "
        f"{label} | "
        f"{crop1} <--> {crop2}"
    )


print("\n==============================================")
print("DIFFERENT-LABEL PAIRS WITH HIGH SIMILARITY")
print("==============================================")

for score, crop1, crop2, label1, label2 in different_pairs[:10]:

    print(
        f"{score:.4f} | "
        f"{label1} vs {label2} | "
        f"{crop1} <--> {crop2}"
    )


print("\n==============================================")
print("SUMMARY")
print("==============================================")

print("Same-label pairs    :", len(same_pairs))
print("Different-label pairs:", len(different_pairs))