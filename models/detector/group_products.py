import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\AI\infilect_assignment\models\detector"

CSV_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "product_recognition",
    "product_recognition_clean.csv"
)

CROPS_DIR = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "product_recognition"
)
OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "product_recognition",
    "grouped_products.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Visual similarity threshold
VISUAL_THRESHOLD = 0.82


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).upper().strip()

    if text in ["UNKNOWN", "NAN", "NONE", ""]:
        return ""

    # Remove OCR separators
    text = re.sub(r"[^A-Z0-9 ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PRODUCT GROUPING")
print("=" * 60)

print("\nLoading CSV...")

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))

# Normalize OCR text
df["normalized_text"] = df["clean_product_text"].apply(normalize_text)


# ============================================================
# LOAD RESNET50
# ============================================================

print("\nLoading ResNet50...")

weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(weights=weights)

# Remove final classification layer
model.fc = nn.Identity()

model.eval()

device = torch.device("cpu")

model = model.to(device)

transform = weights.transforms()

print("ResNet50 loaded.")
print("Device:", device)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating visual embeddings...")

embeddings = []

valid_indices = []

for i, row in df.iterrows():

    crop_name = str(row["crop"])

    crop_path = os.path.join(CROPS_DIR, crop_name)

    if not os.path.exists(crop_path):

        print("Missing crop:", crop_name)

        # Zero vector for missing crop
        embeddings.append(np.zeros(2048))

        continue

    try:

        image = Image.open(crop_path).convert("RGB")

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():

            embedding = model(image_tensor)

        embedding = embedding.squeeze().numpy()

        embeddings.append(embedding)

        valid_indices.append(i)

    except Exception as e:

        print("Error:", crop_name, "|", e)

        embeddings.append(np.zeros(2048))


embeddings = np.array(embeddings)

print("\nEmbedding matrix:", embeddings.shape)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

norms[norms == 0] = 1

embeddings = embeddings / norms


# ============================================================
# SIMILARITY MATRIX
# ============================================================

print("\nCalculating visual similarity...")

similarity_matrix = cosine_similarity(embeddings)

print("Similarity matrix:", similarity_matrix.shape)


# ============================================================
# GROUPING
# ============================================================

print("\nGrouping products...")

n = len(df)

uf = UnionFind(n)


# ------------------------------------------------------------
# RULE 1
#
# Exact same OCR text = same group
# ------------------------------------------------------------

text_groups = {}

for i, text in enumerate(df["normalized_text"]):

    if text == "":
        continue

    if text not in text_groups:
        text_groups[text] = []

    text_groups[text].append(i)


for text, indices in text_groups.items():

    if len(indices) < 2:
        continue

    first = indices[0]

    for idx in indices[1:]:

        uf.union(first, idx)


# ------------------------------------------------------------
# RULE 2
#
# Visual similarity fallback
#
# Only use visual similarity when OCR cannot provide
# useful information.
# ------------------------------------------------------------

for i in range(n):

    text_i = df.iloc[i]["normalized_text"]

    for j in range(i + 1, n):

        text_j = df.iloc[j]["normalized_text"]

        similarity = similarity_matrix[i, j]

        # If both products have useful OCR text and the
        # OCR text is different, do NOT merge them using
        # visual similarity.
        if text_i and text_j and text_i != text_j:
            continue

        # Use visual similarity when OCR is unavailable
        if similarity >= VISUAL_THRESHOLD:

            uf.union(i, j)


# ============================================================
# ASSIGN GROUP IDs
# ============================================================

print("\nAssigning group IDs...")

root_to_group = {}

group_ids = []

next_group_id = 1

for i in range(n):

    root = uf.find(i)

    if root not in root_to_group:

        root_to_group[root] = next_group_id

        next_group_id += 1

    group_ids.append(root_to_group[root])


df["group_id"] = group_ids


# ============================================================
# CLEAN OUTPUT
# ============================================================

output_columns = [
    "global_object_id",
    "image",
    "object_number",
    "product_text",
    "clean_product_text",
    "ocr_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop",
    "group_id"
]

df[output_columns].to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GROUPING COMPLETE")
print("=" * 60)

print("Total objects :", len(df))
print("Total groups  :", df["group_id"].nunique())
print("Output file   :", OUTPUT_CSV)

print("\nObjects per group:")
print(
    df["group_id"]
    .value_counts()
    .sort_index()
    .head(30)
)

print("\nFirst 20 results:")

print(
    df[
        [
            "global_object_id",
            "image",
            "clean_product_text",
            "group_id"
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)