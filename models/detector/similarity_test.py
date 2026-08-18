import os
import torch
import numpy as np

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


CROP_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition"


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

transform = weights.transforms()


# ============================================================
# LOAD 20 CROPS
# ============================================================

files = sorted([
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])[:20]


embeddings = []


for filename in files:

    path = os.path.join(CROP_DIR, filename)

    image = Image.open(path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    embedding = embedding.squeeze().numpy()

    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-8
    )

    embeddings.append(embedding)


embeddings = np.array(embeddings)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity = embeddings @ embeddings.T


# ============================================================
# PRINT MOST SIMILAR PAIRS
# ============================================================

pairs = []

for i in range(len(files)):

    for j in range(i + 1, len(files)):

        pairs.append(
            (
                similarity[i, j],
                files[i],
                files[j]
            )
        )


pairs.sort(reverse=True)


print("\n==============================================")
print("MOST VISUALLY SIMILAR PRODUCT PAIRS")
print("==============================================\n")


for score, file1, file2 in pairs[:15]:

    print(
        f"{score:.4f}  |  "
        f"{file1}  <-->  {file2}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("SIMILARITY TEST COMPLETE")
print("==============================================")

print("Highest similarity:", pairs[0][0])
print("Lowest similarity :", pairs[-1][0])