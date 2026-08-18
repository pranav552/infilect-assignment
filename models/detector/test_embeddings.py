import os
import torch
import numpy as np

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


# ============================================================
# PATH
# ============================================================

CROP_DIR = r"C:\AI\infilect_assignment\models\detector\runs\detect\product_recognition"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# LOAD RESNET50
# ============================================================

print("Loading ResNet50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

print("ResNet50 ready.")


# ============================================================
# PREPROCESSING
# ============================================================

transform = weights.transforms()


# ============================================================
# FIND CROPS
# ============================================================

files = sorted([
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])[:20]

print("Testing crops:", len(files))


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

embeddings = []

for i, filename in enumerate(files, 1):

    path = os.path.join(CROP_DIR, filename)

    image = Image.open(path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model(tensor)

    embedding = embedding.squeeze().numpy()

    # Normalize
    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-8
    )

    embeddings.append(embedding)

    print(
        f"{i:02d}. {filename} "
        f"→ embedding shape: {embedding.shape}"
    )


# ============================================================
# CONVERT TO ARRAY
# ============================================================

embeddings = np.array(embeddings)

print("\n======================================")
print("EMBEDDING TEST COMPLETE")
print("======================================")

print("Embedding matrix shape:", embeddings.shape)

print(
    "Expected:",
    f"({len(files)}, 2048)"
)