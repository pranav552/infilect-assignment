from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO

import os
import re
import uuid
from difflib import SequenceMatcher
import numpy as np
import torch
import torch.nn as nn
import easyocr

from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models", "detector", "candidate_v2", "candidate_v2.pt"
)

UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DETECTION_CONFIDENCE = 0.50

VISUAL_THRESHOLD = 0.78


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading Candidate 2...")

detector = YOLO(MODEL_PATH)

print("Candidate 2 loaded successfully.")


# ============================================================
# LOAD EASYOCR
# ============================================================

print("\nLoading EasyOCR...")

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully.")


# ============================================================
# LOAD RESNET50
# ============================================================

print("\nLoading ResNet50...")

weights = ResNet50_Weights.IMAGENET1K_V2

embedding_model = resnet50(
    weights=weights
)

# Remove classification layer
embedding_model.fc = nn.Identity()

embedding_model.eval()

device = torch.device("cpu")

embedding_model = embedding_model.to(device)

embedding_transform = weights.transforms()

print("ResNet50 loaded successfully.")
print("Device:", device)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_ocr_text(text):

    if not text:
        return ""

    text = str(text).strip()

    # Remove common OCR/detection garbage
    text = re.sub(
        r"\bobject\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove decimal values
    text = re.sub(
        r"\b\d+\.\d+\b",
        " ",
        text
    )

    # Remove standalone numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^A-Za-z\s|]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return ""

    parts = [
        part.strip()
        for part in text.split("|")
    ]

    useful = []

    for part in parts:

        if len(part) < 3:
            continue

        if not re.search(
            r"[A-Za-z]",
            part
        ):
            continue

        useful.append(part)

    if not useful:
        return ""

    return " | ".join(
        useful[:3]
    ).upper()

# ============================================================
# BRAND EXTRACTION
# ============================================================

def extract_brand_identifier(ocr_text):
    """
    Extract a lightweight brand identifier from cleaned OCR text.

    The method is intentionally generic:
    - No fixed list of brands is used.
    - OCR text is split into meaningful tokens.
    - Very short/common product-description words are ignored.
    - The first strong token is treated as the brand identifier.

    Example:

        "BOURNVITA CHOCOLATE" -> "BOURNVITA"
        "BOURNVITA 500G"      -> "BOURNVITA"
        "COMPLAN PISTA"       -> "COMPLAN"
        "CADBURY DAIRY MILK"  -> "CADBURY"
    """

    if not ocr_text:
        return ""

    text = str(ocr_text).upper().strip()

    # Extract alphabetic tokens
    tokens = re.findall(
        r"[A-Z]+",
        text
    )

    # Generic words that are more likely to describe
    # the product rather than identify its brand.
    ignored_words = {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "FROM",
        "NEW",
        "PACK",
        "POWDER",
        "MILK",
        "FOOD",
        "DRINK",
        "CHOCOLATE",
        "VANILLA",
        "STRAWBERRY",
        "ORIGINAL",
        "PREMIUM",
        "INSTANT",
        "FLAVOUR",
        "FLAVOR",
        "PLUS",
        "LITE",
        "ZERO",
        "FREE",
        "MAX",
        "GOLD",
        "CLASSIC",
        "CREAM",
        "BISCUIT",
        "BISCUITS",
        "SOAP",
        "SHAMPOO",
        "CONDITIONER",
        "DETERGENT",
        "LIQUID",
        "BAR",
        "BARS",
        "SIZE",
        "VALUE"
    }

    meaningful_tokens = []

    for token in tokens:

        token = token.strip()

        if len(token) < 3:
            continue

        if token in ignored_words:
            continue

        meaningful_tokens.append(token)

    if not meaningful_tokens:
        return ""

    # The first strong token is used as the brand identifier.
    return meaningful_tokens[0]


def brands_are_similar(
    brand_a,
    brand_b,
    threshold=0.88
):
    """
    Compare two OCR-derived brand identifiers.

    Exact matches are always accepted.

    Fuzzy matching handles small OCR errors such as:

        BOURNVITA
        BOURNVTA

    The relatively high threshold prevents unrelated brands
    from being merged too easily.
    """

    if not brand_a or not brand_b:
        return False

    brand_a = brand_a.upper().strip()
    brand_b = brand_b.upper().strip()

    # Exact brand match
    if brand_a == brand_b:
        return True

    # Avoid comparing extremely different-length tokens.
    length_difference = abs(
        len(brand_a) - len(brand_b)
    )

    if length_difference > 2:
        return False

    similarity = SequenceMatcher(
        None,
        brand_a,
        brand_b
    ).ratio()

    return similarity >= threshold

# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_DIR,
        filename
    )
# ============================================================
# DETECT + GROUP
# ============================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image provided"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "Empty filename"
        }), 400


    # --------------------------------------------------------
    # SAVE UPLOAD
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        return jsonify({
            "error": "Unsupported image format"
        }), 400


    unique_name = (
        str(uuid.uuid4())
        + extension
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    file.save(image_path)


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_np = np.array(image)

    height, width = image_np.shape[:2]


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    print(
        f"\nProcessing image: "
        f"{file.filename}"
    )

    results = detector.predict(
        source=image_np,
        conf=DETECTION_CONFIDENCE,
        device="cpu",
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    print(
        "Detected objects:",
        len(boxes)
    )


    # --------------------------------------------------------
    # NO DETECTIONS
    # --------------------------------------------------------

    if len(boxes) == 0:

        return jsonify({
            "total_objects": 0,
            "total_groups": 0,
            "objects": []
        })


    # ========================================================
    # PROCESS EACH DETECTED OBJECT
    # ========================================================

    object_data = []

    embeddings = []


    for object_index, box in enumerate(
        boxes
    ):

        coordinates = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = coordinates

        x1 = max(
            0,
            int(x1)
        )

        y1 = max(
            0,
            int(y1)
        )

        x2 = min(
            width,
            int(x2)
        )

        y2 = min(
            height,
            int(y2)
        )


        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop = image_np[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # OCR ON PRODUCT CROP
        # ----------------------------------------------------

        ocr_text = ""

        if crop.size > 0:

            ocr_results = (
                ocr_reader.readtext(
                    crop,
                    detail=1
                )
            )

            texts = []

            for detection in ocr_results:

                _, text, confidence = (
                    detection
                )

                if confidence >= 0.30:

                    cleaned = (
                        clean_ocr_text(
                            text
                        )
                    )

                    if cleaned:

                        texts.append(
                            cleaned
                        )


            # Remove duplicates
            unique_texts = []

            for text in texts:

                if text not in unique_texts:

                    unique_texts.append(
                        text
                    )

            ocr_text = " | ".join(
                unique_texts[:3]
            )


        # ----------------------------------------------------
        # RESNET50 EMBEDDING
        # ----------------------------------------------------

        if crop.size > 0:

            crop_image = Image.fromarray(
                crop
            ).convert("RGB")

            tensor = (
                embedding_transform(
                    crop_image
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():

                embedding = (
                    embedding_model(
                        tensor
                    )
                )

            embedding = (
                embedding
                .squeeze()
                .numpy()
            )

            norm = np.linalg.norm(
                embedding
            )

            if norm > 0:

                embedding = (
                    embedding / norm
                )

        else:

            embedding = np.zeros(
                2048
            )


        embeddings.append(
            embedding
        )


        object_data.append({
            "object_id": object_index + 1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "ocr_text": ocr_text
        })


    # ========================================================
    # GROUPING
    # ========================================================

    n = len(object_data)

    uf = UnionFind(n)


    # ========================================================
    # EXTRACT BRAND IDENTIFIER
    # ========================================================

    for obj in object_data:

        obj["brand_identifier"] = (
            extract_brand_identifier(
                obj["ocr_text"]
            )
        )


    # ========================================================
    # RULE 1
    #
    # BRAND-LEVEL GROUPING
    # ========================================================
    #
    # Products are grouped using their extracted brand
    # identifier instead of their complete OCR text.
    #
    # Example:
    #
    #   BOURNVITA CHOCOLATE
    #   BOURNVITA 500G
    #   BOURNVITA HEALTH DRINK
    #
    # all produce:
    #
    #   BOURNVITA
    #
    # and therefore belong to the same brand group.
    # ========================================================

    for i in range(n):

        brand_i = object_data[i][
            "brand_identifier"
        ]

        # No usable OCR for this object
        if not brand_i:
            continue

        for j in range(
            i + 1,
            n
        ):

            brand_j = object_data[j][
                "brand_identifier"
            ]

            # No usable OCR for second object
            if not brand_j:
                continue

            # Same / similar brand
            if brands_are_similar(
                brand_i,
                brand_j
            ):

                uf.union(
                    i,
                    j
                )


    # ========================================================
    # RULE 2
    #
    # VISUAL SIMILARITY FALLBACK
    # ========================================================
    #
    # ResNet50 is used ONLY when BOTH objects have no usable
    # OCR-derived brand identifier.
    #
    # If either object has a known brand, visual similarity
    # cannot override the brand information.
    # ========================================================

    embeddings = np.array(
        embeddings
    )


    similarity_matrix = (
        embeddings @ embeddings.T
    )


    for i in range(n):

        brand_i = object_data[i][
            "brand_identifier"
        ]

        for j in range(
            i + 1,
            n
        ):

            brand_j = object_data[j][
                "brand_identifier"
            ]

            # ------------------------------------------------
            # If either object has usable OCR/brand information,
            # DO NOT use visual similarity to merge them.
            # ------------------------------------------------

            if brand_i or brand_j:
                continue

            similarity = (
                similarity_matrix[
                    i,
                    j
                ]
            )

            # Visual fallback
            if similarity >= VISUAL_THRESHOLD:

                uf.union(
                    i,
                    j
                )


    # ========================================================
    # ASSIGN GROUP IDs
    # ========================================================

    root_to_group = {}

    next_group_id = 1


    for i in range(n):

        root = uf.find(i)

        if root not in root_to_group:

            root_to_group[root] = (
                next_group_id
            )

            next_group_id += 1

        object_data[i][
            "group_id"
        ] = root_to_group[root]
    # ========================================================
    # REMOVE OCR FROM FINAL REQUIRED
    # RESPONSE
    # ========================================================

    final_objects = []


    for obj in object_data:

        final_objects.append({

            "object_id":
                obj["object_id"],

            "x1":
                obj["x1"],

            "y1":
                obj["y1"],

            "x2":
                obj["x2"],

            "y2":
                obj["y2"],

            "group_id":
                obj["group_id"]

        })

    # ========================================================
    # CREATE ANNOTATED IMAGE
    # ========================================================

    annotated_image = image.copy()

    draw = ImageDraw.Draw(
        annotated_image
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24
        )
    except:
        font = ImageFont.load_default()


    for obj in final_objects:

        x1 = obj["x1"]
        y1 = obj["y1"]
        x2 = obj["x2"]
        y2 = obj["y2"]

        group_id = obj["group_id"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        label = f"Group {group_id}"

        try:
            bbox = draw.textbbox(
                (x1, y1),
                label,
                font=font
            )

            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        except:

            text_width = 80
            text_height = 20

        draw.rectangle(
            [
                x1,
                max(0, y1 - text_height - 8),
                x1 + text_width + 10,
                y1
            ],
            fill="red"
        )

        draw.text(
            (
                x1 + 5,
                max(0, y1 - text_height - 5)
            ),
            label,
            fill="white",
            font=font
        )


    annotated_filename = (
        "result_"
        + os.path.basename(image_path)
    )

    annotated_path = os.path.join(
        UPLOAD_DIR,
        annotated_filename
    )

    annotated_image.save(
        annotated_path
    )


    # ========================================================
    # FINAL JSON
    # ========================================================

    total_groups = len(
        set(
            obj["group_id"]
            for obj in final_objects
        )
    )

    response = {
        "total_objects": len(final_objects),
        "total_groups": total_groups,
        "annotated_image": f"/uploads/{annotated_filename}",
        "objects": final_objects
    }

    print(
        "Groups:",
        total_groups
    )

    return jsonify(
        response
    )

# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
