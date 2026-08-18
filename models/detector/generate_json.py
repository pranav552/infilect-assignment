import json
import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\AI\infilect_assignment\models\detector"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "product_recognition",
    "grouped_products.csv"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "product_recognition",
    "grouped_products.json"
)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 60)
print("GENERATING FINAL JSON")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print("Objects found:", len(df))


# ============================================================
# BUILD JSON
# ============================================================

objects = []

for _, row in df.iterrows():

    obj = {
        "object_id": int(row["global_object_id"]),
        "image": str(row["image"]),
        "x1": int(row["x1"]),
        "y1": int(row["y1"]),
        "x2": int(row["x2"]),
        "y2": int(row["y2"]),
        "group_id": int(row["group_id"])
    }

    objects.append(obj)


output = {
    "total_objects": len(objects),
    "total_groups": int(df["group_id"].nunique()),
    "objects": objects
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("JSON GENERATION COMPLETE")
print("=" * 60)

print(
    "Total objects:",
    output["total_objects"]
)

print(
    "Total groups:",
    output["total_groups"]
)

print(
    "JSON saved:",
    OUTPUT_JSON
)

print("\nFirst object:")

print(
    json.dumps(
        objects[0],
        indent=2
    )
)

print("\nFirst group IDs:")

print(
    [
        obj["group_id"]
        for obj in objects[:10]
    ]
)