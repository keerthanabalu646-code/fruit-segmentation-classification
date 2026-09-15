import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. IMAGE PATH
# ============================================================

IMAGE_PATH = "data/input/fruits.jpg"

OUTPUT_DIR = Path("data/output/yolo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "fruit_segmentation.jpg"


# ============================================================
# 2. LOAD YOLO SEGMENTATION MODEL
# ============================================================

print("\nLoading YOLO segmentation model...")

model = YOLO("yolo26n-seg.pt")

print("Model loaded successfully.")


# ============================================================
# 3. RUN SEGMENTATION
# ============================================================

print("\nRunning fruit segmentation...")

results = model.predict(
    source=IMAGE_PATH,

    # Detect low-confidence objects too
    conf=0.15,

    # Higher resolution
    imgsz=1024,

    # Allow overlapping objects
    iou=0.45,

    # Only fruit classes:
    # 46 = banana
    # 47 = apple
    # 49 = orange
    classes=[46, 47, 49],

    # High-quality masks
    retina_masks=True,

    verbose=True
)


result = results[0]


# ============================================================
# 4. CREATE SEGMENTED IMAGE
# ============================================================

segmented_image = result.plot(
    boxes=True,
    labels=True,
    conf=True,
    masks=True
)


# ============================================================
# 5. SAVE RESULT
# ============================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    segmented_image
)


# ============================================================
# 6. COUNT FRUITS
# ============================================================

banana_count = 0
apple_count = 0
orange_count = 0


if result.boxes is not None:

    for i in range(len(result.boxes)):

        class_id = int(
            result.boxes.cls[i]
        )

        confidence = float(
            result.boxes.conf[i]
        )

        fruit_name = result.names[class_id]


        if fruit_name == "banana":
            banana_count += 1

        elif fruit_name == "apple":
            apple_count += 1

        elif fruit_name == "orange":
            orange_count += 1


        print(
            f"{fruit_name:<10} "
            f"Confidence: {confidence:.2f}"
        )


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

total = (
    banana_count
    + apple_count
    + orange_count
)


print("\n")
print("=" * 45)
print("FRUIT SEGMENTATION RESULTS")
print("=" * 45)

print(f"Bananas : {banana_count}")
print(f"Apples  : {apple_count}")
print(f"Oranges : {orange_count}")

print("-" * 45)

print(f"Total fruits detected: {total}")

print("-" * 45)

print("Output:")
print(OUTPUT_PATH)

print("=" * 45)