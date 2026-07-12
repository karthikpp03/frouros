import cv2
import os
import random

# -------------------------------
# CONFIG
# -------------------------------
PERSON_NAME = "karthik"       # Change to Dad, Mom, etc.
CAMERA_INDEX = 1
OUTPUT_VIDEO = f"{PERSON_NAME}.mp4"

SAVE_FOLDER = os.path.join("faces", PERSON_NAME)
NUM_FRAMES = 50

os.makedirs(SAVE_FOLDER, exist_ok=True)

# -------------------------------
# RECORD VIDEO
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

print("=" * 50)
print("Recording...")
print("Move your face naturally:")
print("- Look left")
print("- Look right")
print("- Look up")
print("- Look down")
print("- Smile")
print("- Wear glasses if needed")
print("- Move closer/farther")
print()
print("Press 'q' to stop recording.")
print("=" * 50)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    writer.write(frame)

    cv2.imshow("Recording Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Recording completed.")

# -------------------------------
# RANDOM FRAME EXTRACTION
# -------------------------------

cap = cv2.VideoCapture(OUTPUT_VIDEO)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total Frames : {frame_count}")

if frame_count < NUM_FRAMES:
    selected = list(range(frame_count))
else:
    selected = sorted(random.sample(range(frame_count), NUM_FRAMES))

saved = 0
index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if index in selected:
        filename = os.path.join(
            SAVE_FOLDER,
            f"{saved+1:03d}.jpg"
        )
        cv2.imwrite(filename, frame)
        saved += 1

    index += 1

cap.release()

print("=" * 50)
print(f"Saved {saved} images")
print(f"Location : {SAVE_FOLDER}")
print("=" * 50)