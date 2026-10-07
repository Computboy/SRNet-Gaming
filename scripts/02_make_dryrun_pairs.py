from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
HR_DIR = ROOT / "data" / "dryrun" / "HR"
LR_DIR = ROOT / "data" / "dryrun" / "LR"
SCALE = 2

LR_DIR.mkdir(parents=True, exist_ok=True)

files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
    files.extend(HR_DIR.glob(ext))

if not files:
    raise SystemExit(
        f"No images found in {HR_DIR}\n"
        "Put a few ordinary images / game screenshots there first."
    )

count = 0
for src in sorted(files):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print("Skip unreadable:", src.name)
        continue

    h, w = img.shape[:2]
    h2 = (h // SCALE) * SCALE
    w2 = (w // SCALE) * SCALE
    if (h2, w2) != (h, w):
        img = img[:h2, :w2]

    lr = cv2.resize(
        img,
        (w2 // SCALE, h2 // SCALE),
        interpolation=cv2.INTER_AREA,
    )

    dst = LR_DIR / (src.stem + ".png")
    cv2.imwrite(str(dst), lr)
    count += 1
    print(f"{src.name}: HR {w2}x{h2} -> LR {w2//SCALE}x{h2//SCALE}")

print(f"\nDone. Generated {count} LR images in {LR_DIR}")
