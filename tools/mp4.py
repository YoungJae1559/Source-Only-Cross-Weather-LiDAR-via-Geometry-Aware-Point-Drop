import re
from pathlib import Path
import cv2

IMG_DIR = Path("/home/vip/harry/LiDARWeather/TCSVT/semantickitti/ours_split_2")
OUT_MP4 = Path("/home/vip/harry/LiDARWeather/TCSVT/semantickitti/ours_split_2.mp4")
FPS = 10.0

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def main():
    print("script:", Path(__file__).resolve())
    print("IMG_DIR:", IMG_DIR, "exists:", IMG_DIR.exists(), "is_dir:", IMG_DIR.is_dir())

    files = sorted(list(IMG_DIR.glob("*.png")), key=lambda p: natural_key(p.name))
    print("png_count:", len(files))

    if len(files) == 0:
        raise ValueError(f"No PNG files found in {IMG_DIR}")

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first image: {files[0]}")
    h, w = first.shape[:2]

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(img)
        if i % 200 == 0:
            print("written:", i, "/", len(files))

    writer.release()
    print("saved:", OUT_MP4)

if __name__ == "__main__":
    main()
