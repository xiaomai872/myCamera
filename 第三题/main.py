import os

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "armor.mp4")
DST = os.path.join(BASE, "out", "result.mp4")


def main():
    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        print("ERROR: cannot open", SRC)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"fps={fps} size={w}x{h}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    writer = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    print(f"done, wrote {n} frames -> {DST}")


if __name__ == "__main__":
    main()
