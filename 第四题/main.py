import os

import cv2

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "armor.mp4")
DST = os.path.join(BASE, "out", "result.mp4")
MODEL = os.path.join(BASE, "models", "armor.pt")

CONF = 0.25
NMS_IOU = 0.45
TRACK_IOU = 0.30
MAX_LOST = 8


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0.0 else 0.0


def detect(model, frame):
    out = []
    r = model.predict(frame, conf=CONF, iou=NMS_IOU, verbose=False)[0]
    if r.boxes is None:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    for b, s, c in zip(xyxy, conf, cls):
        out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(c)))
    return out


def pick(dets, target):
    if not dets:
        return None
    if target is None:
        return max(dets, key=lambda d: d[4])
    best, best_iou = None, -1.0
    for d in dets:
        v = iou(target, d[:4])
        if v > best_iou:
            best, best_iou = d, v
    if best_iou < TRACK_IOU:
        return max(dets, key=lambda d: d[4])
    return best


def main():
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        print("ERROR: cannot open", SRC)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    writer = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    target = None
    lost = 0
    trail = []

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detect(model, frame)
        hit = pick(dets, target)

        if hit is None:
            lost += 1
            if lost > MAX_LOST:
                target = None
        else:
            target = hit[:4]
            lost = 0
            cx = int((target[0] + target[2]) * 0.5)
            cy = int((target[1] + target[3]) * 0.5)
            trail.append((cx, cy))
            if len(trail) > 60:
                trail.pop(0)

        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d[:4]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "%.2f" % d[4], (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if target is not None:
            cx = int((target[0] + target[2]) * 0.5)
            cy = int((target[1] + target[3]) * 0.5)
            cv2.circle(vis, (cx, cy), 6, (255, 0, 0), 2)
            cv2.putText(vis, "LOCK (%d,%d)" % (cx, cy), (cx + 8, max(cy - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            for i in range(1, len(trail)):
                cv2.line(vis, trail[i - 1], trail[i], (0, 200, 255), 2)

        writer.write(vis)
        n += 1

    cap.release()
    writer.release()
    print("done, wrote %d frames -> %s" % (n, DST))


if __name__ == "__main__":
    main()
