"""Shared normalized-coordinate crop from the evaluation pipeline."""
import numpy as np
from PIL import Image

def crop_region(image, bbox_2d, polygon=None):
    w, h = image.size
    x1 = int(bbox_2d[0] / 1000 * w)
    y1 = int(bbox_2d[1] / 1000 * h)
    x2 = int(bbox_2d[2] / 1000 * w)
    y2 = int(bbox_2d[3] / 1000 * h)
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return image.crop((0, 0, 1, 1))

    # Polygon crop: fill outside polygon with gray, crop by bounding rect
    if polygon and len(polygon) > 4:
        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] = pts[:, 0] / 1000 * w
        pts[:, 1] = pts[:, 1] / 1000 * h
        px_min = max(0, int(np.floor(pts[:, 0].min())))
        py_min = max(0, int(np.floor(pts[:, 1].min())))
        px_max = min(w, int(np.ceil(pts[:, 0].max())))
        py_max = min(h, int(np.ceil(pts[:, 1].max())))
        if px_max > px_min and py_max > py_min:
            pts_shifted = pts - np.array([px_min, py_min], dtype=np.float32)
            crop_w, crop_h = px_max - px_min, py_max - py_min
            mask = Image.fromarray(np.zeros((crop_h, crop_w), dtype=np.uint8))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.polygon(pts_shifted.flatten().tolist(), fill=255)
            gray_bg = Image.new("RGB", (crop_w, crop_h), (128, 128, 128))
            region_crop = image.crop((px_min, py_min, px_max, py_max))
            gray_bg.paste(region_crop, mask=mask)
            return gray_bg

    return image.crop((x1, y1, x2, y2))


# ═══════════════════════════════════════════════════════════════════════════════
# VLM server + inference
# ═══════════════════════════════════════════════════════════════════════════════
