"""Competition mask area in square pixels at the 640-pixel letterbox scale."""

import numpy as np
from pycocotools import mask as mask_utils


def polygon_area_at_640(poly_norm, width, height):
    points = np.asarray(poly_norm, dtype=np.float64) * [width, height]
    rle = mask_utils.frPyObjects([points.ravel().tolist()], height, width)[0]
    scale = 640.0 / max(width, height)
    return float(mask_utils.area(rle)) * (scale * scale)


def export_coordinates(points):
    """Measure the exact coordinates that the downstream reader will parse."""
    values = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return values, np.fromstring(values, dtype=np.float64, sep=" ").reshape(-1, 2)
