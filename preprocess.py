import numpy as np
import cv2

def resample_path(points, n_points=128):
    """
    Resample a 2D path to a fixed number of points using arc-length parameterization.
    points: Nx2 numpy array (float32/float64)
    """
    if len(points) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    # compute cumulative arc length
    deltas = np.diff(pts, axis=0)
    seg_lens = np.sqrt((deltas**2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = arc[-1]
    if total_len < 1e-6:
        return np.tile(pts[0], (n_points, 1)).astype(np.float32)

    target = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    for i, t in enumerate(target):
        j = np.searchsorted(arc, t) - 1
        j = np.clip(j, 0, len(pts) - 2)
        t0, t1 = arc[j], arc[j+1]
        if t1 - t0 < 1e-6:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        resampled[i] = (1 - alpha) * pts[j] + alpha * pts[j+1]
    return resampled

def normalize_path(points, size=28, padding=2):
    """
    Normalize coordinates to fit inside a size x size box with padding.
    Returns normalized points in pixel coordinates.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    wh = np.maximum(max_xy - min_xy, 1e-6)
    scale = (size - 2*padding) / np.max(wh)
    pts_norm = (pts - min_xy) * scale + padding
    return pts_norm

def path_to_image(points, size=28, stroke=2):
    """
    Rasterize path points onto a grayscale image.
    points: Nx2 in pixel coords (already normalized)
    """
    img = np.zeros((size, size), dtype=np.uint8)
    if len(points) < 2:
        return img
    pts = points.astype(np.int32)
    for i in range(len(pts)-1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), 255, stroke, lineType=cv2.LINE_AA)
    # Dilate slightly to make strokes thicker/connected
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def preprocess_points_to_28x28(points, out_size=28, padding=2, n_points=128):
    """
    Full pipeline: resample -> normalize -> rasterize -> 28x28
    Returns image as float32 [0,1], shape (28,28,1)
    """
    if len(points) == 0:
        return np.zeros((out_size, out_size, 1), dtype=np.float32)
    rp = resample_path(points, n_points=n_points)
    npix = normalize_path(rp, size=out_size, padding=padding)
    img = path_to_image(npix, size=out_size, stroke=2)
    img = (img.astype(np.float32) / 255.0)[..., None]
    return img