import numpy as np
from cvtools import rgb_to_hsv01, rgb_to_lab, quantize_uniform, color_histogram

def _dummy_rgb(h=32, w=24):
    # gradiente simple RGB
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 2] = 128
    return img

def test_rgb_to_hsv01_range():
    img = _dummy_rgb()
    hsv = rgb_to_hsv01(img)
    assert hsv.dtype == np.float32
    assert hsv.min() >= 0.0 and hsv.max() <= 1.0

def test_rgb_to_lab_shape():
    img = _dummy_rgb()
    lab = rgb_to_lab(img)
    assert lab.shape == img.shape
    assert lab.dtype == np.float32

def test_quantize_uniform_colors():
    img = _dummy_rgb(64, 64)
    K = 16
    q = quantize_uniform(img, K=K)
    # contar colores únicos (aprox <= L^3 con L=round(K^(1/3)))
    uniq = np.unique(q.reshape(-1, 3), axis=0)
    assert uniq.shape[0] <= 64  # margen (p.ej., L^3=64 si L=4)

def test_color_histogram_bins():
    img = _dummy_rgb()
    r, g, b, edges = color_histogram(img, bins=16, show=False)
    assert len(r) == len(g) == len(b) == 16
    assert edges is not None
