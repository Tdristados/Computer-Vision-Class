import numpy as np
from cvtools import convolve2d, sobel_x, sobel_y, canny, laplacian

def _dummy_checker(h=32, w=32, sz=4):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, sz):
        for j in range(0, w, sz):
            if ((i // sz) + (j // sz)) % 2 == 0:
                img[i:i+sz, j:j+sz] = 255
    return img

def test_convolve2d_identity():
    img = _dummy_checker()
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    out = convolve2d(img, identity, padding="same")
    assert np.allclose(out, img)

def test_sobel_outputs():
    img = _dummy_checker()
    gx = sobel_x(img)
    gy = sobel_y(img)
    assert gx.shape == img.shape
    assert gy.shape == img.shape
    # deben tener bordes (no todo cero)
    assert np.abs(gx).sum() > 0
    assert np.abs(gy).sum() > 0

def test_canny_binary():
    img = _dummy_checker()
    edges = canny(img, 50, 150)
    # binaria uint8 (0/255)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})

def test_laplacian_has_pos_neg():
    img = _dummy_checker().astype(np.float32)
    lap = laplacian(img, ksize=3)
    # Debe haber positivos y/o negativos alrededor de bordes
    assert lap.shape == img.shape
    assert np.any(lap != 0)
