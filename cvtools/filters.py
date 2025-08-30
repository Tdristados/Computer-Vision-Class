import numpy as np
import cv2

def _to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def convolve2d(image, kernel, padding="same", stride=1):
    """
    Convolución 2D genérica (conv = correlación con kernel volteado).
    Usa cv2.filter2D para eficiencia, invirtiendo el kernel manualmente.
    padding: "same" (por defecto) o "valid"
    stride: solo 1 soportado para esta implementación simple.
    """
    if stride != 1:
        raise NotImplementedError("Esta versión soporta stride=1.")
    img = np.asarray(image)
    ker = np.asarray(kernel, dtype=np.float32)
    ker_flip = np.flipud(np.fliplr(ker))
    border = cv2.BORDER_REPLICATE if padding == "same" else cv2.BORDER_ISOLATED
    out = cv2.filter2D(img, ddepth=-1, kernel=ker_flip, borderType=border)
    if padding == "valid":
        kh, kw = ker.shape[:2]
        pad_h = kh // 2
        pad_w = kw // 2
        out = out[pad_h: -pad_h or None, pad_w: -pad_w or None]
    return out


def sobel_x(image):
    """
    Gradiente aproximado en X usando el kernel Sobel clásico.
    """
    img = np.asarray(image)
    gray = _to_gray(img).astype(np.float32)
    gx = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    return gx


def sobel_y(image):
    """
    Gradiente aproximado en Y usando el kernel Sobel clásico.
    """
    img = np.asarray(image)
    gray = _to_gray(img).astype(np.float32)
    gy = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    return gy


def canny(image, low_threshold=100, high_threshold=200):
    """
    Detector de Canny estándar. Convierte a escala de grises internamente.
    Retorna imagen binaria (uint8 0/255).
    """
    gray = _to_gray(np.asarray(image)).astype(np.uint8)
    edges = cv2.Canny(gray, threshold1=int(low_threshold), threshold2=int(high_threshold))
    return edges


def laplacian(image, ksize=3):
    """
    Filtro Laplaciano (resalta cambios bruscos de intensidad / bordes de segunda derivada).
    Retorna imagen float32 con valores positivos/negativos (realzar y/o afilar).
    """
    gray = _to_gray(np.asarray(image)).astype(np.float32)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=int(ksize))
    return lap
