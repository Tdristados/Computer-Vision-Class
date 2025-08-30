import numpy as np
import cv2
import matplotlib.pyplot as plt

def _ensure_rgb(img):
    """Convierte BGR->RGB si parece provenir de cv2.imread (heurística simple)."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        # No hay forma perfecta de saber; asumimos que el usuario trae BGR de cv2.
        # Si ya es RGB, la conversión BGR2RGB hará poco daño (swap).
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb_to_hsv01(img_rgb):
    """
    Convierte RGB (uint8 [0..255]) a HSV normalizado en [0,1] por canal.
    Retorna float32 con H,S,V en [0,1].
    """
    rgb = _ensure_rgb(img_rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 179.0   # OpenCV: H en [0,179]
    s = hsv[:, :, 1].astype(np.float32) / 255.0   # S en [0,255]
    v = hsv[:, :, 2].astype(np.float32) / 255.0   # V en [0,255]
    return np.stack([h, s, v], axis=2)


def rgb_to_lab(img_rgb):
    """
    Convierte RGB [0..255] a CIE Lab estándar (float32).
    Con entrada normalizada, OpenCV produce L en [0,100], a,b aprox. [-128..127].
    """
    rgb = _ensure_rgb(img_rgb).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)


def color_histogram(img_rgb, bins=32, show=False):
    """
    Calcula histograma por canal en RGB.
    Retorna (histR, histG, histB, bin_edges),
    donde cada hist es array de tamaño `bins`.
    Si show=True, dibuja los histogramas con matplotlib.
    """
    rgb = _ensure_rgb(img_rgb)
    channels = cv2.split(rgb)
    hist_list = []
    edges = None
    for ch in channels:
        hist, edges = np.histogram(ch.ravel(), bins=bins, range=(0, 256))
        hist_list.append(hist.astype(np.int64))

    if show:
        plt.figure()
        plt.title("Histogramas por canal (RGB)")
        plt.plot(hist_list[0], label="R")
        plt.plot(hist_list[1], label="G")
        plt.plot(hist_list[2], label="B")
        plt.legend()
        plt.xlabel("Bins")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    return hist_list[0], hist_list[1], hist_list[2], edges


def quantize_uniform(img_rgb, K=64):
    """
    Cuantización uniforme simple a ~K colores.
    Aproximación: L = round(K^(1/3)) niveles por canal -> L^3 colores como máximo.
    """
    if K < 2:
        raise ValueError("K debe ser >= 2.")
    rgb = _ensure_rgb(img_rgb).astype(np.float32)
    L = int(np.clip(round(K ** (1.0 / 3.0)), 2, 256))
    step = 255.0 / max(L - 1, 1)
    q = np.round(rgb / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


def reduce_image_size_by_color(img_rgb, K=64, output_path=None, ext=".png"):
    """
    Reduce el número de colores y reporta el tamaño comprimido en KB (estimado con PNG/JPEG).
    - img_rgb: imagen en RGB/BGR (uint8)
    - K: número aprox. de colores
    - output_path: si se da, guarda la imagen.
    - ext: '.png' (sin pérdidas) o '.jpg' (con pérdidas).
    Retorna: (img_quant, size_kb)
    """
    img_q = quantize_uniform(img_rgb, K=K)
    # Usamos PNG por defecto para tamaño determinista sin pérdidas
    if ext.lower() == ".jpg" or ext.lower() == ".jpeg":
        ok, buff = cv2.imencode(".jpg", cv2.cvtColor(img_q, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        ok, buff = cv2.imencode(".png", cv2.cvtColor(img_q, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    size_kb = len(buff.tobytes()) / 1024.0
    if output_path:
        with open(output_path, "wb") as f:
            f.write(buff)
    return img_q, size_kb
