import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from cvtools import (
    # camera
    project_pinhole, normalize_points, denormalize_points,
    radial_distort_normalized, reproject_with_focals,
    # color
    rgb_to_hsv01, rgb_to_lab, color_histogram,
    quantize_uniform, reduce_image_size_by_color,
    # filters
    convolve2d, sobel_x, sobel_y, canny, laplacian
)

DATA_DIR = Path("data")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def list_images(folder: Path):
    if not folder.exists():
        return []
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    return sorted(imgs)

def demo_camera():
    print("=== Demo camera.py ===")
    xs, ys = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
    zs = np.full_like(xs, 3.0)
    P = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    uv_400 = project_pinhole(P, fx=400, fy=400, cx=320, cy=240)
    uv_800 = project_pinhole(P, fx=800, fy=800, cx=320, cy=240)

    xy_norm = normalize_points(uv_800, fx=800, fy=800, cx=320, cy=240)
    xy_dist = radial_distort_normalized(xy_norm, k1=0.1, k2=-0.05)
    uv_dist = denormalize_points(xy_dist, fx=800, fy=800, cx=320, cy=240)

    plt.figure()
    plt.scatter(uv_400[:,0], uv_400[:,1], label="f=400", s=20)
    plt.scatter(uv_800[:,0], uv_800[:,1], label="f=800", s=20)
    plt.scatter(uv_dist[:,0], uv_dist[:,1], label="f=800 + dist.", s=20)
    plt.gca().invert_yaxis(); plt.legend()
    plt.title("Proyección pinhole y distorsión radial")
    plt.tight_layout(); plt.show()

    outs = reproject_with_focals(P, focals=[200, 400, 800], cx=320, cy=240)
    plt.figure()
    for f, uv in zip([200,400,800], outs):
        plt.scatter(uv[:,0], uv[:,1], label=f"f={f}", s=20)
    plt.gca().invert_yaxis(); plt.legend()
    plt.title("Cambio de perspectiva con f"); plt.tight_layout(); plt.show()

def demo_color(img):
    print("=== Demo color.py ===")
    hsv = rgb_to_hsv01(img)
    lab = rgb_to_lab(img)
    print(f"HSV rango: [{hsv.min():.3f}, {hsv.max():.3f}]  LAB shape: {lab.shape}")

    color_histogram(img, bins=32, show=True)

    q16 = quantize_uniform(img, K=16)
    q64 = quantize_uniform(img, K=64)
    _, size_kb_png = reduce_image_size_by_color(img, K=64, output_path=None, ext=".png")
    print(f"Tamaño comprimido aprox (PNG) con K=64: {size_kb_png:.1f} KB")

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(q16); axes[1].set_title("Quant K=16"); axes[1].axis("off")
    axes[2].imshow(q64); axes[2].set_title("Quant K=64"); axes[2].axis("off")
    plt.tight_layout(); plt.show()

def demo_filters(img):
    print("=== Demo filters.py ===")
    kernel = np.ones((3,3), np.float32) / 9.0
    smoothed = convolve2d(img, kernel, padding="same")

    gx = sobel_x(img)
    gy = sobel_y(img)
    edges = canny(img, 80, 160)
    lap = laplacian(img, ksize=3)

    fig1, axes1 = plt.subplots(1,2, figsize=(10,4))
    axes1[0].imshow(img); axes1[0].set_title("Original"); axes1[0].axis("off")
    axes1[1].imshow(smoothed); axes1[1].set_title("Convolve 3x3"); axes1[1].axis("off")
    plt.tight_layout(); plt.show()

    fig2, axes2 = plt.subplots(1,3, figsize=(12,4))
    axes2[0].imshow(np.abs(gx), cmap="gray"); axes2[0].set_title("Sobel X"); axes2[0].axis("off")
    axes2[1].imshow(np.abs(gy), cmap="gray"); axes2[1].set_title("Sobel Y"); axes2[1].axis("off")
    axes2[2].imshow(edges, cmap="gray"); axes2[2].set_title("Canny"); axes2[2].axis("off")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow(lap, cmap="gray"); plt.title("Laplaciano")
    plt.axis("off"); plt.tight_layout(); plt.show()

def main():
    parser = argparse.ArgumentParser(description="Demo de cvtools")
    parser.add_argument("--img", type=str, help="Ruta a una imagen específica")
    parser.add_argument("--all", action="store_true", help="Procesar todas las imágenes en data/")
    args = parser.parse_args()

    if args.all:
        imgs = list_images(DATA_DIR)
        if not imgs:
            raise SystemExit(f"No se encontraron imágenes en {DATA_DIR}/")
        print(f"Procesando {len(imgs)} imágenes en {DATA_DIR}/")
        demo_camera()  # corre una vez (no depende de imagen)
        for p in imgs:
            print(f"\n--- {p.name} ---")
            img = load_rgb(p)
            demo_color(img)
            demo_filters(img)
        return

    if args.img:
        path = Path(args.img)
        if not path.exists():
            raise SystemExit(f"No existe la ruta: {path}")
        img = load_rgb(path)
        demo_camera()
        demo_color(img)
        demo_filters(img)
        return

    # Sin argumentos: intenta tomar la primera imagen disponible en data/
    imgs = list_images(DATA_DIR)
    if not imgs:
        raise SystemExit(
            f"No hay imágenes. Pon archivos en {DATA_DIR}/ (p. ej. .jpg, .png) "
            "o ejecuta con --img / --all."
        )
    print(f"Usando {imgs[0].name} (primera encontrada en {DATA_DIR}/)")
    demo_camera()
    img = load_rgb(imgs[0])
    demo_color(img)
    demo_filters(img)

if __name__ == "__main__":
    main()