"""
Autor: Andrés Manrique
cvtools: Biblioteca de los códigos realizados en Introducción a visión por computadora 2025-2.
"""

from .camera import (
    project_pinhole,
    radial_distort_normalized,
    normalize_points,
    denormalize_points,
    reproject_with_focals,
)

from .color import (
    rgb_to_hsv01,
    rgb_to_lab,
    color_histogram,
    quantize_uniform,
    reduce_image_size_by_color,
)

from .filters import (
    convolve2d,
    sobel_x,
    sobel_y,
    canny,
    laplacian,
)

__all__ = [
    # camera
    "project_pinhole",
    "radial_distort_normalized",
    "normalize_points",
    "denormalize_points",
    "reproject_with_focals",
    # color
    "rgb_to_hsv01",
    "rgb_to_lab",
    "color_histogram",
    "quantize_uniform",
    "reduce_image_size_by_color",
    # filters
    "convolve2d",
    "sobel_x",
    "sobel_y",
    "canny",
    "laplacian",
]
