import numpy as np
from cvtools import (
    project_pinhole,
    normalize_points,
    denormalize_points,
    radial_distort_normalized,
    reproject_with_focals,
)

def test_project_pinhole_basic():
    pts3 = np.array([[1.0, 2.0, 4.0]])  # X=1, Y=2, Z=4
    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    uv = project_pinhole(pts3, fx=fx, fy=fy, cx=cx, cy=cy)
    # u = 800*(1/4)+320 = 520; v = 800*(2/4)+240 = 640
    assert np.allclose(uv, [[520.0, 640.0]], atol=1e-6)


def test_normalize_denormalize_roundtrip():
    uv = np.array([[520.0, 640.0], [100.0, 200.0]])
    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    xy = normalize_points(uv, fx=fx, fy=fy, cx=cx, cy=cy)
    uv2 = denormalize_points(xy, fx=fx, fy=fy, cx=cx, cy=cy)
    assert np.allclose(uv, uv2, atol=1e-6)


def test_radial_distortion_center_stable():
    # En el centro (0,0) no hay distorsión
    xy = np.array([[0.0, 0.0], [0.1, -0.2]])
    k1, k2 = 0.1, -0.05
    xy_d = radial_distort_normalized(xy, k1=k1, k2=k2)
    assert np.allclose(xy_d[0], [0.0, 0.0], atol=1e-9)
    # punto fuera del centro debe escalarse
    assert not np.allclose(xy_d[1], xy[1], atol=1e-9)


def test_reproject_with_focals_monotonic_u():
    P = np.array([[1.0, 0.0, 5.0]])
    focals = [200, 400, 800]
    outs = reproject_with_focals(P, focals=focals, cx=0.0, cy=0.0)
    us = [o[0, 0] for o in outs]  # u coord
    # u debe crecer con f (u = f*X/Z)
    assert us[0] < us[1] < us[2]
