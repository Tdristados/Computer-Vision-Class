import numpy as np

def project_pinhole(points_3d, fx, fy=None, cx=0.0, cy=0.0):
    """
    Proyección pinhole de puntos 3D a 2D en píxeles.
    points_3d: (N,3) con Z>0
    fx, fy: focales en píxeles (si fy=None, usa fy=fx)
    cx, cy: punto principal en píxeles
    Retorna: (N,2) con coordenadas de imagen [u, v]
    """
    P = np.asarray(points_3d, dtype=np.float64)
    if P.shape[-1] != 3:
        raise ValueError("points_3d debe tener forma (N,3).")
    fx = float(fx)
    fy = float(fy if fy is not None else fx)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    if np.any(Z <= 0):
        raise ValueError("Todos los puntos deben tener Z>0 para proyectar.")
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)


def normalize_points(points_2d, fx, fy=None, cx=0.0, cy=0.0):
    """
    Normaliza puntos en píxeles a coords. de cámara (x,y) con z=1:
    x = (u - cx)/fx, y = (v - cy)/fy
    """
    pts = np.asarray(points_2d, dtype=np.float64)
    fx = float(fx)
    fy = float(fy if fy is not None else fx)
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    return np.stack([x, y], axis=1)


def denormalize_points(xy_norm, fx, fy=None, cx=0.0, cy=0.0):
    """
    Desnormaliza (x,y) -> (u,v) en píxeles usando fx,fy,cx,cy.
    """
    pts = np.asarray(xy_norm, dtype=np.float64)
    fx = float(fx)
    fy = float(fy if fy is not None else fx)
    u = pts[:, 0] * fx + cx
    v = pts[:, 1] * fy + cy
    return np.stack([u, v], axis=1)


def radial_distort_normalized(xy_norm, k1=0.0, k2=0.0):
    """
    Aplica distorsión radial a coords. NORMALIZADAS (x,y) (modelo k1,k2):
    x_d = x * (1 + k1*r^2 + k2*r^4),  y_d = y * (1 + k1*r^2 + k2*r^4)
    Retorna (N,2) distorsionado en el espacio normalizado.
    """
    xy = np.asarray(xy_norm, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    scale = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    xd = x * scale
    yd = y * scale
    return np.stack([xd, yd], axis=1)


def reproject_with_focals(points_3d, focals, cx=0.0, cy=0.0, aspect=1.0):
    """
    Proyecta los mismos puntos 3D con una lista de focales para ver cambio de perspectiva.
    focals: iterable de valores f (fx), y fy = aspect * fx
    Retorna lista [P2D_f1, P2D_f2, ...], cada uno (N,2)
    """
    outs = []
    for f in focals:
        fx = float(f)
        fy = float(aspect) * fx
        outs.append(project_pinhole(points_3d, fx=fx, fy=fy, cx=cx, cy=cy))
    return outs
