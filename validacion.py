
#-------------------------------------
# VALIDACIÓN DE MAXWELL
#-------------------------------------

from dataclasses import dataclass
import numpy as np

from modos import ParametrosModo
from campos import calcular_campos


@dataclass(frozen=True)
class ResultadoValidacion:
    error_faraday: float
    error_ampere: float
    error_gauss_e: float
    error_gauss_h: float
    error_fronteras: float


#-------------------------------------
# DERIVADAS NUMÉRICAS
#-------------------------------------

def derivada(p, x, y, z, componente, eje, h):
    coordenadas_positivas = [x, y, z]
    coordenadas_negativas = [x, y, z]

    coordenadas_positivas[eje] += h
    coordenadas_negativas[eje] -= h

    c_positivo = calcular_campos(
        p, *coordenadas_positivas
    )

    c_negativo = calcular_campos(
        p, *coordenadas_negativas
    )

    return (
        getattr(c_positivo, componente)
        - getattr(c_negativo, componente)
    ) / (2 * h)


#-------------------------------------
# ECUACIONES DE MAXWELL
#-------------------------------------

def validar_maxwell(
    p: ParametrosModo,
    x: float,
    y: float,
    z: float = 0.0,
    h: float = 1e-6,
) -> ResultadoValidacion:
    if not (0 < x < p.a and 0 < y < p.b):
        raise ValueError(
            "El punto debe estar dentro de la guía."
        )

    if h <= 0 or not np.isfinite(h):
        raise ValueError("h debe ser finito y positivo.")

    if min(x, p.a - x, y, p.b - y) <= h:
        raise ValueError(
            "El paso h debe permanecer dentro de la guía."
        )

    c = calcular_campos(p, x, y, z)

    # Derivadas del campo eléctrico
    dEz_dy = derivada(p, x, y, z, "Ez", 1, h)
    dEy_dz = derivada(p, x, y, z, "Ey", 2, h)

    dEx_dz = derivada(p, x, y, z, "Ex", 2, h)
    dEz_dx = derivada(p, x, y, z, "Ez", 0, h)

    dEy_dx = derivada(p, x, y, z, "Ey", 0, h)
    dEx_dy = derivada(p, x, y, z, "Ex", 1, h)

    rot_e = np.array([
        dEz_dy - dEy_dz,
        dEx_dz - dEz_dx,
        dEy_dx - dEx_dy,
    ], dtype=complex).reshape(3)

    # Derivadas del campo magnético
    dHz_dy = derivada(p, x, y, z, "Hz", 1, h)
    dHy_dz = derivada(p, x, y, z, "Hy", 2, h)

    dHx_dz = derivada(p, x, y, z, "Hx", 2, h)
    dHz_dx = derivada(p, x, y, z, "Hz", 0, h)

    dHy_dx = derivada(p, x, y, z, "Hy", 0, h)
    dHx_dy = derivada(p, x, y, z, "Hx", 1, h)

    rot_h = np.array([
        dHz_dy - dHy_dz,
        dHx_dz - dHz_dx,
        dHy_dx - dHx_dy,
    ], dtype=complex).reshape(3)

    E = np.array([c.Ex, c.Ey, c.Ez], dtype=complex).reshape(3)
    H = np.array([c.Hx, c.Hy, c.Hz], dtype=complex).reshape(3)

    # Residuos de las ecuaciones de Maxwell
    r_faraday = rot_e + 1j * p.omega * p.mu * H
    r_ampere = rot_h - 1j * p.omega * p.epsilon * E

    # Divergencias
    div_e = (
        derivada(p, x, y, z, "Ex", 0, h)
        + derivada(p, x, y, z, "Ey", 1, h)
        + derivada(p, x, y, z, "Ez", 2, h)
    )

    div_h = (
        derivada(p, x, y, z, "Hx", 0, h)
        + derivada(p, x, y, z, "Hy", 1, h)
        + derivada(p, x, y, z, "Hz", 2, h)
    )

    # Normalización para comparar errores
    escala_e = max(
        np.linalg.norm(rot_e),
        p.omega * p.mu * np.linalg.norm(H),
        1e-30,
    )

    escala_h = max(
        np.linalg.norm(rot_h),
        p.omega * p.epsilon * np.linalg.norm(E),
        1e-30,
    )

    escala_div_e = max(
        p.kc * np.linalg.norm(E), 1e-30
    )

    escala_div_h = max(
        p.kc * np.linalg.norm(H), 1e-30
    )

    error_faraday = float(
        np.linalg.norm(r_faraday) / escala_e
    )

    error_ampere = float(
        np.linalg.norm(r_ampere) / escala_h
    )

    error_gauss_e = float(
        abs(div_e) / escala_div_e
    )

    error_gauss_h = float(
        abs(div_h) / escala_div_h
    )

    error_fronteras = validar_fronteras(p)

    return ResultadoValidacion(
        error_faraday=error_faraday,
        error_ampere=error_ampere,
        error_gauss_e=error_gauss_e,
        error_gauss_h=error_gauss_h,
        error_fronteras=error_fronteras,
    )


#-------------------------------------
# CONDICIONES DE FRONTERA
#-------------------------------------

def validar_fronteras(p: ParametrosModo, puntos=100):
    x = np.linspace(0, p.a, puntos)
    y = np.linspace(0, p.b, puntos)

    errores = []

    # Paredes x=0 y x=a: Ey y Ez tangenciales
    for xb in (0.0, p.a):
        c = calcular_campos(p, xb, y)
        errores.extend([
            np.max(np.abs(c.Ey)),
            np.max(np.abs(c.Ez)),
        ])

    # Paredes y=0 y y=b: Ex y Ez tangenciales
    for yb in (0.0, p.b):
        c = calcular_campos(p, x, yb)
        errores.extend([
            np.max(np.abs(c.Ex)),
            np.max(np.abs(c.Ez)),
        ])

    return float(max(errores))
