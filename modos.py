
"""
modos.py

Validación modal y cálculo de parámetros electromagnéticos
para guías de onda rectangulares ideales.

Convención fasorial:
    exp(j*omega*t - j*beta*z)

Para modos evanescentes:
    beta = -1j*alpha

Unidades: Sistema Internacional (SI).
"""

from dataclasses import dataclass
from math import pi, sqrt, hypot, isfinite
from numbers import Integral

# Constantes físicas en unidades SI
EPSILON_0 = 8.8541878128e-12  # F/m
MU_0 = 1.25663706212e-6       # H/m


@dataclass(frozen=True)
class ParametrosModo:
    modo: str
    m: int
    n: int
    a: float
    b: float
    frecuencia: float

    epsilon: float
    mu: float
    omega: float

    k: float
    kx: float
    ky: float
    kc: float

    frecuencia_corte: float
    beta: complex
    alfa: float
    estado: str


def validar_modo(modo: str, m: int, n: int) -> None:
    """Valida el tipo de modo y sus índices."""

    if not isinstance(modo, str):
        raise ValueError("El modo debe ser TE o TM.")

    modo = modo.upper().strip()

    if modo not in ("TE", "TM"):
        raise ValueError("El modo debe ser TE o TM.")
    #***********************************************************************************
    #Protección contra valores no enteros o booleanos, y contra valores negativos
    if (
        isinstance(m, bool)
        or isinstance(n, bool)
        or not isinstance(m, Integral)
        or not isinstance(n, Integral)
    ):
        raise ValueError("m y n deben ser enteros.")

    if m < 0 or n < 0:
        raise ValueError("m y n no pueden ser negativos.")

    if modo == "TE" and m == 0 and n == 0:
        raise ValueError("El modo TE00 no existe.")

    if modo == "TM" and (m == 0 or n == 0):
        raise ValueError(
            "Los modos TM requieren m >= 1 y n >= 1."
        )
    #***********************************************************************************


def calcular_parametros(
    modo: str,
    m: int,
    n: int,
    a: float,
    b: float,
    frecuencia: float,
    epsilon_r: float = 1.0,
    mu_r: float = 1.0,
) -> ParametrosModo:
    """
    Calcula los parámetros de un modo TE o TM.

    Parámetros:
        modo: 'TE' o 'TM'
        m, n: índices modales
        a, b: dimensiones de la guía [m]
        frecuencia: frecuencia de operación [Hz]
        epsilon_r: permitividad relativa
        mu_r: permeabilidad relativa
    """

    validar_modo(modo, m, n)
    modo = modo.upper().strip()
    # Diccionario para validar y convertir los valores a float
    valores = {
        "a": a,
        "b": b,
        "frecuencia": frecuencia,
        "epsilon_r": epsilon_r,
        "mu_r": mu_r,
    }
    # Validar que los parámetros sean numéricos, finitos y positivos.
    for nombre, valor in valores.items():
        try:
            numero = float(valor)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"{nombre} debe ser un número positivo."
            ) from None

        if not isfinite(numero) or numero <= 0:
            raise ValueError(
                f"{nombre} debe ser finito y positivo."
            )

        valores[nombre] = numero
    # Recuperar los parámetros físicos ya validados y convertidos a float.
    a = valores["a"]
    b = valores["b"]
    frecuencia = valores["frecuencia"]
    epsilon_r = valores["epsilon_r"]
    mu_r = valores["mu_r"]

    # Constantes y parámetros electromagnéticos calculados
    epsilon = EPSILON_0 * epsilon_r
    mu = MU_0 * mu_r

    omega = 2 * pi * frecuencia

    k = omega * sqrt(mu * epsilon)

    kx = m * pi / a
    ky = n * pi / b

    kc = hypot(kx, ky)

    frecuencia_corte = kc / (
        2 * pi * sqrt(mu * epsilon)
    )

    # Tolerancia numérica para el corte
    tolerancia = 1e-12

    diferencia = frecuencia - frecuencia_corte
    escala = max(frecuencia, frecuencia_corte)

    if abs(diferencia) <= tolerancia * escala:
        estado = "En el corte"
        beta = 0j
        alfa = 0.0

    elif frecuencia > frecuencia_corte:
        estado = "Propagante"
        beta = complex(
            sqrt(max(0.0, k**2 - kc**2)), 0.0
        )
        alfa = 0.0

    else:
        estado = "Evanescente"
        alfa = sqrt(max(0.0, kc**2 - k**2))
        beta = -1j * alfa

    return ParametrosModo(
        modo=modo,
        m=m,
        n=n,
        a=a,
        b=b,
        frecuencia=frecuencia,
        epsilon=epsilon,
        mu=mu,
        omega=omega,
        k=k,
        kx=kx,
        ky=ky,
        kc=kc,
        frecuencia_corte=frecuencia_corte,
        beta=beta,
        alfa=alfa,
        estado=estado,
    )


