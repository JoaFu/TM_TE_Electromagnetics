
"""
campos.py

Cálculo analítico de las seis componentes de los campos
electromagnéticos en una guía de onda rectangular ideal.

Convención fasorial:
    exp(j*omega*t - j*beta*z)

Unidades:
    E: V/m
    H: A/m
    x, y, z: m

Las amplitudes E0 y H0 son parámetros de normalización.
"""

from dataclasses import dataclass
import numpy as np

from modos import ParametrosModo


@dataclass(frozen=True)
class CamposEM:
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray


def calcular_campos(
    p: ParametrosModo,
    x,
    y,
    z=0.0,
    amplitud=1.0,
) -> CamposEM:
    """
    Calcula los fasores de los campos E y H.

    Parámetros:
        p: objeto ParametrosModo de modos.py
        x, y, z: coordenadas espaciales [m].
                 Admiten escalares o arreglos compatibles.
        amplitud: E0 para TM [V/m] o H0 para TE [A/m].

    Retorna:
        CamposEM con las seis componentes complejas.
    """
    # Validaciones de entrada
    if not isinstance(p, ParametrosModo):
        raise TypeError(
            "p debe ser un objeto ParametrosModo."
        )

    if not np.isfinite(amplitud):
        raise ValueError("La amplitud debe ser finita.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(z))
    ):
        raise ValueError(
            "Las coordenadas deben ser finitas."
        )

    # Crear arreglos compatibles por broadcasting
    x, y, z = np.broadcast_arrays(x, y, z)

    # Factor longitudinal, válido para propagación
    # y para atenuación evanescente.
    F = np.exp(-1j * p.beta * z)

    # Funciones trigonométricas transversales
    sx = np.sin(p.kx * x)
    cx = np.cos(p.kx * x)

    sy = np.sin(p.ky * y)
    cy = np.cos(p.ky * y)

    # Inicializar las seis componentes complejas
    Ex = np.zeros(x.shape, dtype=complex)
    Ey = np.zeros(x.shape, dtype=complex)
    Ez = np.zeros(x.shape, dtype=complex)

    Hx = np.zeros(x.shape, dtype=complex)
    Hy = np.zeros(x.shape, dtype=complex)
    Hz = np.zeros(x.shape, dtype=complex)

    if p.modo == "TM":

        # Componente longitudinal eléctrica
        Ez = amplitud * sx * sy * F

        # Derivadas analíticas de Ez
        dEz_dx = amplitud * p.kx * cx * sy * F
        dEz_dy = amplitud * p.ky * sx * cy * F

        # Componentes transversales
        Ex = -1j * p.beta / p.kc**2 * dEz_dx
        Ey = -1j * p.beta / p.kc**2 * dEz_dy

        Hx = 1j * p.omega * p.epsilon / p.kc**2 * dEz_dy
        Hy = -1j * p.omega * p.epsilon / p.kc**2 * dEz_dx

    elif p.modo == "TE":

        # Componente longitudinal magnética
        Hz = amplitud * cx * cy * F

        # Derivadas analíticas de Hz
        dHz_dx = -amplitud * p.kx * sx * cy * F
        dHz_dy = -amplitud * p.ky * cx * sy * F

        # Componentes transversales
        Ex = -1j * p.omega * p.mu / p.kc**2 * dHz_dy
        Ey = 1j * p.omega * p.mu / p.kc**2 * dHz_dx

        Hx = -1j * p.beta / p.kc**2 * dHz_dx
        Hy = -1j * p.beta / p.kc**2 * dHz_dy

    else:
        raise ValueError("Tipo de modo no reconocido.")

    return CamposEM(
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        Hx=Hx,
        Hy=Hy,
        Hz=Hz,
    )


def campo_instantaneo(componente, frecuencia, tiempo):
    """
    Convierte un fasor a su valor físico instantáneo.

    componente: arreglo complejo
    frecuencia: Hz
    tiempo: segundos
    """
    omega = 2 * np.pi * frecuencia

    return np.real(
        np.asarray(componente) * np.exp(1j * omega * tiempo)
    )


def magnitud_campo(c1, c2, c3):
    """
    Calcula la magnitud vectorial de un campo fasorial.
    """
    return np.sqrt(
        np.abs(c1)**2 +
        np.abs(c2)**2 +
        np.abs(c3)**2
    )
