
#-------------------------------------
# GRÁFICAS DE CAMPOS
#-------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from modos import ParametrosModo
from campos import calcular_campos, magnitud_campo


#-------------------------------------
# FUNCIONES AUXILIARES
#-------------------------------------

def fase_grados(grados):
    if not np.isfinite(grados):
        raise ValueError("La fase debe ser finita.")

    return np.deg2rad(grados)


def crear_malla(p: ParametrosModo, nx=101, ny=81):
    if not isinstance(p, ParametrosModo):
        raise TypeError("p debe ser un objeto ParametrosModo.")

    if (
        not isinstance(nx, (int, np.integer))
        or not isinstance(ny, (int, np.integer))
        or isinstance(nx, (bool, np.bool_))
        or isinstance(ny, (bool, np.bool_))
        or nx < 3
        or ny < 3
    ):
        raise ValueError(
            "nx y ny deben ser enteros mayores o iguales a 3."
        )

    x = np.linspace(0, p.a, nx)
    y = np.linspace(0, p.b, ny)

    return np.meshgrid(x, y)


def obtener_componentes(p, X, Y, z=0.0, fase=0.0):
    if not np.isfinite(fase):
        raise ValueError("La fase debe ser finita.")

    c = calcular_campos(p, X, Y, z=z)

    factor = np.exp(1j * fase)

    E = tuple(
        np.real(componente * factor)
        for componente in (c.Ex, c.Ey, c.Ez)
    )

    H = tuple(
        np.real(componente * factor)
        for componente in (c.Hx, c.Hy, c.Hz)
    )

    return E, H


def obtener_vector(E, H, campo):
    if not isinstance(campo, str):
        raise ValueError("El campo debe ser E o H.")

    campo = campo.upper().strip()

    if campo == "E":
        return E

    if campo == "H":
        return H

    raise ValueError("El campo debe ser E o H.")


def validar_resolucion(n, nombre):
    if (
        not isinstance(n, (int, np.integer))
        or isinstance(n, (bool, np.bool_))
        or n < 3
    ):
        raise ValueError(
            f"{nombre} debe ser un entero mayor o igual a 3."
        )


def mostrar_campo_nulo(ax, mensaje):
    ax.text(
        0.5,
        0.5,
        mensaje,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


#-------------------------------------
# LÍNEAS DE CAMPO TRANSVERSALES
#-------------------------------------

def graficar_lineas(
    p: ParametrosModo,
    campo="E",
    fase=0.0,
    z=0.0,
    nx=101,
    ny=81,
):
    X, Y = crear_malla(p, nx, ny)

    E, H = obtener_componentes(
        p, X, Y, z=z, fase=fase
    )

    Fx, Fy, _ = obtener_vector(E, H, campo)

    magnitud = np.hypot(Fx, Fy)
    maximo = float(np.max(magnitud))

    fig, ax = plt.subplots(figsize=(8, 5))

    if maximo > 1e-12:

        ax.streamplot(
            X[0, :],
            Y[:, 0],
            Fx,
            Fy,
            color=magnitud / maximo,
            cmap="viridis",
            density=1.5,
            linewidth=1.2,
            arrowsize=1.0,
        )

    else:
        mostrar_campo_nulo(
            ax,
            "Campo transversal nulo en esta fase",
        )

    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            f"Líneas de campo {campo.upper()} - "
            f"{p.modo}{p.m}{p.n}"
        ),
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, p.a)
    ax.set_ylim(0, p.b)

    fig.tight_layout()

    return fig


#-------------------------------------
# VECTORES DE CAMPO TRANSVERSALES
#-------------------------------------

def graficar_vectores(
    p: ParametrosModo,
    campo="E",
    fase=0.0,
    z=0.0,
    nx=21,
    ny=15,
):
    X, Y = crear_malla(p, nx, ny)

    E, H = obtener_componentes(
        p, X, Y, z=z, fase=fase
    )

    Fx, Fy, _ = obtener_vector(E, H, campo)

    magnitud = np.hypot(Fx, Fy)
    maximo = float(np.max(magnitud))

    fig, ax = plt.subplots(figsize=(8, 5))

    if maximo > 1e-12:

        ax.quiver(
            X,
            Y,
            Fx / maximo,
            Fy / maximo,
            magnitud / maximo,
            cmap="viridis",
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=60 / max(p.a, p.b),
        )

    else:
        mostrar_campo_nulo(
            ax,
            "Campo transversal nulo en esta fase",
        )

    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            f"Vectores de campo {campo.upper()} - "
            f"{p.modo}{p.m}{p.n}"
        ),
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, p.a)
    ax.set_ylim(0, p.b)

    fig.tight_layout()

    return fig


#-------------------------------------
# MAPAS DE MAGNITUD FASORIAL
#-------------------------------------

def graficar_magnitud(
    p: ParametrosModo,
    campo="E",
    z=0.0,
    nx=101,
    ny=81,
):
    X, Y = crear_malla(p, nx, ny)

    c = calcular_campos(p, X, Y, z=z)

    campo = campo.upper().strip()

    if campo == "E":

        M = magnitud_campo(
            c.Ex, c.Ey, c.Ez
        )

        unidad = "V/m"

    elif campo == "H":

        M = magnitud_campo(
            c.Hx, c.Hy, c.Hz
        )

        unidad = "A/m"

    else:
        raise ValueError("El campo debe ser E o H.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if np.max(M) > 0:

        grafica = ax.contourf(
            X,
            Y,
            M,
            levels=30,
            cmap="viridis",
        )

        fig.colorbar(
            grafica,
            ax=ax,
            label=f"Magnitud ({unidad})",
        )

    else:
        mostrar_campo_nulo(
            ax,
            "Magnitud del campo nula",
        )

    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            f"Magnitud fasorial {campo} - "
            f"{p.modo}{p.m}{p.n}"
        ),
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, p.a)
    ax.set_ylim(0, p.b)

    fig.tight_layout()

    return fig


#-------------------------------------
# CORTES LONGITUDINALES
#-------------------------------------

def graficar_longitudinal(
    p: ParametrosModo,
    campo="E",
    plano="xz",
    fase=0.0,
    longitud=0.10,
    posicion=None,
    representacion="lineas",
    n_transversal=101,
    nz=151,
):
    # Validación de parámetros

    if not np.isfinite(longitud) or longitud <= 0:
        raise ValueError(
            "La longitud debe ser finita y positiva."
        )

    validar_resolucion(
        n_transversal, "n_transversal"
    )

    validar_resolucion(nz, "nz")

    campo = campo.upper().strip()
    plano = plano.lower().strip()
    representacion = representacion.lower().strip()

    if campo not in ("E", "H"):
        raise ValueError("El campo debe ser E o H.")

    if plano not in ("xz", "yz"):
        raise ValueError("El plano debe ser xz o yz.")

    if representacion not in ("lineas", "vectores"):
        raise ValueError(
            "La representación debe ser lineas o vectores."
        )

    # Construcción del corte espacial

    z = np.linspace(0, longitud, nz)

    if plano == "xz":

        posicion = (
            p.b / 2 if posicion is None else posicion
        )

        if (
            not np.isfinite(posicion)
            or not 0 <= posicion <= p.b
        ):
            raise ValueError(
                "La posición y está fuera de la guía."
            )

        x = np.linspace(0, p.a, n_transversal)

        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, posicion)

        coordenada = X
        eje_transversal = x
        etiqueta = "x (m)"
        limite = p.a

    else:

        posicion = (
            p.a / 2 if posicion is None else posicion
        )

        if (
            not np.isfinite(posicion)
            or not 0 <= posicion <= p.a
        ):
            raise ValueError(
                "La posición x está fuera de la guía."
            )

        y = np.linspace(0, p.b, n_transversal)

        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, posicion)

        coordenada = Y
        eje_transversal = y
        etiqueta = "y (m)"
        limite = p.b

    E, H = obtener_componentes(
        p, X, Y, z=Z, fase=fase
    )

    Fx, Fy, Fz = obtener_vector(E, H, campo)

    # Componentes proyectadas sobre el plano
    U = Fx if plano == "xz" else Fy
    V = Fz

    magnitud = np.hypot(U, V)
    maximo = float(np.max(magnitud))

    fig, ax = plt.subplots(figsize=(10, 5))

    if maximo > 1e-12:

        magnitud_normalizada = magnitud / maximo

        if representacion == "lineas":

            ax.streamplot(
                eje_transversal,
                z,
                U,
                V,
                color=magnitud_normalizada,
                cmap="viridis",
                density=1.4,
                linewidth=1.2,
                arrowsize=1.0,
                broken_streamlines=True,
            )

        else:

            paso_z = max(1, nz // 22)
            paso_t = max(
                1, n_transversal // 18
            )

            ax.contourf(
                coordenada,
                Z,
                magnitud_normalizada,
                levels=25,
                cmap="viridis",
            )

            ax.quiver(
                coordenada[::paso_z, ::paso_t],
                Z[::paso_z, ::paso_t],
                U[::paso_z, ::paso_t] / maximo,
                V[::paso_z, ::paso_t] / maximo,
                color="white",
                pivot="mid",
            )

    else:

        mostrar_campo_nulo(
            ax,
            "Campo proyectado nulo en este plano y fase",
        )

    ax.set(
        xlabel=etiqueta,
        ylabel="z (m)",
        title=(
            f"Campo {campo} - "
            f"{p.modo}{p.m}{p.n} - "
            f"Plano {plano}"
        ),
    )

    ax.set_xlim(0, limite)
    ax.set_ylim(0, longitud)

    fig.tight_layout()

    return fig
