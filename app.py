"""Interfaz Streamlit para la simulación de modos TE y TM.

La aplicación es deliberadamente una capa de presentación: todos los
cálculos electromagnéticos y las figuras se delegan a los módulos del
proyecto (`modos`, `campos`, `validacion` y `graficas`).
"""

from io import BytesIO
from math import isfinite

import matplotlib.pyplot as plt
import streamlit as st

from graficas import (
    fase_grados,
    graficar_lineas,
    graficar_longitudinal,
    graficar_magnitud,
    graficar_vectores,
)
from modos import calcular_parametros
from validacion import validar_maxwell


st.set_page_config(
    page_title="Guías de onda TE/TM",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Tipografía local: no requiere descargar fuentes externas. */
    html, body, .stApp, input, textarea, button,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stMetric"], [data-baseweb="select"],
    [data-testid="stTable"] {
        font-family: "Courier New", Courier, monospace;
    }
    .stApp { background: #f6f8f6; color: #182c23; }
    .block-container { max-width: 1440px; padding: 3rem 2.5rem 4rem; }
    [data-testid="stHeader"] { background: rgba(246,248,246,.92); }
    [data-testid="stSidebar"] {
        background: #edf3ee; border-right: 1px solid #dce6de;
    }
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        color: #182c23; letter-spacing: -.035em;
    }
    h2, h3 { font-weight: 600 !important; }
    .hero {
        position: relative; padding: 2.5rem 2.7rem; border-radius: 26px;
        background: #ffffff; border: 1px solid #e0e8e2;
        box-shadow: 0 12px 40px rgba(20,56,34,.035); margin-bottom: 1.6rem;
        overflow: hidden;
    }
    .hero::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0;
        height: 4px; background: #007a3d;
    }
    .hero .eyebrow {
        color: #007a3d; font-size: .75rem; letter-spacing: .16em;
        font-weight: bold; margin-bottom: 1.25rem;
    }
    .hero h1 { color: #182c23; font-size: clamp(2rem, 3.5vw, 3.4rem);
               line-height: 1.08; font-weight: 650; margin: 0 0 .8rem; padding: 0; }
    .hero p { color: #52675b; max-width: 760px; line-height: 1.65; margin: 0; }
    .hero .tags { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.4rem; }
    .hero .tags span { border: 1px solid #d8e7dc; border-radius: 100px;
                      padding: .3rem .7rem; color: #24633d; font-size: .72rem; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e0e8e2; border-radius: 18px;
        padding: 1rem 1.15rem; min-height: 105px;
    }
    [data-testid="stMetricValue"] { color: #006b35; font-size: clamp(1.05rem, 1.45vw, 1.5rem); }
    [data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; }
    [data-testid="stMetricLabel"] { color: #52675b; }
    [data-testid="stExpander"], [data-testid="stForm"] {
        background: #fff; border: 1px solid #e0e8e2; border-radius: 18px;
    }
    [data-testid="stImage"] {
        background: white; border: 1px solid #e0e8e2; border-radius: 18px;
        padding: .6rem; overflow: hidden;
    }
    [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 12px; border: 1px solid #bed5c5;
        background: #fff; color: #006b35;
        transition: background .18s ease, border-color .18s ease;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #e5f2e9; border-color: #007a3d; color: #004e28;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #007a3d; border-color: #007a3d; color: #fff;
    }
    button:focus-visible, input:focus-visible {
        outline: 2px solid #007a3d !important; outline-offset: 3px;
    }
    [data-testid="stCaptionContainer"] { color: #52675b; }
    hr { border-color: #dce6de !important; margin: 1.8rem 0 !important; }
    @media (max-width: 700px) {
        .block-container { padding: 1.5rem 1rem 3rem; }
        .hero { padding: 1.7rem 1.3rem; border-radius: 20px; }
    }
    @media (prefers-reduced-motion: reduce) {
        button { transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _formatear_frecuencia(valor: float) -> str:
    """Devuelve una frecuencia en una unidad cómoda para lectura."""
    if valor >= 1e9:
        return f"{valor / 1e9:.5g} GHz"
    if valor >= 1e6:
        return f"{valor / 1e6:.5g} MHz"
    return f"{valor:.5g} Hz"


def _formatear_beta(beta: complex) -> str:
    if abs(beta.imag) < 1e-12:
        return f"{beta.real:.5g} rad/m"
    return f"{beta.imag:.5g}j rad/m"


def _figura_bytes(fig, extension: str) -> bytes:
    """Serializa una figura Matplotlib para el botón de descarga."""
    buffer = BytesIO()
    fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
    return buffer.getvalue()


def _mostrar_parametros(p) -> None:
    """Presenta los parámetros calculados sin repetir la matemática."""
    columnas = st.columns(4)
    columnas[0].metric("Modo", f"{p.modo}({p.m}, {p.n})")
    columnas[1].metric("Estado", p.estado)
    columnas[2].metric("Frecuencia de corte", _formatear_frecuencia(p.frecuencia_corte))
    columnas[3].metric("β", _formatear_beta(p.beta))

    with st.expander("Parámetros electromagnéticos y de propagación"):
        datos = {
            "Dimensiones": f"a = {p.a * 1e3:.4g} mm · b = {p.b * 1e3:.4g} mm",
            "Frecuencia de operación": _formatear_frecuencia(p.frecuencia),
            "kₓ": f"{p.kx:.6g} rad/m",
            "kᵧ": f"{p.ky:.6g} rad/m",
            "kᶜ": f"{p.kc:.6g} rad/m",
            "β": _formatear_beta(p.beta),
            "α": f"{p.alfa:.6g} Np/m",
            "ε": f"{p.epsilon:.6g} F/m",
            "μ": f"{p.mu:.6g} H/m",
        }
        st.table(datos)


def main() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">UVG · TEORÍA ELECTROMAGNÉTICA 2</div>'
        '<h1>Guías de onda<br>rectangulares.</h1>'
        '<p>Exploración interactiva de los modos transversales eléctricos (TE) '
        'y transversales magnéticos (TM).</p>'
        '<div class="tags"><span>MODOS TE / TM</span><span>CAMPOS E / H</span>'
        '<span>GUÍA RECTANGULAR IDEAL</span></div></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div style="color:#007a3d;font-weight:bold;letter-spacing:.15em;'
                    'font-size:.8rem;margin-bottom:1rem">UVG / Joaquín Fuentes</div>',
                    unsafe_allow_html=True)
        st.header("Configuración")
        st.caption("Unidades geométricas: milímetros · frecuencia: GHz")
        modo = st.radio("Tipo de modo", ["TE", "TM"], horizontal=True)
        m_texto = st.text_input("Índice m", value="1")
        n_texto = st.text_input(
            "Índice n", value="0" if modo == "TE" else "1",
            help="Para TM, ambos índices deben ser mayores o iguales a 1.",
        )

        st.subheader("Geometría y medio")
        a_mm = st.number_input("Ancho a (mm)", min_value=0.01, value=40.0, step=1.0)
        b_mm = st.number_input("Alto b (mm)", min_value=0.01, value=20.0, step=1.0)
        frecuencia_ghz = st.number_input(
            "Frecuencia de operación (GHz)", min_value=0.000001,
            value=5.0 if modo == "TE" else 10.0, step=0.5, format="%.6f",
        )
        epsilon_r = st.number_input("Permitividad relativa εᵣ", min_value=0.000001, value=1.0, step=0.1)
        mu_r = st.number_input("Permeabilidad relativa μᵣ", min_value=0.000001, value=1.0, step=0.1)

        st.subheader("Visualización")
        fase_deg = st.slider("Fase temporal (°)", 0.0, 360.0, 90.0, 5.0)
        z_mm = st.number_input("Posición transversal z (mm)", min_value=0.0, value=0.0, step=1.0)
        st.subheader("Resolución")
        nx = st.slider("Puntos en x", 21, 501, 81, 10)
        ny = st.slider("Puntos en y", 21, 501, 61, 10)

    try:
        try:
            m, n = int(m_texto), int(n_texto)
        except ValueError:
            raise ValueError("Los índices m y n deben ser enteros no negativos.") from None
        p = calcular_parametros(
            modo, m, n, a_mm * 1e-3, b_mm * 1e-3,
            frecuencia_ghz * 1e9, epsilon_r, mu_r,
        )
        if not all(isfinite(v) for v in (
            p.k, p.kx, p.ky, p.kc, p.omega, p.frecuencia_corte,
            p.beta.real, p.beta.imag, p.epsilon, p.mu,
        )):
            raise OverflowError
    except (OverflowError, ZeroDivisionError):
        st.error("La configuración supera la precisión numérica disponible. Reduzca la escala de los valores.")
        return
    except ValueError as exc:
        st.error(f"Configuración no válida: {exc}")
        st.info("TE permite m o n igual a cero, excepto TE₀₀. TM exige m ≥ 1 y n ≥ 1.")
        return

    _mostrar_parametros(p)
    if p.estado == "Propagante":
        st.success("El modo es propagante a la frecuencia seleccionada.")
    elif p.estado == "Evanescente":
        st.warning("El modo es evanescente: no transporta propagación longitudinal y decae con z.")
    else:
        st.warning("La frecuencia coincide con el corte; β = 0.")

    st.divider()
    col_izq, col_der = st.columns([1, 2])
    with col_izq:
        st.subheader("Representación")
        vista = st.selectbox(
            "Vista de campo",
            ["Líneas transversales", "Vectores transversales", "Magnitud fasorial", "Corte longitudinal"],
        )
        campo = st.radio("Campo", ["E", "H"], horizontal=True)
        representacion = "lineas"
        plano = "xz"
        longitud_mm = 100.0
        posicion_mm = b_mm / 2
        if vista == "Corte longitudinal":
            plano = st.radio("Plano longitudinal", ["xz", "yz"], horizontal=True)
            representacion = st.radio("Estilo", ["lineas", "vectores"], horizontal=True)
            longitud_mm = st.number_input("Longitud del corte (mm)", min_value=0.01, value=100.0, step=10.0)
            limite_posicion = b_mm if plano == "xz" else a_mm
            posicion_mm = st.number_input(
                f"Posición {'y' if plano == 'xz' else 'x'} (mm)",
                min_value=0.0, max_value=float(limite_posicion), value=float(limite_posicion / 2), step=1.0,
            )
        st.caption("Las líneas y flechas muestran campos instantáneos; la magnitud fasorial no depende de la fase temporal.")
        st.caption("Amplitud longitudinal de referencia: H₀ = 1 A/m para TE; E₀ = 1 V/m para TM.")

    puntos_x = max(11, nx // 4) if vista == "Vectores transversales" else nx
    puntos_y = max(9, ny // 4) if vista == "Vectores transversales" else ny
    if vista == "Corte longitudinal":
        indice = m if plano == "xz" else n
        insuficiente = nx < 10 * indice + 1
        ciclos_z = abs(p.beta.real) * longitud_mm * 1e-3 / (2 * 3.141592653589793)
        insuficiente = insuficiente or max(31, ny) < 20 * ciclos_z + 1
    else:
        insuficiente = puntos_x < 10 * m + 1 or puntos_y < 10 * n + 1
    if insuficiente:
        st.warning("La malla puede omitir variaciones del campo para estos índices o esta longitud. Aumente la resolución o reduzca la longitud del corte. El modo sigue siendo físicamente válido; la figura es una aproximación limitada por el muestreo.")

    fase = fase_grados(fase_deg)
    fig = None
    try:
        if vista == "Líneas transversales":
            fig = graficar_lineas(p, campo=campo, fase=fase, z=z_mm * 1e-3, nx=nx, ny=ny)
        elif vista == "Vectores transversales":
            fig = graficar_vectores(p, campo=campo, fase=fase, z=z_mm * 1e-3, nx=max(11, nx // 4), ny=max(9, ny // 4))
        elif vista == "Magnitud fasorial":
            fig = graficar_magnitud(p, campo=campo, z=z_mm * 1e-3, nx=nx, ny=ny)
        else:
            fig = graficar_longitudinal(
                p, campo=campo, plano=plano, fase=fase, longitud=longitud_mm * 1e-3,
                posicion=posicion_mm * 1e-3, representacion=representacion,
                n_transversal=nx, nz=max(31, ny),
            )
    except (ValueError, FloatingPointError) as exc:
        st.error(f"No fue posible generar la figura: {exc}")
        return

    try:
        with col_der:
            st.pyplot(fig, width="stretch")
            nombre = f"{p.modo}_{p.m}_{p.n}_{campo}_{vista.lower().replace(' ', '_')}"
            descargas = st.columns(3)
            for columna, extension, mime in zip(descargas, ("png", "svg", "pdf"), ("image/png", "image/svg+xml", "application/pdf")):
                columna.download_button(
                    f"Descargar {extension.upper()}", _figura_bytes(fig, extension),
                    file_name=f"{nombre}.{extension}", mime=mime, width="stretch",
                    on_click="ignore",
                )
    finally:
        plt.close(fig)

    st.divider()
    st.subheader("Verificación física")
    with st.form("verificacion"):
        st.caption("Evalúa las ecuaciones en un punto interior y el campo eléctrico tangencial sobre las paredes PEC.")
        x_rel = st.slider("Posición x/a (%)", 1.0, 99.0, 33.0, 1.0)
        y_rel = st.slider("Posición y/b (%)", 1.0, 99.0, 35.0, 1.0)
        ejecutar_validacion = st.form_submit_button("Verificar Maxwell y fronteras")
    if ejecutar_validacion:
        x_val, y_val = p.a * x_rel / 100, p.b * y_rel / 100
        if p.alfa * z_mm * 1e-3 > 700:
            st.warning("El campo evanescente está por debajo del rango numérico útil en este corte. Acerque z al origen para verificar Maxwell; residuos nulos aquí no demostrarían precisión.")
            return
        # Resuelve la escala espacial más corta y mantiene el stencil interior.
        h = min(min(p.a, p.b) / 20000, 1e-4 / max(p.kc, abs(p.beta)))
        try:
            resultado = validar_maxwell(p, x_val, y_val, z_mm * 1e-3, h=h)
            errores = {
                "Faraday": resultado.error_faraday,
                "Ampère": resultado.error_ampere,
                "Gauss (E)": resultado.error_gauss_e,
                "Gauss (H)": resultado.error_gauss_h,
                "Fronteras PEC": resultado.error_fronteras,
            }
            filas = []
            for clave, valor in errores.items():
                tolerancia = 1e-8 if clave == "Fronteras PEC" else 1e-4
                filas.append({"Comprobación": clave, "Residuo": f"{valor:.3e}",
                              "Tolerancia": f"{tolerancia:.0e}",
                              "Resultado": "Cumple" if isfinite(valor) and valor < tolerancia else "Revisar"})
            st.table(filas)
            if all(fila["Resultado"] == "Cumple" for fila in filas):
                st.success("Los residuos evaluados cumplen las tolerancias de las pruebas del proyecto.")
            else:
                st.warning("Algún residuo requiere revisión. Compruebe el punto y la escala numérica; este resultado no demuestra por sí solo que el modo sea inválido.")
            st.caption(f"Punto: x = {x_val:.5g} m, y = {y_val:.5g} m, z = {z_mm * 1e-3:.5g} m; paso h = {h:.3e} m. Los cuatro residuos diferenciales son normalizados; el de frontera es absoluto (V/m), con la amplitud de referencia indicada.")
            st.caption("Esta comprobación local por diferencias finitas no sustituye una validación de toda la distribución. En regiones donde el campo evanescente es numéricamente nulo, los residuos pueden ser cero por subdesbordamiento.")
        except (ValueError, OverflowError, FloatingPointError) as exc:
            st.warning(f"No se pudo evaluar el punto seleccionado: {exc}")


if __name__ == "__main__":
    main()
