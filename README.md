# Simulación de modos TE y TM en guías rectangulares

Aplicación académica para calcular y visualizar soluciones analíticas de los campos
en una guía rectangular ideal. La interfaz está en español y se ejecuta localmente.
El informe académico vigente está en `documento_article/Informe_TE_TM.docx`. La
plantilla original `documento_article/SciencePG_Manuscript_Template.docx` se conserva
como referencia y no es una versión del informe.

## Requisitos

- Windows y Python 3.13.
- Un navegador web actual.
- Conexión a Internet solo para instalar las dependencias.

## Preparación

Desde PowerShell, en la carpeta que contiene `app.py`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si ya existe un entorno `.venv` funcional, no es necesario crearlo de nuevo. El
entorno que se conserva en esta carpeta es local; para usar el proyecto en otro
equipo, crea allí un entorno nuevo e instala las dependencias.

## Ejecución

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1
```

Abre [http://localhost:8501](http://localhost:8501) en el navegador. Mantén abierta
la terminal mientras uses la aplicación y presiona **Ctrl+C** para detenerla.
La configuración visual se carga desde `.streamlit/config.toml`.

## Uso

1. Selecciona la familia TE o TM e introduce los índices modales `m` y `n`.
2. Define las dimensiones de la guía, la frecuencia y las propiedades relativas
   del medio.
3. Selecciona el campo y la vista que quieras representar; ajusta la fase y, si
   corresponde, el plano del corte longitudinal.
4. Usa **Verificar Maxwell y fronteras** para consultar la verificación numérica
   en un punto interior.

Las dimensiones de la interfaz están en milímetros y la frecuencia en gigahercios;
los cálculos usan unidades SI. Se permite consultar modos evanescentes y modos
en el corte, además de los propagantes.

### Casos de referencia

Para `a = 40 mm`, `b = 20 mm` y `εᵣ = μᵣ = 1`:

| Modo | Frecuencia | Frecuencia de corte | Estado |
|---|---:|---:|---|
| TE(1,0) | 5 GHz | 3,747406 GHz | Propagante |
| TE(2,0) | 5 GHz | 7,494811 GHz | Evanescente |
| TM(1,1) | 10 GHz | 8,379454 GHz | Propagante |
| TM(2,1) | 15 GHz | 10,599264 GHz | Propagante |
| TM(1,1) | 5 GHz | 8,379454 GHz | Evanescente |

## Convenciones y límites

- La sección transversal es `0 ≤ x ≤ a`, `0 ≤ y ≤ b`; la propagación es en `z`.
- Se usa la convención `exp(jωt − jβz)`. Para evanescencia, `β = −jα`, con
  `α > 0`.
- TE requiere índices enteros no negativos con `m+n>0`; TM requiere `m,n≥1`.
- Las amplitudes longitudinales de referencia son `H₀=1 A/m` para TE y
  `E₀=1 V/m` para TM.
- La precisión numérica y el muestreo limitan la resolución; atiende los avisos
  de la aplicación.
- El modelo no incluye pérdidas, discontinuidades, fuentes ni acoplamiento modal.

## Archivos principales

- `app.py`: interfaz e integración.
- `modos.py`: parámetros modales y estado de propagación.
- `campos.py`: componentes de los campos.
- `validacion.py`: verificación numérica de Maxwell y las fronteras.
- `graficas.py`: generación de las visualizaciones.
- `requirements.txt` y `requirements-lock.txt`: dependencias directas y versiones
  restringidas.
- `.streamlit/config.toml`: configuración visual.
