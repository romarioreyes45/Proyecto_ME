"""
Dashboard para el proyecto final de Modelos Econométricos:
Análisis econométrico de COVID-19 en México.

Ejecutar después de correr el notebook formal completo.

Comando sugerido desde la raíz del proyecto:
    streamlit run src/dashboard_covid_modelos_econometricos.py

Si el archivo está en la raíz del proyecto:
    streamlit run dashboard_covid_modelos_econometricos.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. Configuración general
# ============================================================

st.set_page_config(
    page_title="COVID-19 México | Modelos Econométricos",
    layout="wide",
    initial_sidebar_state="expanded",
)

ETAPAS: Dict[int, str] = {
    1: "Estadística descriptiva",
    2: "Regresión lineal simple",
    3: "Regresión lineal múltiple",
    4: "Variables dummy",
    5: "Modelos Logit y Probit",
    6: "Multicolinealidad",
    7: "Heterocedasticidad",
    8: "Autocorrelación",
    9: "Normalidad de residuos",
    10: "Rezagos distribuidos",
    11: "Ajuste parcial",
    12: "Expectativas adaptativas",
    13: "Series de tiempo y ARIMA",
}

VARIABLES_SERIE = [
    "casos_diarios",
    "hospitalizaciones_diarias",
    "defunciones_diarias",
]


# ============================================================
# 2. Funciones auxiliares
# ============================================================


def detectar_raiz_proyecto() -> Path:
    """
    Detecta la carpeta raíz del proyecto con base en la existencia de
    carpetas resultados_etapa_*. Sirve tanto si el dashboard está en src/
    como si está directamente en la raíz del proyecto.
    """
    archivo_actual = Path(__file__).resolve()
    candidatos = [Path.cwd(), archivo_actual.parent]
    candidatos.extend(list(archivo_actual.parents[:4]))

    for candidato in candidatos:
        if any((candidato / f"resultados_etapa_{i}").exists() for i in range(1, 14)):
            return candidato

    return Path.cwd()


def carpeta_etapa(root: Path, etapa: int) -> Path:
    return root / f"resultados_etapa_{etapa}"


def etapas_disponibles(root: Path) -> List[int]:
    return [i for i in range(1, 14) if carpeta_etapa(root, i).exists()]


@st.cache_data(show_spinner=False)
def leer_excel(ruta: str, hoja: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Lee una hoja de Excel y regresa None si no se puede cargar."""
    try:
        return pd.read_excel(ruta, sheet_name=hoja)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def listar_hojas_excel(ruta: str) -> List[str]:
    """Obtiene los nombres de hojas de un archivo Excel."""
    try:
        return pd.ExcelFile(ruta).sheet_names
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def leer_csv(ruta: str) -> Optional[pd.DataFrame]:
    """Lee un CSV y regresa None si no se puede cargar."""
    try:
        return pd.read_csv(ruta)
    except Exception:
        return None


def normalizar_serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza la tabla de series temporales generada por el notebook.
    La Etapa 1 guarda la fecha como columna; la Etapa 13 puede guardarla
    como índice de Excel.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Si Excel guardó el índice sin nombre, suele aparecer como Unnamed: 0.
    posibles_fechas = [
        "fecha",
        "fecha_caso",
        "fecha_sintomas",
        "Unnamed: 0",
        "index",
    ]

    columna_fecha = None
    for col in posibles_fechas:
        if col in df.columns:
            columna_fecha = col
            break

    if columna_fecha is None:
        # Intento adicional: usar la primera columna si puede convertirse a fecha.
        primera_columna = df.columns[0]
        fechas = pd.to_datetime(df[primera_columna], errors="coerce")
        if fechas.notna().mean() > 0.70:
            columna_fecha = primera_columna

    if columna_fecha is not None:
        df = df.rename(columns={columna_fecha: "fecha"})
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha")

    for col in VARIABLES_SERIE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def cargar_series_temporales(root: Path) -> Tuple[pd.DataFrame, str]:
    """
    Carga la serie temporal desde la Etapa 13 si existe; si no, usa la Etapa 1.
    """
    ruta_etapa_13 = carpeta_etapa(root, 13) / "resultados_series_tiempo_arima.xlsx"
    if ruta_etapa_13.exists():
        df = leer_excel(str(ruta_etapa_13), "series_temporales")
        df = normalizar_serie_temporal(df)
        if not df.empty:
            return df, str(ruta_etapa_13)

    ruta_etapa_1 = carpeta_etapa(root, 1) / "tablas_estadistica_descriptiva.xlsx"
    if ruta_etapa_1.exists():
        df = leer_excel(str(ruta_etapa_1), "series_temporales")
        df = normalizar_serie_temporal(df)
        if not df.empty:
            return df, str(ruta_etapa_1)

    return pd.DataFrame(), ""


def formato_entero(valor: float) -> str:
    if pd.isna(valor):
        return "0"
    return f"{int(round(valor)):,}"


def mostrar_dataframe(df: Optional[pd.DataFrame], max_filas: int = 150) -> None:
    if df is None or df.empty:
        st.info("No se encontró información para esta tabla.")
        return
    st.dataframe(df.head(max_filas), use_container_width=True)


def leer_tabla_clave(root: Path, etapa: int, archivo: str, hoja: str) -> Optional[pd.DataFrame]:
    ruta = carpeta_etapa(root, etapa) / archivo
    if not ruta.exists():
        return None
    return leer_excel(str(ruta), hoja)


# ============================================================
# 3. Carga de resultados del notebook
# ============================================================

ROOT = detectar_raiz_proyecto()
ETAPAS_DISPONIBLES = etapas_disponibles(ROOT)
SERIES, FUENTE_SERIES = cargar_series_temporales(ROOT)

st.title("COVID-19 México: Dashboard econométrico")
st.caption(
    "Proyecto final de Modelos Econométricos | Resultados generados por el notebook formal"
)

with st.sidebar:
    st.header("Configuración")
    st.write("**Raíz detectada del proyecto:**")
    st.code(str(ROOT), language="text")

    if ETAPAS_DISPONIBLES:
        st.success(f"Etapas detectadas: {len(ETAPAS_DISPONIBLES)} de 13")
    else:
        st.error("No se encontraron carpetas resultados_etapa_*.")

    st.markdown("---")
    st.write("**Etapas disponibles**")
    for etapa in ETAPAS_DISPONIBLES:
        st.write(f"{etapa}. {ETAPAS[etapa]}")


if not ETAPAS_DISPONIBLES:
    st.warning(
        "Primero ejecuta el notebook completo. El dashboard necesita las carpetas "
        "resultados_etapa_1, resultados_etapa_2, ..., resultados_etapa_13."
    )
    st.stop()


# ============================================================
# 4. Indicadores generales
# ============================================================

st.subheader("Resumen general del análisis")

if SERIES.empty:
    st.warning(
        "No se encontró la tabla de series temporales. Revisa que exista "
        "resultados_etapa_1/tablas_estadistica_descriptiva.xlsx o "
        "resultados_etapa_13/resultados_series_tiempo_arima.xlsx."
    )
else:
    cols = st.columns(4)

    cols[0].metric(
        "Casos acumulados",
        formato_entero(SERIES.get("casos_diarios", pd.Series(dtype=float)).sum()),
    )
    cols[1].metric(
        "Hospitalizaciones acumuladas",
        formato_entero(SERIES.get("hospitalizaciones_diarias", pd.Series(dtype=float)).sum()),
    )
    cols[2].metric(
        "Defunciones acumuladas",
        formato_entero(SERIES.get("defunciones_diarias", pd.Series(dtype=float)).sum()),
    )

    if "fecha" in SERIES.columns and not SERIES["fecha"].isna().all():
        fecha_min = SERIES["fecha"].min().date()
        fecha_max = SERIES["fecha"].max().date()
        cols[3].metric("Periodo analizado", f"{fecha_min} a {fecha_max}")
    else:
        cols[3].metric("Periodo analizado", "No disponible")

    with st.expander("Fuente de la serie temporal"):
        st.code(FUENTE_SERIES, language="text")


# ============================================================
# 5. Pestañas principales
# ============================================================

tab_series, tab_modelos, tab_diagnosticos, tab_tablas, tab_figuras = st.tabs(
    [
        "Series temporales",
        "Resultados de modelos",
        "Diagnósticos econométricos",
        "Tablas generadas",
        "Figuras generadas",
    ]
)


# ------------------------------------------------------------
# Pestaña 1: Series temporales
# ------------------------------------------------------------

with tab_series:
    st.subheader("Evolución temporal")

    if SERIES.empty:
        st.info("No hay series temporales disponibles para graficar.")
    else:
        variables_disponibles = [v for v in VARIABLES_SERIE if v in SERIES.columns]

        if not variables_disponibles:
            st.info("La tabla cargada no contiene las variables temporales esperadas.")
        else:
            variable = st.selectbox(
                "Variable a visualizar",
                variables_disponibles,
                format_func=lambda x: x.replace("_", " ").capitalize(),
            )

            fig = px.line(
                SERIES,
                x="fecha" if "fecha" in SERIES.columns else SERIES.index,
                y=variable,
                title=f"Serie diaria: {variable.replace('_', ' ')}",
                labels={variable: variable.replace("_", " "), "fecha": "Fecha"},
            )
            st.plotly_chart(fig, use_container_width=True)

            if len(variables_disponibles) >= 2:
                st.markdown("#### Relación entre variables diarias")
                col_x, col_y = st.columns(2)
                x_var = col_x.selectbox("Variable eje X", variables_disponibles, index=0)
                y_default = 2 if len(variables_disponibles) > 2 else 1
                y_var = col_y.selectbox("Variable eje Y", variables_disponibles, index=y_default)

                fig_scatter = px.scatter(
                    SERIES,
                    x=x_var,
                    y=y_var,
                    opacity=0.55,
                    title=f"{y_var.replace('_', ' ')} vs {x_var.replace('_', ' ')}",
                    labels={
                        x_var: x_var.replace("_", " "),
                        y_var: y_var.replace("_", " "),
                    },
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("Vista previa de la serie temporal"):
            mostrar_dataframe(SERIES)


# ------------------------------------------------------------
# Pestaña 2: Resultados de modelos
# ------------------------------------------------------------

with tab_modelos:
    st.subheader("Resumen de modelos estimados")

    st.markdown("#### Regresión lineal simple")
    metricas_rls = leer_tabla_clave(
        ROOT,
        2,
        "resultados_regresion_lineal_simple.xlsx",
        "metricas_modelo",
    )
    mostrar_dataframe(metricas_rls)

    st.markdown("#### Regresión lineal múltiple")
    metricas_rlm = leer_tabla_clave(
        ROOT,
        3,
        "resultados_regresion_lineal_multiple.xlsx",
        "metricas_modelo",
    )
    mostrar_dataframe(metricas_rlm)

    st.markdown("#### Logit y Probit")
    comparacion_lp = leer_tabla_clave(
        ROOT,
        5,
        "resultados_logit_probit.xlsx",
        "comparacion_modelos",
    )
    mostrar_dataframe(comparacion_lp)

    st.markdown("#### Rezagos distribuidos")
    metricas_rezagos = leer_tabla_clave(
        ROOT,
        10,
        "resultados_modelos_rezagos_distribuidos.xlsx",
        "metricas_modelos",
    )
    mostrar_dataframe(metricas_rezagos)

    st.markdown("#### Ajuste parcial")
    metricas_ajuste = leer_tabla_clave(
        ROOT,
        11,
        "resultados_modelo_ajuste_parcial.xlsx",
        "metricas_modelos",
    )
    mostrar_dataframe(metricas_ajuste)

    st.markdown("#### Expectativas adaptativas")
    metricas_exp = leer_tabla_clave(
        ROOT,
        12,
        "resultados_expectativas_adaptativas.xlsx",
        "metricas_modelos",
    )
    mostrar_dataframe(metricas_exp)

    st.markdown("#### Comparación AR, MA, ARMA y ARIMA")
    comparacion_arima = leer_tabla_clave(
        ROOT,
        13,
        "resultados_series_tiempo_arima.xlsx",
        "comparacion_modelos",
    )
    mostrar_dataframe(comparacion_arima)

    st.markdown("#### Pronósticos 7, 14 y 30 días")
    pronosticos = leer_tabla_clave(
        ROOT,
        13,
        "resultados_series_tiempo_arima.xlsx",
        "pronosticos_7_14_30",
    )
    mostrar_dataframe(pronosticos)


# ------------------------------------------------------------
# Pestaña 3: Diagnósticos econométricos
# ------------------------------------------------------------

with tab_diagnosticos:
    st.subheader("Pruebas y diagnósticos")

    st.markdown("#### Multicolinealidad")
    vif_clinico = leer_tabla_clave(
        ROOT,
        6,
        "resultados_multicolinealidad.xlsx",
        "vif_clinico",
    )
    mostrar_dataframe(vif_clinico)

    st.markdown("#### Heterocedasticidad")
    hetero = leer_tabla_clave(
        ROOT,
        7,
        "resultados_heterocedasticidad.xlsx",
        "pruebas_hetero",
    )
    mostrar_dataframe(hetero)

    st.markdown("#### Autocorrelación")
    auto = leer_tabla_clave(
        ROOT,
        8,
        "resultados_autocorrelacion.xlsx",
        "pruebas_autocorrelacion",
    )
    mostrar_dataframe(auto)

    st.markdown("#### Normalidad de residuos")
    normalidad = leer_tabla_clave(
        ROOT,
        9,
        "resultados_normalidad.xlsx",
        "pruebas_normalidad",
    )
    mostrar_dataframe(normalidad)

    st.markdown("#### Prueba ADF sobre series originales")
    adf_original = leer_tabla_clave(
        ROOT,
        13,
        "resultados_series_tiempo_arima.xlsx",
        "adf_original",
    )
    mostrar_dataframe(adf_original)

    st.markdown("#### Prueba ADF sobre primeras diferencias")
    adf_diff = leer_tabla_clave(
        ROOT,
        13,
        "resultados_series_tiempo_arima.xlsx",
        "adf_diferenciada",
    )
    mostrar_dataframe(adf_diff)


# ------------------------------------------------------------
# Pestaña 4: Tablas generadas
# ------------------------------------------------------------

with tab_tablas:
    st.subheader("Explorador de archivos de resultados")

    etapa_seleccionada = st.selectbox(
        "Selecciona una etapa",
        ETAPAS_DISPONIBLES,
        format_func=lambda e: f"Etapa {e}: {ETAPAS[e]}",
    )

    carpeta = carpeta_etapa(ROOT, etapa_seleccionada)
    archivos_excel = sorted(carpeta.glob("*.xlsx"))
    archivos_csv = sorted(carpeta.glob("*.csv"))
    archivos_txt = sorted(carpeta.glob("*.txt"))

    if not archivos_excel and not archivos_csv and not archivos_txt:
        st.info("Esta etapa no tiene tablas o archivos de texto disponibles.")

    for archivo in archivos_excel:
        with st.expander(f"{archivo.name}"):
            hojas = listar_hojas_excel(str(archivo))
            if not hojas:
                st.warning("No se pudieron leer las hojas del archivo.")
                continue
            hoja = st.selectbox(
                f"Hoja de {archivo.name}",
                hojas,
                key=f"hoja_{etapa_seleccionada}_{archivo.name}",
            )
            df_hoja = leer_excel(str(archivo), hoja)
            mostrar_dataframe(df_hoja)

    for archivo in archivos_csv:
        with st.expander(f"{archivo.name}"):
            mostrar_dataframe(leer_csv(str(archivo)))

    for archivo in archivos_txt:
        with st.expander(f"{archivo.name}"):
            try:
                texto = archivo.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                texto = archivo.read_text(encoding="latin-1")
            st.text_area(
                "Contenido",
                texto,
                height=300,
                key=f"txt_{etapa_seleccionada}_{archivo.name}",
            )


# ------------------------------------------------------------
# Pestaña 5: Figuras generadas
# ------------------------------------------------------------

with tab_figuras:
    st.subheader("Galería de gráficas")

    etapa_figuras = st.selectbox(
        "Selecciona la etapa para visualizar figuras",
        ETAPAS_DISPONIBLES,
        format_func=lambda e: f"Etapa {e}: {ETAPAS[e]}",
        key="etapa_figuras",
    )

    carpeta_fig = carpeta_etapa(ROOT, etapa_figuras)
    imagenes = sorted(list(carpeta_fig.glob("*.png")) + list(carpeta_fig.glob("*.jpg")))

    if not imagenes:
        st.info("No se encontraron imágenes en esta etapa.")
    else:
        columnas = st.columns(2)
        for i, imagen in enumerate(imagenes):
            with columnas[i % 2]:
                st.image(str(imagen), caption=imagen.name, use_container_width=True)


# ============================================================
# 6. Nota final
# ============================================================

st.markdown("---")
st.caption(
    "Este dashboard resume los archivos generados por el notebook. "
    "Si cambias nombres de carpetas, hojas o archivos en el notebook, actualiza las rutas correspondientes en este script."
)
