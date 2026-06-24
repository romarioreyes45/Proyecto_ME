"""
Estadística descriptiva
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **Etapa 1: Estadística descriptiva**
######################################################################



# %% [notebook cell 3]
# ============================================================
# ETAPA 1. ESTADÍSTICA DESCRIPTIVA
# Proyecto: Modelos Econométricos - COVID-19 en México
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import warnings
from pathlib import Path
from IPython.display import display

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Configuración general
# ------------------------------------------------------------
# Coloca el archivo oficial en data/raw/.
# Puede ser .csv o .zip. Si cambias el nombre, actualiza DATA_PATH.
DATA_PATH = Path("data/raw/COVID19MEXICO.csv")

# Si DATA_PATH no existe, se intenta buscar un CSV o ZIP en data/raw/.
if not DATA_PATH.exists():
    raw_dir = Path("data/raw")
    candidatos = []
    if raw_dir.exists():
        candidatos = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.zip"))

    if len(candidatos) == 1:
        DATA_PATH = candidatos[0]
        print(f"Archivo identificado en data/raw: {DATA_PATH}")
    else:
        raise FileNotFoundError(
            "No se encontró el archivo de datos. Coloca el CSV/ZIP oficial en data/raw/ "
            "o actualiza la variable DATA_PATH."
        )

# Si tu equipo se satura, usa por ejemplo SAMPLE_MAX = 200000.
# Para usar toda la base, deja SAMPLE_MAX = None.
SAMPLE_MAX = None
RANDOM_STATE = 42

# Códigos metodológicos usados para identificar casos COVID.
# IMPORTANTE: valida estos códigos con el diccionario oficial de la base que descargues.
CODIGOS_CONFIRMADOS_CLASIFICACION = [1, 2, 3]
CODIGO_PCR_SARS_COV_2 = 34

# Carpeta para guardar gráficas y tablas de la etapa 1
OUT_DIR = Path("resultados_etapa_1")
OUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Funciones para cargar el archivo
# ------------------------------------------------------------

def obtener_csv_desde_zip(path):
    """
    Si el archivo es .zip, identifica el primer CSV dentro del comprimido.
    """
    with zipfile.ZipFile(path, "r") as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No se encontró ningún archivo CSV dentro del ZIP.")
        return csv_files[0]


def leer_encabezado(path):
    """
    Lee únicamente los nombres de columnas del archivo.
    """
    if str(path).lower().endswith(".zip"):
        csv_name = obtener_csv_desde_zip(path)
        with zipfile.ZipFile(path, "r") as z:
            with z.open(csv_name) as f:
                header = pd.read_csv(f, nrows=0)
    else:
        header = pd.read_csv(path, nrows=0)
    
    header.columns = header.columns.str.upper().str.strip()
    return header.columns.tolist()


def leer_datos_covid(path, columnas_necesarias):
    """
    Lee únicamente las columnas necesarias para no cargar toda la base.
    """
    columnas_disponibles = leer_encabezado(path)
    columnas_a_leer = [c for c in columnas_necesarias if c in columnas_disponibles]
    
    print("Columnas que sí se encontraron:")
    print(columnas_a_leer)
    
    faltantes = [c for c in columnas_necesarias if c not in columnas_disponibles]
    if faltantes:
        print("\nColumnas que NO se encontraron en la base:")
        print(faltantes)
    
    if str(path).lower().endswith(".zip"):
        csv_name = obtener_csv_desde_zip(path)
        with zipfile.ZipFile(path, "r") as z:
            with z.open(csv_name) as f:
                df = pd.read_csv(f, usecols=columnas_a_leer, dtype=str, low_memory=False)
    else:
        df = pd.read_csv(path, usecols=columnas_a_leer, dtype=str, low_memory=False)
    
    df.columns = df.columns.str.upper().str.strip()
    return df


# ------------------------------------------------------------
# 3. Columnas necesarias para esta etapa
# ------------------------------------------------------------

columnas_necesarias = [
    "FECHA_INGRESO",
    "FECHA_SINTOMAS",
    "FECHA_DEF",
    "SEXO",
    "TIPO_PACIENTE",
    "EDAD",
    "DIABETES",
    "HIPERTENSION",
    "OBESIDAD",
    "NEUMONIA",
    "INMUSUPR",
    "UCI",
    "ENTIDAD_RES",

    # Variables nuevas / actuales
    "CLASIFICACION_FINAL_COVID",
    "RESULTADO_PCR",
    "RESULTADO_PCR_COINFECCION",

    # Variables históricas, por si se usa una base anterior
    "CLASIFICACION_FINAL",
    "RESULTADO_LAB"
]

df = leer_datos_covid(DATA_PATH, columnas_necesarias)

print("\nDimensiones de la base cargada:")
print(df.shape)

display(df.head())



# %% [notebook cell 4]
# ------------------------------------------------------------
# 4. Limpieza y transformación de variables
# ------------------------------------------------------------

# Convertir nombres a minúsculas para trabajar más cómodo
df.columns = df.columns.str.lower().str.strip()

# Convertir edad a numérica
if "edad" in df.columns:
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
else:
    raise ValueError("No se encontró la columna EDAD; es necesaria para el análisis descriptivo y los modelos.")

# Fechas
for col in ["fecha_ingreso", "fecha_sintomas", "fecha_def"]:
    if col in df.columns:
        df[col] = df[col].replace("9999-99-99", np.nan)
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Fecha principal del caso:
# Usamos fecha de síntomas si existe; si no, fecha de ingreso
if "fecha_sintomas" in df.columns and "fecha_ingreso" in df.columns:
    df["fecha_caso"] = df["fecha_sintomas"].fillna(df["fecha_ingreso"])
elif "fecha_ingreso" in df.columns:
    df["fecha_caso"] = df["fecha_ingreso"]
else:
    raise ValueError("No se encontró una fecha válida para construir la serie temporal.")

# Función para transformar variables Sí/No
# En la base oficial:
# 1 = Sí
# 2 = No
# 97, 98, 99 = No especificado / No aplica / Se ignora
def convertir_si_no(serie):
    serie_num = pd.to_numeric(serie, errors="coerce")
    return serie_num.map({1: 1, 2: 0})

# Sexo:
# 1 = Mujer, 2 = Hombre
if "sexo" in df.columns:
    df["sexo_masculino"] = pd.to_numeric(df["sexo"], errors="coerce").map({2: 1, 1: 0})

# Tipo de paciente:
# 1 = Ambulatorio, 2 = Hospitalizado
if "tipo_paciente" in df.columns:
    df["hospitalizado"] = pd.to_numeric(df["tipo_paciente"], errors="coerce").map({2: 1, 1: 0})

# Defunción:
# Si hay fecha de defunción válida, entonces defuncion = 1
if "fecha_def" in df.columns:
    df["defuncion"] = np.where(df["fecha_def"].notna(), 1, 0)

# Comorbilidades y condiciones clínicas
vars_binarias_originales = [
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmusupr",
    "uci"
]

for col in vars_binarias_originales:
    if col in df.columns:
        df[col] = convertir_si_no(df[col])

# Renombrar inmunosupresión para que sea más claro
if "inmusupr" in df.columns:
    df = df.rename(columns={"inmusupr": "inmunosupresion"})

# Número de comorbilidades principales
comorbilidades = ["diabetes", "hipertension", "obesidad", "neumonia", "inmunosupresion"]
comorbilidades_existentes = [c for c in comorbilidades if c in df.columns]

df["num_comorbilidades"] = df[comorbilidades_existentes].sum(axis=1, skipna=True)

display(df.head())


# ------------------------------------------------------------
# 5. Filtrar casos confirmados de COVID-19
# ------------------------------------------------------------

df["caso_confirmado"] = False

# ------------------------------------------------------------
# Criterio 1: Clasificación final COVID
# ------------------------------------------------------------

df["covid_por_clasificacion"] = False

if "clasificacion_final_covid" in df.columns:
    clasif_covid = pd.to_numeric(df["clasificacion_final_covid"], errors="coerce")
    
    # Generalmente 1, 2 y 3 corresponden a casos confirmados de COVID-19
    # según el catálogo de clasificación final COVID.
    df["covid_por_clasificacion"] = clasif_covid.isin(CODIGOS_CONFIRMADOS_CLASIFICACION)

elif "clasificacion_final" in df.columns:
    clasif = pd.to_numeric(df["clasificacion_final"], errors="coerce")
    df["covid_por_clasificacion"] = clasif.isin(CODIGOS_CONFIRMADOS_CLASIFICACION)


# ------------------------------------------------------------
# Criterio 2: Resultado PCR
# ------------------------------------------------------------

df["covid_por_pcr"] = False

if "resultado_pcr" in df.columns:
    resultado_pcr = pd.to_numeric(df["resultado_pcr"], errors="coerce")
    
    # Según el catálogo PCR:
    # 34 = SARS-CoV-2
    df["covid_por_pcr"] = resultado_pcr.eq(CODIGO_PCR_SARS_COV_2)

if "resultado_pcr_coinfeccion" in df.columns:
    resultado_pcr_coinf = pd.to_numeric(df["resultado_pcr_coinfeccion"], errors="coerce")
    
    # También se considera COVID si SARS-CoV-2 aparece como coinfección.
    df["covid_por_pcr"] = df["covid_por_pcr"] | resultado_pcr_coinf.eq(CODIGO_PCR_SARS_COV_2)


# ------------------------------------------------------------
# Criterio final
# ------------------------------------------------------------

# Recomendación metodológica:
# usar CLASIFICACION_FINAL_COVID como criterio principal.
# Si no existe clasificación final, usar PCR.
if "clasificacion_final_covid" in df.columns or "clasificacion_final" in df.columns:
    df["caso_confirmado"] = df["covid_por_clasificacion"]
else:
    df["caso_confirmado"] = df["covid_por_pcr"]


df_covid = df[df["caso_confirmado"] == True].copy()

# Muestreo opcional para equipos con poca memoria.
# Se conserva la aleatoriedad controlada para que los resultados sean reproducibles.
if SAMPLE_MAX is not None and len(df_covid) > SAMPLE_MAX:
    df_covid = df_covid.sample(n=SAMPLE_MAX, random_state=RANDOM_STATE).copy()
    print(f"Se aplicó muestreo reproducible. Registros usados: {len(df_covid)}")


print("Registros totales cargados:", len(df))
print("Casos COVID por clasificación final:", df["covid_por_clasificacion"].sum())
print("Casos COVID por PCR:", df["covid_por_pcr"].sum())
print("Registros considerados como casos confirmados de COVID-19:", len(df_covid))

display(df_covid.head())



# %% [notebook cell 5]
# ------------------------------------------------------------
# Revisión de variables usadas para identificar COVID-19
# ------------------------------------------------------------

if "clasificacion_final_covid" in df.columns:
    print("Distribución de CLASIFICACION_FINAL_COVID:")
    print(
        pd.to_numeric(df["clasificacion_final_covid"], errors="coerce")
        .value_counts(dropna=False)
        .sort_index()
    )

if "resultado_pcr" in df.columns:
    print("\nDistribución de RESULTADO_PCR:")
    print(
        pd.to_numeric(df["resultado_pcr"], errors="coerce")
        .value_counts(dropna=False)
        .sort_index()
    )

if "resultado_pcr_coinfeccion" in df.columns:
    print("\nDistribución de RESULTADO_PCR_COINFECCION:")
    print(
        pd.to_numeric(df["resultado_pcr_coinfeccion"], errors="coerce")
        .value_counts(dropna=False)
        .sort_index()
    )


# ------------------------------------------------------------
# 6. Función para calcular estadística descriptiva
# ------------------------------------------------------------

def moda_segura(serie):
    """
    Calcula la moda evitando errores cuando no hay moda.
    """
    moda = serie.dropna().mode()
    if len(moda) == 0:
        return np.nan
    return moda.iloc[0]


def resumen_descriptivo(df_base, variables):
    """
    Calcula medidas de tendencia central, dispersión y forma.
    """
    resultados = []

    for var in variables:
        serie = pd.to_numeric(df_base[var], errors="coerce").dropna()
        
        if len(serie) == 0:
            continue
        
        media = serie.mean()
        desv = serie.std()
        
        resultados.append({
            "variable": var,
            "n": serie.count(),
            "media": media,
            "mediana": serie.median(),
            "moda": moda_segura(serie),
            "varianza": serie.var(),
            "desviacion_estandar": desv,
            "coeficiente_variacion": desv / media if media != 0 else np.nan,
            "asimetria": serie.skew(),
            "curtosis": serie.kurtosis(),
            "minimo": serie.min(),
            "maximo": serie.max()
        })

    return pd.DataFrame(resultados)


# ------------------------------------------------------------
# 7. Estadística descriptiva individual
# ------------------------------------------------------------

variables_individuales = [
    "edad",
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion",
    "num_comorbilidades",
    "hospitalizado",
    "defuncion"
]

variables_individuales = [v for v in variables_individuales if v in df_covid.columns]

resumen_individual = resumen_descriptivo(df_covid, variables_individuales)

print("Resumen descriptivo de variables individuales")
display(resumen_individual)


# ------------------------------------------------------------
# 8. Tabla de proporciones para variables dummy
# ------------------------------------------------------------

variables_dummy = [
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion",
    "hospitalizado",
    "defuncion"
]

variables_dummy = [v for v in variables_dummy if v in df_covid.columns]

tabla_dummies = []

for var in variables_dummy:
    conteos = df_covid[var].value_counts(dropna=False)
    porcentajes = df_covid[var].value_counts(normalize=True, dropna=False) * 100
    
    tabla_dummies.append({
        "variable": var,
        "conteo_0": conteos.get(0, 0),
        "conteo_1": conteos.get(1, 0),
        "porcentaje_1": porcentajes.get(1, 0),
        "valores_perdidos": df_covid[var].isna().sum()
    })

tabla_dummies = pd.DataFrame(tabla_dummies)

print("Tabla de proporciones de variables binarias")
display(tabla_dummies)



# %% [notebook cell 6]
# ------------------------------------------------------------
# 9. Construcción de series temporales
# ------------------------------------------------------------

# Casos diarios
casos_diarios = (
    df_covid
    .dropna(subset=["fecha_caso"])
    .groupby("fecha_caso")
    .size()
    .rename("casos_diarios")
)

# Hospitalizaciones diarias
if "hospitalizado" in df_covid.columns:
    hospitalizaciones_diarias = (
        df_covid[df_covid["hospitalizado"] == 1]
        .dropna(subset=["fecha_caso"])
        .groupby("fecha_caso")
        .size()
        .rename("hospitalizaciones_diarias")
    )
else:
    hospitalizaciones_diarias = pd.Series(dtype="float64", name="hospitalizaciones_diarias")

# Defunciones diarias
if "fecha_def" in df_covid.columns:
    defunciones_diarias = (
        df_covid[df_covid["defuncion"] == 1]
        .dropna(subset=["fecha_def"])
        .groupby("fecha_def")
        .size()
        .rename("defunciones_diarias")
    )
else:
    defunciones_diarias = pd.Series(dtype="float64", name="defunciones_diarias")

# Unir series
fecha_min = min(
    casos_diarios.index.min(),
    hospitalizaciones_diarias.index.min() if len(hospitalizaciones_diarias) > 0 else casos_diarios.index.min(),
    defunciones_diarias.index.min() if len(defunciones_diarias) > 0 else casos_diarios.index.min()
)

fecha_max = max(
    casos_diarios.index.max(),
    hospitalizaciones_diarias.index.max() if len(hospitalizaciones_diarias) > 0 else casos_diarios.index.max(),
    defunciones_diarias.index.max() if len(defunciones_diarias) > 0 else casos_diarios.index.max()
)

fechas = pd.date_range(start=fecha_min, end=fecha_max, freq="D")

df_ts = pd.DataFrame(index=fechas)
df_ts = df_ts.join(casos_diarios)
df_ts = df_ts.join(hospitalizaciones_diarias)
df_ts = df_ts.join(defunciones_diarias)

df_ts = df_ts.fillna(0).reset_index().rename(columns={"index": "fecha"})

print("Serie temporal construida")
display(df_ts.head())

print("Últimos registros")
display(df_ts.tail())


# ------------------------------------------------------------
# 10. Estadística descriptiva de series temporales
# ------------------------------------------------------------

variables_temporales = [
    "casos_diarios",
    "hospitalizaciones_diarias",
    "defunciones_diarias"
]

resumen_temporal = resumen_descriptivo(df_ts, variables_temporales)

print("Resumen descriptivo de series temporales")
display(resumen_temporal)


# ------------------------------------------------------------
# 11. Histogramas obligatorios
# ------------------------------------------------------------

def guardar_histograma(data, variable, titulo, xlabel, bins=30):
    plt.figure(figsize=(8, 5))
    plt.hist(data[variable].dropna(), bins=bins)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"histograma_{variable}.png", dpi=300)
    plt.show()


# Histograma 1: Edad
guardar_histograma(
    df_covid,
    "edad",
    "Distribución de la edad de los pacientes",
    "Edad",
    bins=30
)

# Histograma 2: Número de comorbilidades
guardar_histograma(
    df_covid,
    "num_comorbilidades",
    "Distribución del número de comorbilidades",
    "Número de comorbilidades",
    bins=10
)

# Histograma 3: Casos diarios
guardar_histograma(
    df_ts,
    "casos_diarios",
    "Distribución de casos diarios",
    "Casos diarios",
    bins=30
)

# Histograma 4: Hospitalizaciones diarias
guardar_histograma(
    df_ts,
    "hospitalizaciones_diarias",
    "Distribución de hospitalizaciones diarias",
    "Hospitalizaciones diarias",
    bins=30
)

# Histograma 5: Defunciones diarias
guardar_histograma(
    df_ts,
    "defunciones_diarias",
    "Distribución de defunciones diarias",
    "Defunciones diarias",
    bins=30
)



# %% [notebook cell 7]
# ------------------------------------------------------------
# 12. Diagramas de caja obligatorios
# ------------------------------------------------------------

def boxplot_por_grupo(df_base, variable_numerica, variable_grupo, titulo, etiquetas):
    data_plot = df_base[[variable_numerica, variable_grupo]].dropna()
    
    grupos = [
        data_plot[data_plot[variable_grupo] == 0][variable_numerica],
        data_plot[data_plot[variable_grupo] == 1][variable_numerica]
    ]
    
    plt.figure(figsize=(8, 5))
    plt.boxplot(grupos, labels=etiquetas)
    plt.title(titulo)
    plt.ylabel(variable_numerica.capitalize())
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"boxplot_{variable_numerica}_por_{variable_grupo}.png", dpi=300)
    plt.show()


# Boxplot 1: Edad por defunción
boxplot_por_grupo(
    df_covid,
    "edad",
    "defuncion",
    "Distribución de edad según condición de defunción",
    ["No falleció", "Falleció"]
)

# Boxplot 2: Edad por hospitalización
boxplot_por_grupo(
    df_covid,
    "edad",
    "hospitalizado",
    "Distribución de edad según tipo de atención",
    ["Ambulatorio", "Hospitalizado"]
)


# ------------------------------------------------------------
# 13. Diagramas de dispersión obligatorios
# ------------------------------------------------------------

def guardar_dispersion(df_base, x, y, titulo, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.scatter(df_base[x], df_base[y], alpha=0.6)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"dispersion_{x}_vs_{y}.png", dpi=300)
    plt.show()


# Dispersión 1: Casos diarios vs defunciones diarias
guardar_dispersion(
    df_ts,
    "casos_diarios",
    "defunciones_diarias",
    "Relación entre casos diarios y defunciones diarias",
    "Casos diarios",
    "Defunciones diarias"
)

# Dispersión 2: Hospitalizaciones diarias vs defunciones diarias
guardar_dispersion(
    df_ts,
    "hospitalizaciones_diarias",
    "defunciones_diarias",
    "Relación entre hospitalizaciones diarias y defunciones diarias",
    "Hospitalizaciones diarias",
    "Defunciones diarias"
)


# ------------------------------------------------------------
# 14. Matriz de correlación
# ------------------------------------------------------------

variables_correlacion = [
    "edad",
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion",
    "num_comorbilidades",
    "hospitalizado",
    "defuncion"
]

variables_correlacion = [v for v in variables_correlacion if v in df_covid.columns]

matriz_corr = df_covid[variables_correlacion].corr()

print("Matriz de correlación")
display(matriz_corr)

plt.figure(figsize=(10, 8))
plt.imshow(matriz_corr, aspect="auto")
plt.colorbar(label="Correlación")
plt.xticks(range(len(matriz_corr.columns)), matriz_corr.columns, rotation=90)
plt.yticks(range(len(matriz_corr.index)), matriz_corr.index)
plt.title("Matriz de correlación de variables principales")
plt.tight_layout()
plt.savefig(OUT_DIR / "matriz_correlacion.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 15. Guardar tablas de resultados
# ------------------------------------------------------------

ruta_excel = OUT_DIR / "tablas_estadistica_descriptiva.xlsx"

with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
    resumen_individual.to_excel(writer, sheet_name="resumen_individual", index=False)
    tabla_dummies.to_excel(writer, sheet_name="variables_dummy", index=False)
    resumen_temporal.to_excel(writer, sheet_name="resumen_temporal", index=False)
    matriz_corr.to_excel(writer, sheet_name="matriz_correlacion")
    df_ts.to_excel(writer, sheet_name="series_temporales", index=False)

print(f"Tablas guardadas en: {ruta_excel}")
print(f"Gráficas guardadas en la carpeta: {OUT_DIR}")
