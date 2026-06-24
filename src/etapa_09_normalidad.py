"""
NORMALIDAD
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 9. NORMALIDAD**
######################################################################



# %% [notebook cell 36]
# ============================================================
# ETAPA 9. NORMALIDAD
# Pruebas de Jarque-Bera, Shapiro-Wilk y análisis de residuos
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display
from scipy.stats import jarque_bera, shapiro, skew, kurtosis, probplot

# Carpeta de salida para esta etapa
OUT_DIR_9 = Path("resultados_etapa_9")
OUT_DIR_9.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Modelo de regresión lineal simple
# Defunciones diarias ~ Hospitalizaciones diarias
# ------------------------------------------------------------

df_rls_norm = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias"
]].copy()

for col in df_rls_norm.columns:
    df_rls_norm[col] = pd.to_numeric(df_rls_norm[col], errors="coerce")

df_rls_norm = df_rls_norm.dropna()

Y_rls_norm = df_rls_norm["defunciones_diarias"]
X_rls_norm = df_rls_norm[["hospitalizaciones_diarias"]]
X_rls_norm = sm.add_constant(X_rls_norm)

modelo_rls_norm = sm.OLS(Y_rls_norm, X_rls_norm).fit()

print(modelo_rls_norm.summary())


# ------------------------------------------------------------
# 2. Modelo de regresión lineal múltiple
# Defunciones diarias ~ Hospitalizaciones diarias + Casos diarios
# ------------------------------------------------------------

df_rlm_norm = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias",
    "casos_diarios"
]].copy()

for col in df_rlm_norm.columns:
    df_rlm_norm[col] = pd.to_numeric(df_rlm_norm[col], errors="coerce")

df_rlm_norm = df_rlm_norm.dropna()

Y_rlm_norm = df_rlm_norm["defunciones_diarias"]
X_rlm_norm = df_rlm_norm[[
    "hospitalizaciones_diarias",
    "casos_diarios"
]]
X_rlm_norm = sm.add_constant(X_rlm_norm)

modelo_rlm_norm = sm.OLS(Y_rlm_norm, X_rlm_norm).fit()

print(modelo_rlm_norm.summary())


# ------------------------------------------------------------
# 3. Funciones auxiliares
# ------------------------------------------------------------

def pruebas_normalidad(residuos, nombre_modelo):
    """
    Aplica Jarque-Bera y Shapiro-Wilk a los residuos.
    """
    residuos = pd.Series(residuos).dropna()
    
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(residuos)
    
    # Shapiro-Wilk
    # Nota: Shapiro puede ser muy sensible con muestras grandes.
    # Si n > 5000, se toma una muestra de 5000 para evitar advertencias.
    if len(residuos) > 5000:
        residuos_shapiro = residuos.sample(5000, random_state=42)
    else:
        residuos_shapiro = residuos
    
    shapiro_stat, shapiro_p = shapiro(residuos_shapiro)
    
    resultados = pd.DataFrame({
        "modelo": [nombre_modelo, nombre_modelo],
        "prueba": ["Jarque-Bera", "Shapiro-Wilk"],
        "estadistico": [jb_stat, shapiro_stat],
        "p_valor": [jb_p, shapiro_p],
        "n_residuos": [len(residuos), len(residuos_shapiro)],
        "decision_5": [
            "Rechazar H0: residuos no normales" if jb_p < 0.05 else "No rechazar H0: no hay evidencia suficiente contra normalidad",
            "Rechazar H0: residuos no normales" if shapiro_p < 0.05 else "No rechazar H0: no hay evidencia suficiente contra normalidad"
        ]
    })
    
    return resultados


def estadisticos_residuos(residuos, nombre_modelo):
    """
    Calcula estadísticos descriptivos de los residuos.
    """
    residuos = pd.Series(residuos).dropna()
    
    tabla = pd.DataFrame({
        "modelo": [nombre_modelo],
        "n": [len(residuos)],
        "media": [residuos.mean()],
        "mediana": [residuos.median()],
        "desviacion_estandar": [residuos.std()],
        "minimo": [residuos.min()],
        "maximo": [residuos.max()],
        "asimetria": [skew(residuos)],
        "curtosis": [kurtosis(residuos)]
    })
    
    return tabla


def conclusion_normalidad(tabla_resultados, nombre_modelo):
    """
    Genera una conclusión sobre normalidad.
    """
    sub = tabla_resultados[tabla_resultados["modelo"] == nombre_modelo]
    
    rechazos = (sub["p_valor"] < 0.05).sum()
    
    if rechazos == 0:
        return (
            f"En el modelo {nombre_modelo}, las pruebas aplicadas no rechazan la hipótesis "
            f"de normalidad al 5% de significancia. Por lo tanto, no se encuentra evidencia "
            f"estadística suficiente para afirmar que los residuos se alejan de una distribución normal."
        )
    elif rechazos == len(sub):
        return (
            f"En el modelo {nombre_modelo}, las pruebas Jarque-Bera y Shapiro-Wilk rechazan "
            f"la hipótesis de normalidad al 5% de significancia. Esto sugiere que los residuos "
            f"no siguen una distribución normal."
        )
    else:
        return (
            f"En el modelo {nombre_modelo}, los resultados de normalidad son mixtos. "
            f"Al menos una prueba rechaza la hipótesis de normalidad, por lo que la distribución "
            f"de los residuos debe interpretarse con cautela y complementarse con el análisis gráfico."
        )



# %% [notebook cell 37]
# ------------------------------------------------------------
# 4. Extraer residuos
# ------------------------------------------------------------

residuos_rls_norm = modelo_rls_norm.resid
residuos_rlm_norm = modelo_rlm_norm.resid


# ------------------------------------------------------------
# 5. Pruebas de normalidad
# ------------------------------------------------------------

normalidad_rls = pruebas_normalidad(
    residuos_rls_norm,
    "Regresión lineal simple"
)

normalidad_rlm = pruebas_normalidad(
    residuos_rlm_norm,
    "Regresión lineal múltiple"
)

resultados_normalidad = pd.concat(
    [normalidad_rls, normalidad_rlm],
    ignore_index=True
)

print("Resultados de pruebas de normalidad")
display(resultados_normalidad)


# ------------------------------------------------------------
# 6. Estadísticos descriptivos de residuos
# ------------------------------------------------------------

estad_res_rls = estadisticos_residuos(
    residuos_rls_norm,
    "Regresión lineal simple"
)

estad_res_rlm = estadisticos_residuos(
    residuos_rlm_norm,
    "Regresión lineal múltiple"
)

estadisticos_residuos_norm = pd.concat(
    [estad_res_rls, estad_res_rlm],
    ignore_index=True
)

print("Estadísticos descriptivos de residuos")
display(estadisticos_residuos_norm)


# ------------------------------------------------------------
# 7. Histograma residuos RLS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(residuos_rls_norm, bins=30)
plt.title("Histograma de residuos - Regresión lineal simple")
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_hist_rls = OUT_DIR_9 / "normalidad_histograma_residuos_rls.png"
plt.savefig(ruta_hist_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_hist_rls}")


# ------------------------------------------------------------
# 8. Histograma residuos RLM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(residuos_rlm_norm, bins=30)
plt.title("Histograma de residuos - Regresión lineal múltiple")
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_hist_rlm = OUT_DIR_9 / "normalidad_histograma_residuos_rlm.png"
plt.savefig(ruta_hist_rlm, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_hist_rlm}")


# ------------------------------------------------------------
# 9. Q-Q plot residuos RLS
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))
probplot(residuos_rls_norm, dist="norm", plot=plt)
plt.title("Q-Q plot de residuos - Regresión lineal simple")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_qq_rls = OUT_DIR_9 / "normalidad_qqplot_residuos_rls.png"
plt.savefig(ruta_qq_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_qq_rls}")


# ------------------------------------------------------------
# 10. Q-Q plot residuos RLM
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))
probplot(residuos_rlm_norm, dist="norm", plot=plt)
plt.title("Q-Q plot de residuos - Regresión lineal múltiple")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_qq_rlm = OUT_DIR_9 / "normalidad_qqplot_residuos_rlm.png"
plt.savefig(ruta_qq_rlm, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_qq_rlm}")


# ------------------------------------------------------------
# 11. Conclusiones del diagnóstico
# ------------------------------------------------------------

conclusion_norm_rls = conclusion_normalidad(
    resultados_normalidad,
    "Regresión lineal simple"
)

conclusion_norm_rlm = conclusion_normalidad(
    resultados_normalidad,
    "Regresión lineal múltiple"
)

print(conclusion_norm_rls)
print(conclusion_norm_rlm)



# %% [notebook cell 38]
# ------------------------------------------------------------
# 12. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_normalidad = OUT_DIR_9 / "resultados_normalidad.xlsx"

with pd.ExcelWriter(ruta_excel_normalidad, engine="openpyxl") as writer:
    resultados_normalidad.to_excel(
        writer,
        sheet_name="pruebas_normalidad",
        index=False
    )
    
    estadisticos_residuos_norm.to_excel(
        writer,
        sheet_name="estadisticos_residuos",
        index=False
    )

print(f"Archivo Excel guardado en: {ruta_excel_normalidad}")


# ------------------------------------------------------------
# 13. Guardar resumen en TXT
# ------------------------------------------------------------

ruta_txt_normalidad = OUT_DIR_9 / "resumen_normalidad.txt"

with open(ruta_txt_normalidad, "w", encoding="utf-8") as f:
    f.write("ETAPA 9. NORMALIDAD\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Pruebas aplicadas:\n")
    f.write("1. Jarque-Bera\n")
    f.write("2. Shapiro-Wilk\n\n")
    
    f.write("Resultados de pruebas de normalidad:\n")
    f.write(resultados_normalidad.to_string(index=False))
    f.write("\n\n")
    
    f.write("Estadísticos descriptivos de residuos:\n")
    f.write(estadisticos_residuos_norm.to_string(index=False))
    f.write("\n\n")
    
    f.write("Conclusiones:\n")
    f.write(conclusion_norm_rls + "\n")
    f.write(conclusion_norm_rlm + "\n\n")
    
    f.write("=" * 70 + "\n\n")
    f.write("Resumen modelo RLS:\n")
    f.write(str(modelo_rls_norm.summary()))
    f.write("\n\n")
    
    f.write("Resumen modelo RLM:\n")
    f.write(str(modelo_rlm_norm.summary()))

print(f"Resumen TXT guardado en: {ruta_txt_normalidad}")


# ------------------------------------------------------------
# 14. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

display(resultados_normalidad)

print("\nEstadísticos descriptivos de residuos:")
display(estadisticos_residuos_norm)

print("\nConclusión RLS:")
print(conclusion_norm_rls)

print("\nConclusión RLM:")
print(conclusion_norm_rlm)
