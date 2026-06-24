"""
AUTOCORRELACIÓN
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 8. AUTOCORRELACIÓN**
######################################################################



# %% [notebook cell 32]
# ============================================================
# ETAPA 8. AUTOCORRELACIÓN
# Pruebas de Durbin-Watson y Breusch-Godfrey
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Carpeta de salida para esta etapa
OUT_DIR_8 = Path("resultados_etapa_8")
OUT_DIR_8.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Modelo de regresión lineal simple
# Defunciones diarias ~ Hospitalizaciones diarias
# ------------------------------------------------------------

df_rls_auto = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias"
]].copy()

for col in df_rls_auto.columns:
    df_rls_auto[col] = pd.to_numeric(df_rls_auto[col], errors="coerce")

df_rls_auto = df_rls_auto.dropna()

Y_rls_auto = df_rls_auto["defunciones_diarias"]
X_rls_auto = df_rls_auto[["hospitalizaciones_diarias"]]
X_rls_auto = sm.add_constant(X_rls_auto)

modelo_rls_auto = sm.OLS(Y_rls_auto, X_rls_auto).fit()

print(modelo_rls_auto.summary())


# ------------------------------------------------------------
# 2. Modelo de regresión lineal múltiple
# Defunciones diarias ~ Hospitalizaciones diarias + Casos diarios
# ------------------------------------------------------------

df_rlm_auto = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias",
    "casos_diarios"
]].copy()

for col in df_rlm_auto.columns:
    df_rlm_auto[col] = pd.to_numeric(df_rlm_auto[col], errors="coerce")

df_rlm_auto = df_rlm_auto.dropna()

Y_rlm_auto = df_rlm_auto["defunciones_diarias"]
X_rlm_auto = df_rlm_auto[[
    "hospitalizaciones_diarias",
    "casos_diarios"
]]
X_rlm_auto = sm.add_constant(X_rlm_auto)

modelo_rlm_auto = sm.OLS(Y_rlm_auto, X_rlm_auto).fit()

print(modelo_rlm_auto.summary())


# ------------------------------------------------------------
# 3. Funciones auxiliares
# ------------------------------------------------------------

def interpretar_durbin_watson(dw):
    """
    Interpretación práctica del estadístico Durbin-Watson.
    DW cercano a 2: no autocorrelación.
    DW menor a 2: posible autocorrelación positiva.
    DW mayor a 2: posible autocorrelación negativa.
    """
    if dw < 1.5:
        return "Posible autocorrelación positiva"
    elif dw <= 2.5:
        return "No se observa autocorrelación importante"
    else:
        return "Posible autocorrelación negativa"


def pruebas_autocorrelacion(modelo, nombre_modelo, max_lags=7):
    """
    Aplica Durbin-Watson y Breusch-Godfrey para varios rezagos.
    """
    residuos = modelo.resid
    
    # Durbin-Watson
    dw = durbin_watson(residuos)
    
    filas = []
    
    filas.append({
        "modelo": nombre_modelo,
        "prueba": "Durbin-Watson",
        "rezagos": 1,
        "estadistico": dw,
        "p_valor": np.nan,
        "decision_5": interpretar_durbin_watson(dw)
    })
    
    # Breusch-Godfrey para rezagos 1 a max_lags
    for lag in range(1, max_lags + 1):
        bg = acorr_breusch_godfrey(modelo, nlags=lag)
        
        filas.append({
            "modelo": nombre_modelo,
            "prueba": "Breusch-Godfrey",
            "rezagos": lag,
            "estadistico": bg[0],
            "p_valor": bg[1],
            "decision_5": (
                "Rechazar H0: existe autocorrelación"
                if bg[1] < 0.05
                else "No rechazar H0: no hay evidencia suficiente"
            )
        })
    
    return pd.DataFrame(filas)


def crear_tabla_residuos(modelo, nombre_modelo):
    """
    Crea tabla con predicciones, residuos y residuos rezagados.
    """
    tabla = pd.DataFrame({
        "modelo": nombre_modelo,
        "predichos": modelo.fittedvalues,
        "residuos": modelo.resid
    })
    
    tabla["residuos_lag1"] = tabla["residuos"].shift(1)
    tabla["residuos_lag7"] = tabla["residuos"].shift(7)
    
    return tabla



# %% [notebook cell 33]
# ------------------------------------------------------------
# 4. Aplicar pruebas a ambos modelos
# ------------------------------------------------------------

auto_rls = pruebas_autocorrelacion(
    modelo_rls_auto,
    "Regresión lineal simple",
    max_lags=7
)

auto_rlm = pruebas_autocorrelacion(
    modelo_rlm_auto,
    "Regresión lineal múltiple",
    max_lags=7
)

resultados_autocorrelacion = pd.concat(
    [auto_rls, auto_rlm],
    ignore_index=True
)

print("Resultados de autocorrelación")
display(resultados_autocorrelacion)


# ------------------------------------------------------------
# 5. Crear tablas de residuos
# ------------------------------------------------------------

residuos_rls = crear_tabla_residuos(
    modelo_rls_auto,
    "Regresión lineal simple"
)

residuos_rlm = crear_tabla_residuos(
    modelo_rlm_auto,
    "Regresión lineal múltiple"
)

display(residuos_rls.head())
display(residuos_rlm.head())


# ------------------------------------------------------------
# 6. Gráfica de residuos en el tiempo - RLS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(residuos_rls.index, residuos_rls["residuos"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Residuos en el tiempo - Regresión lineal simple")
plt.xlabel("Observación temporal")
plt.ylabel("Residuo")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_residuos_tiempo_rls = OUT_DIR_8 / "autocorrelacion_residuos_tiempo_rls.png"
plt.savefig(ruta_residuos_tiempo_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_residuos_tiempo_rls}")


# ------------------------------------------------------------
# 7. Gráfica de residuos en el tiempo - RLM
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(residuos_rlm.index, residuos_rlm["residuos"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Residuos en el tiempo - Regresión lineal múltiple")
plt.xlabel("Observación temporal")
plt.ylabel("Residuo")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_residuos_tiempo_rlm = OUT_DIR_8 / "autocorrelacion_residuos_tiempo_rlm.png"
plt.savefig(ruta_residuos_tiempo_rlm, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_residuos_tiempo_rlm}")


# ------------------------------------------------------------
# 8. Dispersión residuos vs residuos rezagados - RLS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(
    residuos_rls["residuos_lag1"],
    residuos_rls["residuos"],
    alpha=0.7
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Residuos actuales vs residuos rezagados - RLS")
plt.xlabel("Residuo rezagado t-1")
plt.ylabel("Residuo actual")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_res_lag_rls = OUT_DIR_8 / "autocorrelacion_residuos_lag1_rls.png"
plt.savefig(ruta_res_lag_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_res_lag_rls}")


# ------------------------------------------------------------
# 9. Dispersión residuos vs residuos rezagados - RLM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(
    residuos_rlm["residuos_lag1"],
    residuos_rlm["residuos"],
    alpha=0.7
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Residuos actuales vs residuos rezagados - RLM")
plt.xlabel("Residuo rezagado t-1")
plt.ylabel("Residuo actual")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_res_lag_rlm = OUT_DIR_8 / "autocorrelacion_residuos_lag1_rlm.png"
plt.savefig(ruta_res_lag_rlm, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_res_lag_rlm}")



# %% [notebook cell 34]
# ------------------------------------------------------------
# 10. Conclusión de resultados
# ------------------------------------------------------------

def conclusion_autocorrelacion(tabla_resultados, nombre_modelo):
    sub = tabla_resultados[
        (tabla_resultados["modelo"] == nombre_modelo) &
        (tabla_resultados["prueba"] == "Breusch-Godfrey")
    ]
    
    existe_auto = (sub["p_valor"] < 0.05).any()
    
    dw_row = tabla_resultados[
        (tabla_resultados["modelo"] == nombre_modelo) &
        (tabla_resultados["prueba"] == "Durbin-Watson")
    ].iloc[0]
    
    dw = dw_row["estadistico"]
    interp_dw = dw_row["decision_5"]
    
    if existe_auto:
        return (
            f"En el modelo {nombre_modelo}, el estadístico Durbin-Watson fue {dw:.4f}, "
            f"lo que sugiere: {interp_dw}. Además, la prueba de Breusch-Godfrey "
            f"rechaza la hipótesis nula en al menos un rezago, por lo que existe evidencia "
            f"de autocorrelación en los residuos. Se recomienda considerar modelos con rezagos "
            f"o modelos de series de tiempo."
        )
    else:
        return (
            f"En el modelo {nombre_modelo}, el estadístico Durbin-Watson fue {dw:.4f}, "
            f"lo que sugiere: {interp_dw}. La prueba de Breusch-Godfrey no rechaza la "
            f"hipótesis nula en los rezagos evaluados, por lo que no se detecta evidencia "
            f"suficiente de autocorrelación al 5%."
        )


conclusion_auto_rls = conclusion_autocorrelacion(
    resultados_autocorrelacion,
    "Regresión lineal simple"
)

conclusion_auto_rlm = conclusion_autocorrelacion(
    resultados_autocorrelacion,
    "Regresión lineal múltiple"
)

print(conclusion_auto_rls)
print(conclusion_auto_rlm)


# ------------------------------------------------------------
# 11. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_auto = OUT_DIR_8 / "resultados_autocorrelacion.xlsx"

with pd.ExcelWriter(ruta_excel_auto, engine="openpyxl") as writer:
    resultados_autocorrelacion.to_excel(
        writer,
        sheet_name="pruebas_autocorrelacion",
        index=False
    )
    
    residuos_rls.to_excel(
        writer,
        sheet_name="residuos_rls",
        index=False
    )
    
    residuos_rlm.to_excel(
        writer,
        sheet_name="residuos_rlm",
        index=False
    )

print(f"Archivo Excel guardado en: {ruta_excel_auto}")


# ------------------------------------------------------------
# 12. Guardar resumen en TXT
# ------------------------------------------------------------

ruta_txt_auto = OUT_DIR_8 / "resumen_autocorrelacion.txt"

with open(ruta_txt_auto, "w", encoding="utf-8") as f:
    f.write("ETAPA 8. AUTOCORRELACIÓN\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Pruebas aplicadas:\n")
    f.write("1. Durbin-Watson\n")
    f.write("2. Breusch-Godfrey\n\n")
    
    f.write("Resultados:\n")
    f.write(resultados_autocorrelacion.to_string(index=False))
    f.write("\n\n")
    
    f.write("Conclusiones:\n")
    f.write(conclusion_auto_rls + "\n")
    f.write(conclusion_auto_rlm + "\n\n")
    
    f.write("=" * 70 + "\n\n")
    f.write("Resumen modelo RLS:\n")
    f.write(str(modelo_rls_auto.summary()))
    f.write("\n\n")
    
    f.write("Resumen modelo RLM:\n")
    f.write(str(modelo_rlm_auto.summary()))

print(f"Resumen TXT guardado en: {ruta_txt_auto}")


# ------------------------------------------------------------
# 13. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

display(resultados_autocorrelacion)

print("\nConclusión RLS:")
print(conclusion_auto_rls)

print("\nConclusión RLM:")
print(conclusion_auto_rlm)
