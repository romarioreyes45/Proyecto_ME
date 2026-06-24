"""
SERIES DE TIEMPO
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 13. SERIES DE TIEMPO**
######################################################################



# %% [notebook cell 54]
# ============================================================
# ETAPA 13. SERIES DE TIEMPO
# ADF, AR, MA, ARMA, ARIMA y pronósticos a 7, 14 y 30 días
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

# Carpeta de salida para esta etapa
OUT_DIR_13 = Path("resultados_etapa_13")
OUT_DIR_13.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar base temporal
# ------------------------------------------------------------

df_series = df_ts.copy()

if "fecha" in df_series.columns:
    df_series["fecha"] = pd.to_datetime(df_series["fecha"], errors="coerce")
    df_series = df_series.sort_values("fecha")
    df_series = df_series.set_index("fecha")
else:
    df_series = df_series.sort_index()

variables_series = [
    "casos_diarios",
    "hospitalizaciones_diarias",
    "defunciones_diarias"
]

for col in variables_series:
    df_series[col] = pd.to_numeric(df_series[col], errors="coerce")

# Asegurar frecuencia diaria
df_series = df_series.asfreq("D")

# Rellenar posibles huecos con 0 porque son conteos diarios
df_series[variables_series] = df_series[variables_series].fillna(0)

print("Rango temporal:")
print(df_series.index.min(), "a", df_series.index.max())

print("\nPrimeras observaciones:")
display(df_series[variables_series].head())

print("\nÚltimas observaciones:")
display(df_series[variables_series].tail())


# ------------------------------------------------------------
# 2. Gráfica de las tres series temporales
# ------------------------------------------------------------

for variable in variables_series:
    plt.figure(figsize=(10, 5))
    plt.plot(df_series.index, df_series[variable])
    plt.title(f"Serie temporal: {variable}")
    plt.xlabel("Fecha")
    plt.ylabel(variable)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    ruta = OUT_DIR_13 / f"serie_temporal_{variable}.png"
    plt.savefig(ruta, dpi=300)
    plt.show()
    
    print(f"Gráfica guardada en: {ruta}")


# ------------------------------------------------------------
# 3. Función para prueba Dickey-Fuller Aumentada
# ------------------------------------------------------------

def aplicar_adf(serie, nombre_serie):
    """
    Aplica prueba ADF.
    H0: la serie tiene raíz unitaria, es decir, no es estacionaria.
    H1: la serie es estacionaria.

    La función valida que la serie tenga suficientes observaciones y variación.
    """
    serie = pd.Series(serie).dropna()

    if len(serie) < 10 or serie.nunique() < 2:
        return {
            "serie": nombre_serie,
            "estadistico_adf": np.nan,
            "p_valor": np.nan,
            "rezagos_usados": np.nan,
            "observaciones": len(serie),
            "valor_critico_1%": np.nan,
            "valor_critico_5%": np.nan,
            "valor_critico_10%": np.nan,
            "decision_5": "No calculable: serie corta o sin variación suficiente"
        }

    try:
        resultado = adfuller(serie, autolag="AIC")

        tabla = {
            "serie": nombre_serie,
            "estadistico_adf": resultado[0],
            "p_valor": resultado[1],
            "rezagos_usados": resultado[2],
            "observaciones": resultado[3],
            "valor_critico_1%": resultado[4]["1%"],
            "valor_critico_5%": resultado[4]["5%"],
            "valor_critico_10%": resultado[4]["10%"],
            "decision_5": (
                "Estacionaria: se rechaza H0"
                if resultado[1] < 0.05
                else "No estacionaria: no se rechaza H0"
            )
        }
    except Exception as e:
        tabla = {
            "serie": nombre_serie,
            "estadistico_adf": np.nan,
            "p_valor": np.nan,
            "rezagos_usados": np.nan,
            "observaciones": len(serie),
            "valor_critico_1%": np.nan,
            "valor_critico_5%": np.nan,
            "valor_critico_10%": np.nan,
            "decision_5": f"No calculable: {e}"
        }

    return tabla


# ------------------------------------------------------------
# 4. Aplicar ADF a las series originales
# ------------------------------------------------------------

resultados_adf_original = []

for variable in variables_series:
    resultados_adf_original.append(
        aplicar_adf(df_series[variable], variable)
    )

tabla_adf_original = pd.DataFrame(resultados_adf_original)

print("Prueba ADF sobre series originales")
display(tabla_adf_original)



# %% [notebook cell 55]
# ------------------------------------------------------------
# 5. Crear primeras diferencias y aplicar ADF
# ------------------------------------------------------------

df_series_diff = df_series[variables_series].diff().dropna()

resultados_adf_diff = []

for variable in variables_series:
    resultados_adf_diff.append(
        aplicar_adf(df_series_diff[variable], f"{variable}_diferenciada")
    )

tabla_adf_diff = pd.DataFrame(resultados_adf_diff)

print("Prueba ADF sobre primeras diferencias")
display(tabla_adf_diff)


# ------------------------------------------------------------
# 6. Gráficas de series diferenciadas
# ------------------------------------------------------------

for variable in variables_series:
    plt.figure(figsize=(10, 5))
    plt.plot(df_series_diff.index, df_series_diff[variable])
    plt.title(f"Primera diferencia: {variable}")
    plt.xlabel("Fecha")
    plt.ylabel(f"Diferencia de {variable}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    ruta = OUT_DIR_13 / f"serie_diferenciada_{variable}.png"
    plt.savefig(ruta, dpi=300)
    plt.show()
    
    print(f"Gráfica guardada en: {ruta}")


# ------------------------------------------------------------
# 7. Serie principal para modelado
# ------------------------------------------------------------

serie_objetivo = "defunciones_diarias"

y = df_series[serie_objetivo].copy()
y = y.dropna()

print("Serie objetivo:", serie_objetivo)
display(y.head())


# ------------------------------------------------------------
# 8. Función para ajustar modelos ARIMA
# ------------------------------------------------------------

def ajustar_modelo_arima(serie, orden, nombre_modelo):
    """
    Ajusta un modelo ARIMA con el orden indicado.
    AR(p)      = ARIMA(p,0,0)
    MA(q)      = ARIMA(0,0,q)
    ARMA(p,q)  = ARIMA(p,0,q)
    ARIMA(p,d,q)
    """
    try:
        modelo = ARIMA(serie, order=orden).fit()
        
        resultado = {
            "modelo": nombre_modelo,
            "orden": str(orden),
            "aic": modelo.aic,
            "bic": modelo.bic,
            "log_likelihood": modelo.llf,
            "convergio": True,
            "objeto_modelo": modelo
        }
        
    except Exception as e:
        resultado = {
            "modelo": nombre_modelo,
            "orden": str(orden),
            "aic": np.nan,
            "bic": np.nan,
            "log_likelihood": np.nan,
            "convergio": False,
            "error": str(e),
            "objeto_modelo": None
        }
    
    return resultado


# ------------------------------------------------------------
# 9. Ajustar modelos requeridos
# ------------------------------------------------------------

modelos_candidatos = [
    ("AR(1)", (1, 0, 0)),
    ("MA(1)", (0, 0, 1)),
    ("ARMA(1,1)", (1, 0, 1)),
    ("ARIMA(1,1,1)", (1, 1, 1))
]

resultados_modelos = []

for nombre, orden in modelos_candidatos:
    resultados_modelos.append(
        ajustar_modelo_arima(y, orden, nombre)
    )

tabla_modelos = pd.DataFrame([
    {
        "modelo": r["modelo"],
        "orden": r["orden"],
        "aic": r["aic"],
        "bic": r["bic"],
        "log_likelihood": r["log_likelihood"],
        "convergio": r["convergio"]
    }
    for r in resultados_modelos
])

tabla_modelos = tabla_modelos.sort_values("aic").reset_index(drop=True)

print("Comparación de modelos AR, MA, ARMA y ARIMA")
display(tabla_modelos)


# ------------------------------------------------------------
# 10. Seleccionar mejor modelo por AIC
# ------------------------------------------------------------

tabla_modelos_validos = tabla_modelos.dropna(subset=["aic"]).copy()

if tabla_modelos_validos.empty:
    raise RuntimeError("Ningún modelo ARIMA pudo ajustarse correctamente. Revisa la serie y los datos de entrada.")

mejor_fila = tabla_modelos_validos.iloc[0]
mejor_nombre = mejor_fila["modelo"]

mejor_modelo = None

for r in resultados_modelos:
    if r["modelo"] == mejor_nombre:
        mejor_modelo = r["objeto_modelo"]

print("Mejor modelo según AIC:")
print(mejor_nombre)
print(mejor_fila)

print("\nResumen del mejor modelo:")
print(mejor_modelo.summary())



# %% [notebook cell 56]
# ------------------------------------------------------------
# 11. Tabla de coeficientes del mejor modelo
# ------------------------------------------------------------

tabla_coef_mejor_modelo = pd.DataFrame({
    "parametro": mejor_modelo.params.index,
    "coeficiente": mejor_modelo.params.values,
    "error_estandar": mejor_modelo.bse.values,
    "z": mejor_modelo.tvalues.values,
    "p_valor": mejor_modelo.pvalues.values,
    "IC_95_inf": mejor_modelo.conf_int().iloc[:, 0].values,
    "IC_95_sup": mejor_modelo.conf_int().iloc[:, 1].values
})

print("Coeficientes del mejor modelo")
display(tabla_coef_mejor_modelo)


# ------------------------------------------------------------
# 12. Residuos del mejor modelo
# ------------------------------------------------------------

residuos_mejor_modelo = mejor_modelo.resid

df_residuos_arima = pd.DataFrame({
    "fecha": residuos_mejor_modelo.index,
    "residuo": residuos_mejor_modelo.values
})

plt.figure(figsize=(10, 5))
plt.plot(residuos_mejor_modelo.index, residuos_mejor_modelo.values)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title(f"Residuos del mejor modelo: {mejor_nombre}")
plt.xlabel("Fecha")
plt.ylabel("Residuo")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_residuos_arima = OUT_DIR_13 / "residuos_mejor_modelo_arima.png"
plt.savefig(ruta_residuos_arima, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_residuos_arima}")


# ------------------------------------------------------------
# 13. Función para pronosticar
# ------------------------------------------------------------

def generar_pronostico(modelo, pasos, nombre_modelo):
    """
    Genera pronóstico con intervalos de confianza.
    """
    forecast_res = modelo.get_forecast(steps=pasos)
    pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    
    fechas_futuras = pd.date_range(
        start=y.index.max() + pd.Timedelta(days=1),
        periods=pasos,
        freq="D"
    )
    
    pronostico = pd.DataFrame({
        "fecha": fechas_futuras,
        "horizonte": pasos,
        "modelo": nombre_modelo,
        "pronostico": pred_mean.values,
        "IC_95_inf": conf_int.iloc[:, 0].values,
        "IC_95_sup": conf_int.iloc[:, 1].values
    })
    
    # Como son conteos, evitamos pronósticos negativos en la tabla final
    pronostico["pronostico_ajustado"] = pronostico["pronostico"].clip(lower=0)
    pronostico["IC_95_inf_ajustado"] = pronostico["IC_95_inf"].clip(lower=0)
    
    return pronostico


# ------------------------------------------------------------
# 14. Generar pronósticos de 7, 14 y 30 días
# ------------------------------------------------------------

pronostico_7 = generar_pronostico(mejor_modelo, 7, mejor_nombre)
pronostico_14 = generar_pronostico(mejor_modelo, 14, mejor_nombre)
pronostico_30 = generar_pronostico(mejor_modelo, 30, mejor_nombre)

pronosticos_total = pd.concat(
    [pronostico_7, pronostico_14, pronostico_30],
    ignore_index=True
)

print("Pronósticos generados")
display(pronosticos_total.head())
display(pronosticos_total.tail())


# ------------------------------------------------------------
# 15. Gráfica de pronóstico a 30 días
# ------------------------------------------------------------

historial = y.copy()

pron_30 = pronostico_30.copy()

plt.figure(figsize=(11, 5))

plt.plot(
    historial.index,
    historial.values,
    label="Serie observada"
)

plt.plot(
    pron_30["fecha"],
    pron_30["pronostico_ajustado"],
    label="Pronóstico 30 días"
)

plt.fill_between(
    pron_30["fecha"],
    pron_30["IC_95_inf_ajustado"],
    pron_30["IC_95_sup"],
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

plt.title(f"Pronóstico de defunciones diarias a 30 días - {mejor_nombre}")
plt.xlabel("Fecha")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_pron_30 = OUT_DIR_13 / "pronostico_defunciones_30_dias.png"
plt.savefig(ruta_pron_30, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_pron_30}")



# %% [notebook cell 57]
# ------------------------------------------------------------
# 16. Gráfica enfocada: últimos 90 días + pronóstico
# ------------------------------------------------------------

historial_90 = historial.tail(90)

plt.figure(figsize=(11, 5))

plt.plot(
    historial_90.index,
    historial_90.values,
    label="Últimos 90 días observados"
)

plt.plot(
    pron_30["fecha"],
    pron_30["pronostico_ajustado"],
    label="Pronóstico 30 días"
)

plt.fill_between(
    pron_30["fecha"],
    pron_30["IC_95_inf_ajustado"],
    pron_30["IC_95_sup"],
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

plt.title(f"Pronóstico enfocado de defunciones diarias - {mejor_nombre}")
plt.xlabel("Fecha")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_pron_30_zoom = OUT_DIR_13 / "pronostico_defunciones_30_dias_zoom.png"
plt.savefig(ruta_pron_30_zoom, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_pron_30_zoom}")


# ------------------------------------------------------------
# 17. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_series = OUT_DIR_13 / "resultados_series_tiempo_arima.xlsx"

with pd.ExcelWriter(ruta_excel_series, engine="openpyxl") as writer:
    df_series[variables_series].to_excel(writer, sheet_name="series_temporales")
    tabla_adf_original.to_excel(writer, sheet_name="adf_original", index=False)
    tabla_adf_diff.to_excel(writer, sheet_name="adf_diferenciada", index=False)
    tabla_modelos.to_excel(writer, sheet_name="comparacion_modelos", index=False)
    tabla_coef_mejor_modelo.to_excel(writer, sheet_name="coef_mejor_modelo", index=False)
    df_residuos_arima.to_excel(writer, sheet_name="residuos_mejor_modelo", index=False)
    pronosticos_total.to_excel(writer, sheet_name="pronosticos_7_14_30", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_series}")


# ------------------------------------------------------------
# 18. Guardar resumen en TXT
# ------------------------------------------------------------

ruta_txt_series = OUT_DIR_13 / "resumen_series_tiempo_arima.txt"

with open(ruta_txt_series, "w", encoding="utf-8") as f:
    f.write("ETAPA 13. SERIES DE TIEMPO\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Series utilizadas:\n")
    f.write(", ".join(variables_series))
    f.write("\n\n")
    
    f.write("Prueba ADF - series originales:\n")
    f.write(tabla_adf_original.to_string(index=False))
    f.write("\n\n")
    
    f.write("Prueba ADF - primeras diferencias:\n")
    f.write(tabla_adf_diff.to_string(index=False))
    f.write("\n\n")
    
    f.write("Comparación AR, MA, ARMA y ARIMA:\n")
    f.write(tabla_modelos.to_string(index=False))
    f.write("\n\n")
    
    f.write(f"Mejor modelo según AIC: {mejor_nombre}\n\n")
    f.write(str(mejor_modelo.summary()))
    f.write("\n\n")
    
    f.write("Pronósticos 7, 14 y 30 días:\n")
    f.write(pronosticos_total.to_string(index=False))

print(f"Resumen TXT guardado en: {ruta_txt_series}")


# ------------------------------------------------------------
# 19. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nADF series originales")
display(tabla_adf_original)

print("\nADF primeras diferencias")
display(tabla_adf_diff)

print("\nComparación de modelos")
display(tabla_modelos)

print("\nMejor modelo según AIC:")
print(mejor_nombre)

print("\nCoeficientes del mejor modelo")
display(tabla_coef_mejor_modelo)

print("\nPronóstico a 7 días")
display(pronostico_7)

print("\nPronóstico a 14 días")
display(pronostico_14)

print("\nPronóstico a 30 días")
display(pronostico_30)
