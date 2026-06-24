"""
MODELOS CON REZAGOS DISTRIBUIDOS
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 10. MODELOS CON REZAGOS DISTRIBUIDOS**
######################################################################



# %% [notebook cell 40]
# ============================================================
# ETAPA 10. MODELOS CON REZAGOS DISTRIBUIDOS
# Defunciones diarias explicadas por casos contemporáneos
# y casos rezagados a 1, 7 y 14 días
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_10 = Path("resultados_etapa_10")
OUT_DIR_10.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar serie temporal
# ------------------------------------------------------------

df_rezagos = df_ts.copy()

# Asegurar que la fecha esté ordenada
if "fecha" in df_rezagos.columns:
    df_rezagos["fecha"] = pd.to_datetime(df_rezagos["fecha"], errors="coerce")
    df_rezagos = df_rezagos.sort_values("fecha")
else:
    df_rezagos = df_rezagos.sort_index()

# Convertir variables a numéricas
variables_base = [
    "defunciones_diarias",
    "casos_diarios",
    "hospitalizaciones_diarias"
]

for col in variables_base:
    if col in df_rezagos.columns:
        df_rezagos[col] = pd.to_numeric(df_rezagos[col], errors="coerce")

display(df_rezagos.head())


# ------------------------------------------------------------
# 2. Crear rezagos de casos diarios
# ------------------------------------------------------------

df_rezagos["casos_lag_1"] = df_rezagos["casos_diarios"].shift(1)
df_rezagos["casos_lag_7"] = df_rezagos["casos_diarios"].shift(7)
df_rezagos["casos_lag_14"] = df_rezagos["casos_diarios"].shift(14)

# También creamos rezagos de hospitalizaciones como análisis complementario
df_rezagos["hospitalizaciones_lag_1"] = df_rezagos["hospitalizaciones_diarias"].shift(1)
df_rezagos["hospitalizaciones_lag_7"] = df_rezagos["hospitalizaciones_diarias"].shift(7)
df_rezagos["hospitalizaciones_lag_14"] = df_rezagos["hospitalizaciones_diarias"].shift(14)

display(df_rezagos.head(20))


# ------------------------------------------------------------
# 3. Modelo principal:
# Defunciones diarias ~ casos contemporáneos y rezagados
# ------------------------------------------------------------

variables_modelo_casos = [
    "casos_diarios",
    "casos_lag_1",
    "casos_lag_7",
    "casos_lag_14"
]

df_modelo_rezagos_casos = df_rezagos[
    ["defunciones_diarias"] + variables_modelo_casos
].dropna().copy()

Y_casos = df_modelo_rezagos_casos["defunciones_diarias"]
X_casos = df_modelo_rezagos_casos[variables_modelo_casos]
X_casos = sm.add_constant(X_casos)

modelo_rezagos_casos = sm.OLS(Y_casos, X_casos).fit()

print(modelo_rezagos_casos.summary())


# ------------------------------------------------------------
# 4. Modelo complementario:
# Defunciones diarias ~ hospitalizaciones contemporáneas y rezagadas
# ------------------------------------------------------------

variables_modelo_hosp = [
    "hospitalizaciones_diarias",
    "hospitalizaciones_lag_1",
    "hospitalizaciones_lag_7",
    "hospitalizaciones_lag_14"
]

df_modelo_rezagos_hosp = df_rezagos[
    ["defunciones_diarias"] + variables_modelo_hosp
].dropna().copy()

Y_hosp = df_modelo_rezagos_hosp["defunciones_diarias"]
X_hosp = df_modelo_rezagos_hosp[variables_modelo_hosp]
X_hosp = sm.add_constant(X_hosp)

modelo_rezagos_hosp = sm.OLS(Y_hosp, X_hosp).fit()

print(modelo_rezagos_hosp.summary())


# ------------------------------------------------------------
# 5. Función para tabla de coeficientes
# ------------------------------------------------------------

def tabla_coeficientes(modelo, nombre_modelo):
    tabla = pd.DataFrame({
        "modelo": nombre_modelo,
        "variable": modelo.params.index,
        "coeficiente": modelo.params.values,
        "error_estandar": modelo.bse.values,
        "t": modelo.tvalues.values,
        "p_valor": modelo.pvalues.values,
        "IC_95_inf": modelo.conf_int()[0].values,
        "IC_95_sup": modelo.conf_int()[1].values
    })
    return tabla


tabla_rezagos_casos = tabla_coeficientes(
    modelo_rezagos_casos,
    "Modelo con rezagos de casos"
)

tabla_rezagos_hosp = tabla_coeficientes(
    modelo_rezagos_hosp,
    "Modelo con rezagos de hospitalizaciones"
)

print("Modelo con rezagos de casos")
display(tabla_rezagos_casos)

print("Modelo con rezagos de hospitalizaciones")
display(tabla_rezagos_hosp)



# %% [notebook cell 41]
# ------------------------------------------------------------
# 6. Métricas generales
# ------------------------------------------------------------

metricas_rezagos = pd.DataFrame({
    "modelo": [
        "Modelo con rezagos de casos",
        "Modelo con rezagos de hospitalizaciones"
    ],
    "variable_dependiente": [
        "defunciones_diarias",
        "defunciones_diarias"
    ],
    "observaciones": [
        int(modelo_rezagos_casos.nobs),
        int(modelo_rezagos_hosp.nobs)
    ],
    "r_cuadrada": [
        modelo_rezagos_casos.rsquared,
        modelo_rezagos_hosp.rsquared
    ],
    "r_cuadrada_ajustada": [
        modelo_rezagos_casos.rsquared_adj,
        modelo_rezagos_hosp.rsquared_adj
    ],
    "estadistico_f": [
        modelo_rezagos_casos.fvalue,
        modelo_rezagos_hosp.fvalue
    ],
    "p_valor_f": [
        modelo_rezagos_casos.f_pvalue,
        modelo_rezagos_hosp.f_pvalue
    ],
    "aic": [
        modelo_rezagos_casos.aic,
        modelo_rezagos_hosp.aic
    ],
    "bic": [
        modelo_rezagos_casos.bic,
        modelo_rezagos_hosp.bic
    ]
})

display(metricas_rezagos)


# ------------------------------------------------------------
# 7. Interpretación de efectos
# ------------------------------------------------------------

def efectos_rezagos(modelo, variables_rezago, nombre_modelo):
    coefs = modelo.params
    
    efecto_contemporaneo = coefs.get(variables_rezago[0], np.nan)
    efecto_lag_1 = coefs.get(variables_rezago[1], np.nan)
    efecto_lag_7 = coefs.get(variables_rezago[2], np.nan)
    efecto_lag_14 = coefs.get(variables_rezago[3], np.nan)
    
    efecto_acumulado = (
        efecto_contemporaneo +
        efecto_lag_1 +
        efecto_lag_7 +
        efecto_lag_14
    )
    
    # En un modelo de rezagos distribuidos finito, el multiplicador de largo plazo
    # se interpreta como la suma de los coeficientes de los rezagos incluidos.
    multiplicador_largo_plazo = efecto_acumulado
    
    tabla = pd.DataFrame({
        "modelo": [nombre_modelo],
        "efecto_contemporaneo": [efecto_contemporaneo],
        "efecto_lag_1": [efecto_lag_1],
        "efecto_lag_7": [efecto_lag_7],
        "efecto_lag_14": [efecto_lag_14],
        "efecto_acumulado": [efecto_acumulado],
        "multiplicador_largo_plazo": [multiplicador_largo_plazo]
    })
    
    return tabla


efectos_casos = efectos_rezagos(
    modelo_rezagos_casos,
    variables_modelo_casos,
    "Modelo con rezagos de casos"
)

efectos_hosp = efectos_rezagos(
    modelo_rezagos_hosp,
    variables_modelo_hosp,
    "Modelo con rezagos de hospitalizaciones"
)

efectos_rezagos_total = pd.concat(
    [efectos_casos, efectos_hosp],
    ignore_index=True
)

display(efectos_rezagos_total)


# ------------------------------------------------------------
# 8. Predicciones y residuos
# ------------------------------------------------------------

df_pred_rezagos_casos = df_modelo_rezagos_casos.copy()
df_pred_rezagos_casos["defunciones_predichas"] = modelo_rezagos_casos.fittedvalues
df_pred_rezagos_casos["residuos"] = modelo_rezagos_casos.resid

df_pred_rezagos_hosp = df_modelo_rezagos_hosp.copy()
df_pred_rezagos_hosp["defunciones_predichas"] = modelo_rezagos_hosp.fittedvalues
df_pred_rezagos_hosp["residuos"] = modelo_rezagos_hosp.resid

display(df_pred_rezagos_casos.head())
display(df_pred_rezagos_hosp.head())


# ------------------------------------------------------------
# 9. Gráfica de coeficientes de rezagos de casos
# ------------------------------------------------------------

coef_plot_casos = tabla_rezagos_casos[
    tabla_rezagos_casos["variable"] != "const"
].copy()

plt.figure(figsize=(8, 5))
plt.bar(coef_plot_casos["variable"], coef_plot_casos["coeficiente"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Coeficientes del modelo con rezagos de casos")
plt.xlabel("Variable")
plt.ylabel("Coeficiente estimado")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_coef_casos = OUT_DIR_10 / "coeficientes_rezagos_casos.png"
plt.savefig(ruta_coef_casos, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_coef_casos}")



# %% [notebook cell 42]
# ------------------------------------------------------------
# 10. Gráfica de coeficientes de rezagos de hospitalizaciones
# ------------------------------------------------------------

coef_plot_hosp = tabla_rezagos_hosp[
    tabla_rezagos_hosp["variable"] != "const"
].copy()

plt.figure(figsize=(8, 5))
plt.bar(coef_plot_hosp["variable"], coef_plot_hosp["coeficiente"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Coeficientes del modelo con rezagos de hospitalizaciones")
plt.xlabel("Variable")
plt.ylabel("Coeficiente estimado")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_coef_hosp = OUT_DIR_10 / "coeficientes_rezagos_hospitalizaciones.png"
plt.savefig(ruta_coef_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_coef_hosp}")


# ------------------------------------------------------------
# 11. Defunciones observadas vs predichas - modelo casos
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_pred_rezagos_casos.index,
    df_pred_rezagos_casos["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_pred_rezagos_casos.index,
    df_pred_rezagos_casos["defunciones_predichas"],
    label="Defunciones predichas"
)

plt.title("Modelo con rezagos de casos: observadas vs predichas")
plt.xlabel("Observación temporal")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_pred_casos = OUT_DIR_10 / "predicciones_rezagos_casos.png"
plt.savefig(ruta_pred_casos, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_pred_casos}")


# ------------------------------------------------------------
# 12. Defunciones observadas vs predichas - modelo hospitalizaciones
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_pred_rezagos_hosp.index,
    df_pred_rezagos_hosp["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_pred_rezagos_hosp.index,
    df_pred_rezagos_hosp["defunciones_predichas"],
    label="Defunciones predichas"
)

plt.title("Modelo con rezagos de hospitalizaciones: observadas vs predichas")
plt.xlabel("Observación temporal")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_pred_hosp = OUT_DIR_10 / "predicciones_rezagos_hospitalizaciones.png"
plt.savefig(ruta_pred_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_pred_hosp}")


# ------------------------------------------------------------
# 13. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_rezagos = OUT_DIR_10 / "resultados_modelos_rezagos_distribuidos.xlsx"

with pd.ExcelWriter(ruta_excel_rezagos, engine="openpyxl") as writer:
    tabla_rezagos_casos.to_excel(writer, sheet_name="coef_rezagos_casos", index=False)
    tabla_rezagos_hosp.to_excel(writer, sheet_name="coef_rezagos_hosp", index=False)
    metricas_rezagos.to_excel(writer, sheet_name="metricas_modelos", index=False)
    efectos_rezagos_total.to_excel(writer, sheet_name="efectos_acumulados", index=False)
    df_pred_rezagos_casos.to_excel(writer, sheet_name="pred_resid_casos", index=False)
    df_pred_rezagos_hosp.to_excel(writer, sheet_name="pred_resid_hosp", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_rezagos}")


# ------------------------------------------------------------
# 14. Guardar resúmenes en TXT
# ------------------------------------------------------------

ruta_txt_rezagos = OUT_DIR_10 / "resumen_modelos_rezagos_distribuidos.txt"

with open(ruta_txt_rezagos, "w", encoding="utf-8") as f:
    f.write("ETAPA 10. MODELOS CON REZAGOS DISTRIBUIDOS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Modelo principal: defunciones diarias explicadas por casos diarios contemporáneos y rezagados.\n")
    f.write(str(modelo_rezagos_casos.summary()))
    f.write("\n\n")
    
    f.write("Modelo complementario: defunciones diarias explicadas por hospitalizaciones contemporáneas y rezagadas.\n")
    f.write(str(modelo_rezagos_hosp.summary()))
    f.write("\n\n")
    
    f.write("Efectos acumulados y multiplicadores:\n")
    f.write(efectos_rezagos_total.to_string(index=False))

print(f"Resumen TXT guardado en: {ruta_txt_rezagos}")


# ------------------------------------------------------------
# 15. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nCoeficientes modelo con rezagos de casos")
display(tabla_rezagos_casos)

print("\nCoeficientes modelo con rezagos de hospitalizaciones")
display(tabla_rezagos_hosp)

print("\nEfectos acumulados y multiplicadores")
display(efectos_rezagos_total)

print("\nMétricas de los modelos")
display(metricas_rezagos)
