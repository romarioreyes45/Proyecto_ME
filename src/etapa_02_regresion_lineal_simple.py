"""
REGRESIÓN LINEAL SIMPLE
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 2. REGRESIÓN LINEAL SIMPLE**
######################################################################



# %% [notebook cell 9]
# ============================================================
# ETAPA 2. REGRESIÓN LINEAL SIMPLE
# Modelo: Defunciones diarias en función de hospitalizaciones diarias
# ============================================================

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_2 = Path("resultados_etapa_2")
OUT_DIR_2.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Preparar datos del modelo
# ------------------------------------------------------------

df_rls = df_ts[["defunciones_diarias", "hospitalizaciones_diarias"]].dropna().copy()

df_rls["defunciones_diarias"] = pd.to_numeric(df_rls["defunciones_diarias"], errors="coerce")
df_rls["hospitalizaciones_diarias"] = pd.to_numeric(df_rls["hospitalizaciones_diarias"], errors="coerce")

df_rls = df_rls.dropna()

print("Observaciones utilizadas en el modelo:", len(df_rls))
display(df_rls.head())


# ------------------------------------------------------------
# 2. Estimar regresión lineal simple
# ------------------------------------------------------------

Y = df_rls["defunciones_diarias"]
X = df_rls[["hospitalizaciones_diarias"]]
X = sm.add_constant(X)

modelo_rls = sm.OLS(Y, X).fit()

print(modelo_rls.summary())


# ------------------------------------------------------------
# 3. Tabla ordenada de resultados
# ------------------------------------------------------------

tabla_rls = pd.DataFrame({
    "coeficiente": modelo_rls.params,
    "error_estandar": modelo_rls.bse,
    "t": modelo_rls.tvalues,
    "p_valor": modelo_rls.pvalues,
    "IC_95_inf": modelo_rls.conf_int()[0],
    "IC_95_sup": modelo_rls.conf_int()[1]
})

display(tabla_rls)

print("R cuadrada:", modelo_rls.rsquared)
print("R cuadrada ajustada:", modelo_rls.rsquared_adj)


# ------------------------------------------------------------
# 5. Guardar resultados de la regresión lineal simple
# ------------------------------------------------------------

# La carpeta OUT_DIR_2 ya fue creada al inicio de la etapa.

# ------------------------------------------------------------
# 6.1. Crear tabla de métricas generales del modelo
# ------------------------------------------------------------

metricas_rls = pd.DataFrame({
    "indicador": [
        "Variable dependiente",
        "Variable explicativa",
        "Número de observaciones",
        "R cuadrada",
        "R cuadrada ajustada",
        "Estadístico F",
        "P-valor F",
        "AIC",
        "BIC"
    ],
    "valor": [
        "defunciones_diarias",
        "hospitalizaciones_diarias",
        int(modelo_rls.nobs),
        modelo_rls.rsquared,
        modelo_rls.rsquared_adj,
        modelo_rls.fvalue,
        modelo_rls.f_pvalue,
        modelo_rls.aic,
        modelo_rls.bic
    ]
})

display(metricas_rls)


# ------------------------------------------------------------
# 6.2. Crear tabla con predicciones y residuos
# ------------------------------------------------------------

df_resultados_rls = df_rls.copy()

df_resultados_rls["defunciones_predichas"] = modelo_rls.fittedvalues
df_resultados_rls["residuos"] = modelo_rls.resid

display(df_resultados_rls.head())


# ------------------------------------------------------------
# 6.3. Guardar tablas en Excel
# ------------------------------------------------------------

ruta_excel_rls = OUT_DIR_2 / "resultados_regresion_lineal_simple.xlsx"

with pd.ExcelWriter(ruta_excel_rls, engine="openpyxl") as writer:
    tabla_rls.to_excel(writer, sheet_name="coeficientes")
    metricas_rls.to_excel(writer, sheet_name="metricas_modelo", index=False)
    df_resultados_rls.to_excel(writer, sheet_name="predicciones_residuos", index=False)
    df_rls.to_excel(writer, sheet_name="datos_modelo", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_rls}")


# ------------------------------------------------------------
# 6.4. Guardar resumen completo del modelo en TXT
# ------------------------------------------------------------

ruta_txt_rls = OUT_DIR_2 / "summary_regresion_lineal_simple.txt"

with open(ruta_txt_rls, "w", encoding="utf-8") as f:
    f.write(str(modelo_rls.summary()))

print(f"Resumen del modelo guardado en: {ruta_txt_rls}")



# %% [notebook cell 10]
# ------------------------------------------------------------
# 6.5. Guardar gráfica de regresión
# ------------------------------------------------------------

ruta_grafica_rls = OUT_DIR_2 / "regresion_simple_defunciones_hospitalizaciones.png"

plt.figure(figsize=(8, 5))

plt.scatter(
    df_rls["hospitalizaciones_diarias"],
    df_rls["defunciones_diarias"],
    alpha=0.6,
    label="Observaciones diarias"
)

x_grid = np.linspace(
    df_rls["hospitalizaciones_diarias"].min(),
    df_rls["hospitalizaciones_diarias"].max(),
    100
)

X_grid = sm.add_constant(pd.DataFrame({"hospitalizaciones_diarias": x_grid}))
predicciones = modelo_rls.predict(X_grid)

plt.plot(x_grid, predicciones, linewidth=2, label="Línea de regresión")

plt.title("Regresión lineal simple: defunciones diarias y hospitalizaciones diarias")
plt.xlabel("Hospitalizaciones diarias")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(ruta_grafica_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_grafica_rls}")


# ------------------------------------------------------------
# 6.6. Resumen para interpretar en el reporte
# ------------------------------------------------------------

coef_hosp = modelo_rls.params["hospitalizaciones_diarias"]
p_hosp = modelo_rls.pvalues["hospitalizaciones_diarias"]
ic_inf = modelo_rls.conf_int().loc["hospitalizaciones_diarias", 0]
ic_sup = modelo_rls.conf_int().loc["hospitalizaciones_diarias", 1]
r2 = modelo_rls.rsquared

print("Resumen para el reporte")
print("---------------------------------------------")
print(f"Coeficiente de hospitalizaciones diarias: {coef_hosp:.6f}")
print(f"P-valor: {p_hosp:.6f}")
print(f"Intervalo de confianza 95%: [{ic_inf:.6f}, {ic_sup:.6f}]")
print(f"R cuadrada: {r2:.6f}")

if p_hosp < 0.05:
    print("\nLectura técnica:")
    print(
        f"El coeficiente es estadísticamente significativo al 5%. "
        f"Por cada hospitalización diaria adicional, las defunciones diarias "
        f"cambian en promedio {coef_hosp:.4f} unidades."
    )
else:
    print("\nLectura técnica:")
    print(
        f"El coeficiente no es estadísticamente significativo al 5%. "
        f"No se cuenta con evidencia suficiente para afirmar que las hospitalizaciones "
        f"diarias explican las defunciones diarias en este modelo simple."
    )
