"""
REGRESIÓN LINEAL MÚLTIPLE
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 3. REGRESIÓN LINEAL MÚLTIPLE**
######################################################################



# %% [notebook cell 12]
# ============================================================
# ETAPA 3. REGRESIÓN LINEAL MÚLTIPLE
# Modelo: Defunciones diarias en función de hospitalizaciones
# y casos diarios
# ============================================================

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_3 = Path("resultados_etapa_3")
OUT_DIR_3.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar datos del modelo
# ------------------------------------------------------------

variables_rlm = [
    "defunciones_diarias",
    "hospitalizaciones_diarias",
    "casos_diarios"
]

df_rlm = df_ts[variables_rlm].copy()

for col in variables_rlm:
    df_rlm[col] = pd.to_numeric(df_rlm[col], errors="coerce")

df_rlm = df_rlm.dropna()

print("Observaciones utilizadas en el modelo:", len(df_rlm))
display(df_rlm.head())


# ------------------------------------------------------------
# 2. Estimar regresión lineal múltiple
# ------------------------------------------------------------

Y = df_rlm["defunciones_diarias"]

X = df_rlm[[
    "hospitalizaciones_diarias",
    "casos_diarios"
]]

X = sm.add_constant(X)

modelo_rlm = sm.OLS(Y, X).fit()

print(modelo_rlm.summary())


# ------------------------------------------------------------
# 3. Tabla ordenada de coeficientes
# ------------------------------------------------------------

tabla_rlm = pd.DataFrame({
    "coeficiente": modelo_rlm.params,
    "error_estandar": modelo_rlm.bse,
    "t": modelo_rlm.tvalues,
    "p_valor": modelo_rlm.pvalues,
    "IC_95_inf": modelo_rlm.conf_int()[0],
    "IC_95_sup": modelo_rlm.conf_int()[1]
})

print("Resultados de la regresión lineal múltiple")
display(tabla_rlm)


# ------------------------------------------------------------
# 4. Métricas generales del modelo
# ------------------------------------------------------------

metricas_rlm = pd.DataFrame({
    "indicador": [
        "Variable dependiente",
        "Variables explicativas",
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
        "hospitalizaciones_diarias, casos_diarios",
        int(modelo_rlm.nobs),
        modelo_rlm.rsquared,
        modelo_rlm.rsquared_adj,
        modelo_rlm.fvalue,
        modelo_rlm.f_pvalue,
        modelo_rlm.aic,
        modelo_rlm.bic
    ]
})

print("Métricas generales del modelo")
display(metricas_rlm)


# ------------------------------------------------------------
# 5. Predicciones y residuos
# ------------------------------------------------------------

df_resultados_rlm = df_rlm.copy()
df_resultados_rlm["defunciones_predichas"] = modelo_rlm.fittedvalues
df_resultados_rlm["residuos"] = modelo_rlm.resid

display(df_resultados_rlm.head())


# ------------------------------------------------------------
# 6. Signos esperados y resultados observados
# ------------------------------------------------------------

signos_esperados = pd.DataFrame({
    "variable": [
        "hospitalizaciones_diarias",
        "casos_diarios"
    ],
    "signo_esperado": [
        "Positivo",
        "Positivo"
    ],
    "justificacion": [
        "Un mayor número de hospitalizaciones refleja mayor gravedad clínica y presión hospitalaria, por lo que se espera una asociación positiva con las defunciones.",
        "Un mayor número de casos confirmados puede anticipar o coincidir con mayor mortalidad, aunque su efecto puede observarse con rezago temporal."
    ],
    "coeficiente_estimado": [
        modelo_rlm.params.get("hospitalizaciones_diarias", np.nan),
        modelo_rlm.params.get("casos_diarios", np.nan)
    ],
    "signo_observado": [
        "Positivo" if modelo_rlm.params.get("hospitalizaciones_diarias", np.nan) > 0 else "Negativo",
        "Positivo" if modelo_rlm.params.get("casos_diarios", np.nan) > 0 else "Negativo"
    ],
    "p_valor": [
        modelo_rlm.pvalues.get("hospitalizaciones_diarias", np.nan),
        modelo_rlm.pvalues.get("casos_diarios", np.nan)
    ]
})

display(signos_esperados)



# %% [notebook cell 13]
# ------------------------------------------------------------
# 7. Gráfica: defunciones reales vs predichas
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df_resultados_rlm.index,
    df_resultados_rlm["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_resultados_rlm.index,
    df_resultados_rlm["defunciones_predichas"],
    label="Defunciones predichas"
)

plt.title("Regresión lineal múltiple: defunciones observadas vs predichas")
plt.xlabel("Observación")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_grafica_pred = OUT_DIR_3 / "rlm_defunciones_observadas_vs_predichas.png"
plt.savefig(ruta_grafica_pred, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_grafica_pred}")


# ------------------------------------------------------------
# 8. Gráfica de residuos
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    df_resultados_rlm["defunciones_predichas"],
    df_resultados_rlm["residuos"],
    alpha=0.7
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Residuos vs valores predichos - Regresión lineal múltiple")
plt.xlabel("Defunciones predichas")
plt.ylabel("Residuos")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_grafica_residuos = OUT_DIR_3 / "rlm_residuos_vs_predichos.png"
plt.savefig(ruta_grafica_residuos, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_grafica_residuos}")


# ------------------------------------------------------------
# 9. Guardar resultados en Excel y TXT
# ------------------------------------------------------------

ruta_excel_rlm = OUT_DIR_3 / "resultados_regresion_lineal_multiple.xlsx"

with pd.ExcelWriter(ruta_excel_rlm, engine="openpyxl") as writer:
    tabla_rlm.to_excel(writer, sheet_name="coeficientes")
    metricas_rlm.to_excel(writer, sheet_name="metricas_modelo", index=False)
    signos_esperados.to_excel(writer, sheet_name="signos_esperados", index=False)
    df_resultados_rlm.to_excel(writer, sheet_name="predicciones_residuos", index=False)
    df_rlm.to_excel(writer, sheet_name="datos_modelo", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_rlm}")


ruta_txt_rlm = OUT_DIR_3 / "summary_regresion_lineal_multiple.txt"

with open(ruta_txt_rlm, "w", encoding="utf-8") as f:
    f.write(str(modelo_rlm.summary()))

print(f"Resumen del modelo guardado en: {ruta_txt_rlm}")


# ------------------------------------------------------------
# 10. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

for var in ["hospitalizaciones_diarias", "casos_diarios"]:
    coef = modelo_rlm.params[var]
    pvalor = modelo_rlm.pvalues[var]
    ic_inf = modelo_rlm.conf_int().loc[var, 0]
    ic_sup = modelo_rlm.conf_int().loc[var, 1]
    
    print(f"\nVariable: {var}")
    print(f"Coeficiente: {coef:.6f}")
    print(f"P-valor: {pvalor:.6f}")
    print(f"Intervalo de confianza 95%: [{ic_inf:.6f}, {ic_sup:.6f}]")
    
    if pvalor < 0.05:
        print("Lectura técnica: estadísticamente significativa al 5%.")
    else:
        print("Lectura técnica: no estadísticamente significativa al 5%.")

print("\nBondad de ajuste")
print(f"R cuadrada: {modelo_rlm.rsquared:.6f}")
print(f"R cuadrada ajustada: {modelo_rlm.rsquared_adj:.6f}")
print(f"Estadístico F: {modelo_rlm.fvalue:.6f}")
print(f"P-valor F: {modelo_rlm.f_pvalue:.6f}")

if modelo_rlm.f_pvalue < 0.05:
    print("El modelo es globalmente significativo al 5%.")
else:
    print("El modelo no es globalmente significativo al 5%.")
