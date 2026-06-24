"""
MODELO DE AJUSTE PARCIAL
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 11. MODELO DE AJUSTE PARCIAL**
######################################################################



# %% [notebook cell 44]
# ============================================================
# ETAPA 11. MODELO DE AJUSTE PARCIAL
# Defunciones diarias y hospitalizaciones diarias
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_11 = Path("resultados_etapa_11")
OUT_DIR_11.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar serie temporal
# ------------------------------------------------------------

df_ajuste = df_ts.copy()

if "fecha" in df_ajuste.columns:
    df_ajuste["fecha"] = pd.to_datetime(df_ajuste["fecha"], errors="coerce")
    df_ajuste = df_ajuste.sort_values("fecha")
else:
    df_ajuste = df_ajuste.sort_index()

variables_ajuste = [
    "defunciones_diarias",
    "hospitalizaciones_diarias",
    "casos_diarios"
]

for col in variables_ajuste:
    if col in df_ajuste.columns:
        df_ajuste[col] = pd.to_numeric(df_ajuste[col], errors="coerce")

# Crear rezago de la variable dependiente
df_ajuste["defunciones_lag_1"] = df_ajuste["defunciones_diarias"].shift(1)

display(df_ajuste.head())


# ------------------------------------------------------------
# 2. Modelo principal de ajuste parcial
# Defunciones_t ~ Hospitalizaciones_t + Defunciones_t-1
# ------------------------------------------------------------

variables_modelo_ajuste = [
    "hospitalizaciones_diarias",
    "defunciones_lag_1"
]

df_modelo_ajuste = df_ajuste[
    ["defunciones_diarias"] + variables_modelo_ajuste
].dropna().copy()

Y_ajuste = df_modelo_ajuste["defunciones_diarias"]
X_ajuste = df_modelo_ajuste[variables_modelo_ajuste]
X_ajuste = sm.add_constant(X_ajuste)

modelo_ajuste_parcial = sm.OLS(Y_ajuste, X_ajuste).fit()

print(modelo_ajuste_parcial.summary())


# ------------------------------------------------------------
# 3. Modelo complementario
# Defunciones_t ~ Casos_t + Defunciones_t-1
# ------------------------------------------------------------

variables_modelo_ajuste_casos = [
    "casos_diarios",
    "defunciones_lag_1"
]

df_modelo_ajuste_casos = df_ajuste[
    ["defunciones_diarias"] + variables_modelo_ajuste_casos
].dropna().copy()

Y_ajuste_casos = df_modelo_ajuste_casos["defunciones_diarias"]
X_ajuste_casos = df_modelo_ajuste_casos[variables_modelo_ajuste_casos]
X_ajuste_casos = sm.add_constant(X_ajuste_casos)

modelo_ajuste_parcial_casos = sm.OLS(Y_ajuste_casos, X_ajuste_casos).fit()

print(modelo_ajuste_parcial_casos.summary())


# ------------------------------------------------------------
# 4. Función para tabla de coeficientes
# ------------------------------------------------------------

def tabla_coeficientes_ajuste(modelo, nombre_modelo):
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


tabla_ajuste_hosp = tabla_coeficientes_ajuste(
    modelo_ajuste_parcial,
    "Ajuste parcial con hospitalizaciones"
)

tabla_ajuste_casos = tabla_coeficientes_ajuste(
    modelo_ajuste_parcial_casos,
    "Ajuste parcial con casos"
)

print("Modelo de ajuste parcial con hospitalizaciones")
display(tabla_ajuste_hosp)

print("Modelo de ajuste parcial con casos")
display(tabla_ajuste_casos)



# %% [notebook cell 45]
# ------------------------------------------------------------
# 5. Métricas generales de los modelos
# ------------------------------------------------------------

metricas_ajuste = pd.DataFrame({
    "modelo": [
        "Ajuste parcial con hospitalizaciones",
        "Ajuste parcial con casos"
    ],
    "variable_dependiente": [
        "defunciones_diarias",
        "defunciones_diarias"
    ],
    "observaciones": [
        int(modelo_ajuste_parcial.nobs),
        int(modelo_ajuste_parcial_casos.nobs)
    ],
    "r_cuadrada": [
        modelo_ajuste_parcial.rsquared,
        modelo_ajuste_parcial_casos.rsquared
    ],
    "r_cuadrada_ajustada": [
        modelo_ajuste_parcial.rsquared_adj,
        modelo_ajuste_parcial_casos.rsquared_adj
    ],
    "estadistico_f": [
        modelo_ajuste_parcial.fvalue,
        modelo_ajuste_parcial_casos.fvalue
    ],
    "p_valor_f": [
        modelo_ajuste_parcial.f_pvalue,
        modelo_ajuste_parcial_casos.f_pvalue
    ],
    "aic": [
        modelo_ajuste_parcial.aic,
        modelo_ajuste_parcial_casos.aic
    ],
    "bic": [
        modelo_ajuste_parcial.bic,
        modelo_ajuste_parcial_casos.bic
    ]
})

display(metricas_ajuste)


# ------------------------------------------------------------
# 6. Cálculo de velocidad de ajuste y largo plazo
# ------------------------------------------------------------

def calcular_parametros_ajuste(modelo, variable_explicativa, nombre_modelo):
    """
    En el modelo:
    Y_t = alpha + beta X_t + rho Y_{t-1} + u_t

    rho = coeficiente de Y_{t-1}
    lambda = 1 - rho
    beta_largo_plazo = beta / lambda
    intercepto_largo_plazo = alpha / lambda
    """
    
    alpha = modelo.params.get("const", np.nan)
    beta_corto_plazo = modelo.params.get(variable_explicativa, np.nan)
    rho = modelo.params.get("defunciones_lag_1", np.nan)
    
    velocidad_ajuste = 1 - rho
    persistencia_temporal = rho
    
    if velocidad_ajuste != 0:
        beta_largo_plazo = beta_corto_plazo / velocidad_ajuste
        intercepto_largo_plazo = alpha / velocidad_ajuste
    else:
        beta_largo_plazo = np.nan
        intercepto_largo_plazo = np.nan
    
    return pd.DataFrame({
        "modelo": [nombre_modelo],
        "variable_explicativa": [variable_explicativa],
        "intercepto_corto_plazo": [alpha],
        "beta_corto_plazo": [beta_corto_plazo],
        "rho_persistencia": [rho],
        "velocidad_ajuste_lambda": [velocidad_ajuste],
        "beta_largo_plazo": [beta_largo_plazo],
        "intercepto_largo_plazo": [intercepto_largo_plazo],
        "interpretacion_velocidad": [
            interpretar_velocidad_ajuste(velocidad_ajuste)
        ]
    })


def interpretar_velocidad_ajuste(lambda_val):
    if pd.isna(lambda_val):
        return "No calculable"
    elif lambda_val < 0:
        return "Ajuste inestable: revisar especificación del modelo"
    elif lambda_val < 0.30:
        return "Ajuste lento: alta persistencia temporal"
    elif lambda_val < 0.70:
        return "Ajuste moderado"
    elif lambda_val <= 1:
        return "Ajuste rápido: baja persistencia temporal"
    else:
        return "Sobreajuste o ajuste mayor a uno: revisar especificación"


parametros_ajuste_hosp = calcular_parametros_ajuste(
    modelo_ajuste_parcial,
    "hospitalizaciones_diarias",
    "Ajuste parcial con hospitalizaciones"
)

parametros_ajuste_casos = calcular_parametros_ajuste(
    modelo_ajuste_parcial_casos,
    "casos_diarios",
    "Ajuste parcial con casos"
)

parametros_ajuste_total = pd.concat(
    [parametros_ajuste_hosp, parametros_ajuste_casos],
    ignore_index=True
)

display(parametros_ajuste_total)


# ------------------------------------------------------------
# 7. Predicciones y residuos
# ------------------------------------------------------------

df_pred_ajuste_hosp = df_modelo_ajuste.copy()
df_pred_ajuste_hosp["defunciones_predichas"] = modelo_ajuste_parcial.fittedvalues
df_pred_ajuste_hosp["residuos"] = modelo_ajuste_parcial.resid

df_pred_ajuste_casos = df_modelo_ajuste_casos.copy()
df_pred_ajuste_casos["defunciones_predichas"] = modelo_ajuste_parcial_casos.fittedvalues
df_pred_ajuste_casos["residuos"] = modelo_ajuste_parcial_casos.resid

display(df_pred_ajuste_hosp.head())
display(df_pred_ajuste_casos.head())



# %% [notebook cell 46]
# ------------------------------------------------------------
# 8. Calcular nivel de equilibrio de largo plazo
# ------------------------------------------------------------

def agregar_equilibrio_largo_plazo(df_pred, parametros, variable_explicativa):
    """
    Equilibrio de largo plazo:
    Y* = intercepto_largo_plazo + beta_largo_plazo * X_t
    """
    df_out = df_pred.copy()
    
    intercepto_lp = parametros["intercepto_largo_plazo"].iloc[0]
    beta_lp = parametros["beta_largo_plazo"].iloc[0]
    
    df_out["equilibrio_largo_plazo"] = (
        intercepto_lp + beta_lp * df_out[variable_explicativa]
    )
    
    df_out["brecha_ajuste"] = (
        df_out["equilibrio_largo_plazo"] -
        df_out["defunciones_diarias"]
    )
    
    return df_out


df_pred_ajuste_hosp = agregar_equilibrio_largo_plazo(
    df_pred_ajuste_hosp,
    parametros_ajuste_hosp,
    "hospitalizaciones_diarias"
)

df_pred_ajuste_casos = agregar_equilibrio_largo_plazo(
    df_pred_ajuste_casos,
    parametros_ajuste_casos,
    "casos_diarios"
)

display(df_pred_ajuste_hosp.head())
display(df_pred_ajuste_casos.head())


# ------------------------------------------------------------
# 9. Gráfica observadas vs predichas - hospitalizaciones
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_pred_ajuste_hosp.index,
    df_pred_ajuste_hosp["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_pred_ajuste_hosp.index,
    df_pred_ajuste_hosp["defunciones_predichas"],
    label="Defunciones predichas"
)

plt.title("Modelo de ajuste parcial con hospitalizaciones")
plt.xlabel("Observación temporal")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_ajuste_hosp = OUT_DIR_11 / "ajuste_parcial_observadas_predichas_hosp.png"
plt.savefig(ruta_ajuste_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_ajuste_hosp}")


# ------------------------------------------------------------
# 10. Gráfica observadas vs equilibrio de largo plazo - hospitalizaciones
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_pred_ajuste_hosp.index,
    df_pred_ajuste_hosp["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_pred_ajuste_hosp.index,
    df_pred_ajuste_hosp["equilibrio_largo_plazo"],
    label="Equilibrio de largo plazo"
)

plt.title("Defunciones observadas y equilibrio de largo plazo")
plt.xlabel("Observación temporal")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_equilibrio_hosp = OUT_DIR_11 / "ajuste_parcial_equilibrio_largo_plazo_hosp.png"
plt.savefig(ruta_equilibrio_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_equilibrio_hosp}")


# ------------------------------------------------------------
# 11. Gráfica de brecha de ajuste - hospitalizaciones
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_pred_ajuste_hosp.index,
    df_pred_ajuste_hosp["brecha_ajuste"]
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Brecha de ajuste hacia el equilibrio de largo plazo")
plt.xlabel("Observación temporal")
plt.ylabel("Equilibrio de largo plazo - defunciones observadas")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_brecha_hosp = OUT_DIR_11 / "ajuste_parcial_brecha_hosp.png"
plt.savefig(ruta_brecha_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_brecha_hosp}")


# ------------------------------------------------------------
# 12. Comparación de velocidad de ajuste
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    parametros_ajuste_total["modelo"],
    parametros_ajuste_total["velocidad_ajuste_lambda"]
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(1, linestyle="--", linewidth=1)

plt.title("Velocidad de ajuste estimada")
plt.xlabel("Modelo")
plt.ylabel("Lambda")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_lambda = OUT_DIR_11 / "velocidad_ajuste_lambda.png"
plt.savefig(ruta_lambda, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_lambda}")



# %% [notebook cell 47]
# ------------------------------------------------------------
# 13. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_ajuste = OUT_DIR_11 / "resultados_modelo_ajuste_parcial.xlsx"

with pd.ExcelWriter(ruta_excel_ajuste, engine="openpyxl") as writer:
    tabla_ajuste_hosp.to_excel(writer, sheet_name="coef_ajuste_hosp", index=False)
    tabla_ajuste_casos.to_excel(writer, sheet_name="coef_ajuste_casos", index=False)
    metricas_ajuste.to_excel(writer, sheet_name="metricas_modelos", index=False)
    parametros_ajuste_total.to_excel(writer, sheet_name="velocidad_equilibrio", index=False)
    df_pred_ajuste_hosp.to_excel(writer, sheet_name="pred_equilibrio_hosp", index=False)
    df_pred_ajuste_casos.to_excel(writer, sheet_name="pred_equilibrio_casos", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_ajuste}")


# ------------------------------------------------------------
# 14. Guardar resúmenes en TXT
# ------------------------------------------------------------

ruta_txt_ajuste = OUT_DIR_11 / "resumen_modelo_ajuste_parcial.txt"

with open(ruta_txt_ajuste, "w", encoding="utf-8") as f:
    f.write("ETAPA 11. MODELO DE AJUSTE PARCIAL\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Modelo principal: defunciones diarias explicadas por hospitalizaciones diarias y defunciones rezagadas.\n")
    f.write(str(modelo_ajuste_parcial.summary()))
    f.write("\n\n")
    
    f.write("Modelo complementario: defunciones diarias explicadas por casos diarios y defunciones rezagadas.\n")
    f.write(str(modelo_ajuste_parcial_casos.summary()))
    f.write("\n\n")
    
    f.write("Parámetros de ajuste:\n")
    f.write(parametros_ajuste_total.to_string(index=False))

print(f"Resumen TXT guardado en: {ruta_txt_ajuste}")


# ------------------------------------------------------------
# 15. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nCoeficientes del modelo de ajuste parcial con hospitalizaciones")
display(tabla_ajuste_hosp)

print("\nCoeficientes del modelo de ajuste parcial con casos")
display(tabla_ajuste_casos)

print("\nVelocidad de ajuste, persistencia y largo plazo")
display(parametros_ajuste_total)

print("\nMétricas de comparación")
display(metricas_ajuste)
