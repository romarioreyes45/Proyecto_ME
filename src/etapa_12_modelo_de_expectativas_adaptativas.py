"""
MODELO DE EXPECTATIVAS ADAPTATIVAS
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 12. MODELO DE EXPECTATIVAS ADAPTATIVAS**
######################################################################



# %% [notebook cell 49]
# ============================================================
# ETAPA 12. MODELO DE EXPECTATIVAS ADAPTATIVAS
# Casos diarios, hospitalizaciones diarias y defunciones diarias
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_12 = Path("resultados_etapa_12")
OUT_DIR_12.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar serie temporal
# ------------------------------------------------------------

df_exp = df_ts.copy()

if "fecha" in df_exp.columns:
    df_exp["fecha"] = pd.to_datetime(df_exp["fecha"], errors="coerce")
    df_exp = df_exp.sort_values("fecha")
else:
    df_exp = df_exp.sort_index()

variables_expectativas = [
    "casos_diarios",
    "hospitalizaciones_diarias",
    "defunciones_diarias"
]

for col in variables_expectativas:
    df_exp[col] = pd.to_numeric(df_exp[col], errors="coerce")

df_exp = df_exp.dropna(subset=variables_expectativas).reset_index(drop=True)

print("Observaciones disponibles:", len(df_exp))
display(df_exp.head())


# ------------------------------------------------------------
# 2. Funciones para expectativas adaptativas
# ------------------------------------------------------------

def calcular_expectativa_adaptativa(serie, lambda_ajuste):
    """
    Calcula expectativas adaptativas:
    E_t = lambda * Y_{t-1} + (1 - lambda) * E_{t-1}

    La primera expectativa se inicializa con el primer valor observado.
    """
    serie = pd.Series(serie).reset_index(drop=True)
    expectativas = np.zeros(len(serie))

    expectativas[0] = serie.iloc[0]

    for t in range(1, len(serie)):
        expectativas[t] = (
            lambda_ajuste * serie.iloc[t - 1] +
            (1 - lambda_ajuste) * expectativas[t - 1]
        )

    return pd.Series(expectativas)


def calcular_metricas_error(y_real, y_pred):
    """
    Calcula métricas de error.
    """
    y_real = pd.Series(y_real).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    error = y_real - y_pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    sesgo = np.mean(error)

    # MAPE evitando división entre cero
    mascara = y_real != 0
    if mascara.sum() > 0:
        mape = np.mean(np.abs((y_real[mascara] - y_pred[mascara]) / y_real[mascara])) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sesgo_promedio": sesgo
    }


def evaluar_lambdas(df_base, variable, lambdas):
    """
    Evalúa distintos valores de lambda y selecciona el mejor según RMSE.
    """
    resultados = []

    y_real = df_base[variable]

    for lam in lambdas:
        expectativa = calcular_expectativa_adaptativa(y_real, lam)
        metricas = calcular_metricas_error(y_real, expectativa)

        resultados.append({
            "variable": variable,
            "lambda": lam,
            "peso_dato_observado_anterior": lam,
            "peso_expectativa_pasada": 1 - lam,
            **metricas
        })

    tabla = pd.DataFrame(resultados)
    tabla = tabla.sort_values("RMSE").reset_index(drop=True)

    return tabla


def estimar_modelo_expectativa(df_base, variable, expectativa_col):
    """
    Estima:
    Y_t = alpha + beta * E_t + u_t
    """
    df_modelo = df_base[[variable, expectativa_col]].dropna().copy()

    Y = df_modelo[variable]
    X = df_modelo[[expectativa_col]]
    X = sm.add_constant(X)

    modelo = sm.OLS(Y, X).fit()

    return modelo, df_modelo


# ------------------------------------------------------------
# 3. Evaluar valores de lambda
# ------------------------------------------------------------

lambdas = np.round(np.arange(0.1, 1.0, 0.1), 2)

resultados_lambdas = []

for variable in variables_expectativas:
    tabla_var = evaluar_lambdas(df_exp, variable, lambdas)
    resultados_lambdas.append(tabla_var)

resultados_lambdas = pd.concat(resultados_lambdas, ignore_index=True)

print("Evaluación de lambdas")
display(resultados_lambdas)



# %% [notebook cell 50]
# ------------------------------------------------------------
# 4. Seleccionar mejor lambda por variable
# ------------------------------------------------------------

mejores_lambdas = (
    resultados_lambdas
    .sort_values("RMSE")
    .groupby("variable")
    .first()
    .reset_index()
)

print("Mejores lambdas según RMSE")
display(mejores_lambdas)


# ------------------------------------------------------------
# 5. Construir expectativas adaptativas finales
# ------------------------------------------------------------

df_expectativas_final = df_exp.copy()

for _, row in mejores_lambdas.iterrows():
    variable = row["variable"]
    lam = row["lambda"]

    col_expectativa = f"expectativa_{variable}"
    col_error = f"error_{variable}"

    df_expectativas_final[col_expectativa] = calcular_expectativa_adaptativa(
        df_expectativas_final[variable],
        lam
    )

    df_expectativas_final[col_error] = (
        df_expectativas_final[variable] -
        df_expectativas_final[col_expectativa]
    )

display(df_expectativas_final.head())


# ------------------------------------------------------------
# 6. Estimar modelos: observado_t ~ expectativa_t
# ------------------------------------------------------------

tablas_coeficientes_exp = []
metricas_modelos_exp = []
modelos_exp = {}
datos_modelos_exp = {}

for variable in variables_expectativas:
    col_expectativa = f"expectativa_{variable}"

    modelo, datos_modelo = estimar_modelo_expectativa(
        df_expectativas_final,
        variable,
        col_expectativa
    )

    modelos_exp[variable] = modelo
    datos_modelos_exp[variable] = datos_modelo

    tabla_coef = pd.DataFrame({
        "variable_modelada": variable,
        "variable": modelo.params.index,
        "coeficiente": modelo.params.values,
        "error_estandar": modelo.bse.values,
        "t": modelo.tvalues.values,
        "p_valor": modelo.pvalues.values,
        "IC_95_inf": modelo.conf_int()[0].values,
        "IC_95_sup": modelo.conf_int()[1].values
    })

    tablas_coeficientes_exp.append(tabla_coef)

    metricas_modelos_exp.append({
        "variable_modelada": variable,
        "observaciones": int(modelo.nobs),
        "r_cuadrada": modelo.rsquared,
        "r_cuadrada_ajustada": modelo.rsquared_adj,
        "estadistico_f": modelo.fvalue,
        "p_valor_f": modelo.f_pvalue,
        "aic": modelo.aic,
        "bic": modelo.bic
    })

tabla_coeficientes_exp = pd.concat(tablas_coeficientes_exp, ignore_index=True)
metricas_modelos_exp = pd.DataFrame(metricas_modelos_exp)

print("Coeficientes de modelos con expectativas adaptativas")
display(tabla_coeficientes_exp)

print("Métricas de modelos con expectativas adaptativas")
display(metricas_modelos_exp)


# ------------------------------------------------------------
# 7. Predicciones y residuos de los modelos
# ------------------------------------------------------------

predicciones_exp = df_expectativas_final.copy()

for variable in variables_expectativas:
    modelo = modelos_exp[variable]
    col_expectativa = f"expectativa_{variable}"
    col_pred = f"pred_modelo_{variable}"
    col_resid = f"residuo_modelo_{variable}"

    X_pred = sm.add_constant(predicciones_exp[[col_expectativa]], has_constant="add")
    predicciones_exp[col_pred] = modelo.predict(X_pred)
    predicciones_exp[col_resid] = predicciones_exp[variable] - predicciones_exp[col_pred]

display(predicciones_exp.head())


# ------------------------------------------------------------
# 8. Tabla de formación de expectativas
# ------------------------------------------------------------

formacion_expectativas = mejores_lambdas.copy()

formacion_expectativas["peso_expectativa_pasada"] = 1 - formacion_expectativas["lambda"]

formacion_expectativas["interpretacion"] = formacion_expectativas.apply(
    lambda row: (
        f"Para {row['variable']}, lambda = {row['lambda']:.2f}. "
        f"Esto significa que la expectativa actual asigna {row['lambda']*100:.1f}% "
        f"de peso al dato observado anterior y {(1-row['lambda'])*100:.1f}% "
        f"de peso a la expectativa pasada."
    ),
    axis=1
)

print("Formación de expectativas")
display(formacion_expectativas)



# %% [notebook cell 51]
# ------------------------------------------------------------
# 9. Gráfica: casos observados vs esperados
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["casos_diarios"],
    label="Casos observados"
)

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["expectativa_casos_diarios"],
    label="Casos esperados"
)

plt.title("Expectativas adaptativas: casos diarios")
plt.xlabel("Observación temporal")
plt.ylabel("Casos diarios")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_exp_casos = OUT_DIR_12 / "expectativas_adaptativas_casos.png"
plt.savefig(ruta_exp_casos, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_exp_casos}")


# ------------------------------------------------------------
# 10. Gráfica: hospitalizaciones observadas vs esperadas
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["hospitalizaciones_diarias"],
    label="Hospitalizaciones observadas"
)

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["expectativa_hospitalizaciones_diarias"],
    label="Hospitalizaciones esperadas"
)

plt.title("Expectativas adaptativas: hospitalizaciones diarias")
plt.xlabel("Observación temporal")
plt.ylabel("Hospitalizaciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_exp_hosp = OUT_DIR_12 / "expectativas_adaptativas_hospitalizaciones.png"
plt.savefig(ruta_exp_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_exp_hosp}")


# ------------------------------------------------------------
# 11. Gráfica: defunciones observadas vs esperadas
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["defunciones_diarias"],
    label="Defunciones observadas"
)

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["expectativa_defunciones_diarias"],
    label="Defunciones esperadas"
)

plt.title("Expectativas adaptativas: defunciones diarias")
plt.xlabel("Observación temporal")
plt.ylabel("Defunciones diarias")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_exp_def = OUT_DIR_12 / "expectativas_adaptativas_defunciones.png"
plt.savefig(ruta_exp_def, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_exp_def}")


# ------------------------------------------------------------
# 12. Gráfica de errores de predicción
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["error_casos_diarios"],
    label="Error casos"
)

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["error_hospitalizaciones_diarias"],
    label="Error hospitalizaciones"
)

plt.plot(
    df_expectativas_final.index,
    df_expectativas_final["error_defunciones_diarias"],
    label="Error defunciones"
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Errores de predicción en expectativas adaptativas")
plt.xlabel("Observación temporal")
plt.ylabel("Error observado - esperado")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_errores_exp = OUT_DIR_12 / "errores_expectativas_adaptativas.png"
plt.savefig(ruta_errores_exp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_errores_exp}")



# %% [notebook cell 52]
# ------------------------------------------------------------
# 13. Gráfica de mejores lambdas
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    mejores_lambdas["variable"],
    mejores_lambdas["lambda"]
)

plt.ylim(0, 1)
plt.title("Mejor parámetro lambda por variable")
plt.xlabel("Variable")
plt.ylabel("Lambda")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_lambdas = OUT_DIR_12 / "mejores_lambdas_expectativas.png"
plt.savefig(ruta_lambdas, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_lambdas}")


# ------------------------------------------------------------
# 14. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_exp = OUT_DIR_12 / "resultados_expectativas_adaptativas.xlsx"

with pd.ExcelWriter(ruta_excel_exp, engine="openpyxl") as writer:
    resultados_lambdas.to_excel(writer, sheet_name="evaluacion_lambdas", index=False)
    mejores_lambdas.to_excel(writer, sheet_name="mejores_lambdas", index=False)
    formacion_expectativas.to_excel(writer, sheet_name="formacion_expectativas", index=False)
    tabla_coeficientes_exp.to_excel(writer, sheet_name="coeficientes_modelos", index=False)
    metricas_modelos_exp.to_excel(writer, sheet_name="metricas_modelos", index=False)
    df_expectativas_final.to_excel(writer, sheet_name="expectativas_errores", index=False)
    predicciones_exp.to_excel(writer, sheet_name="predicciones_residuos", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_exp}")


# ------------------------------------------------------------
# 15. Guardar resumen en TXT
# ------------------------------------------------------------

ruta_txt_exp = OUT_DIR_12 / "resumen_expectativas_adaptativas.txt"

with open(ruta_txt_exp, "w", encoding="utf-8") as f:
    f.write("ETAPA 12. MODELO DE EXPECTATIVAS ADAPTATIVAS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Fórmula utilizada:\n")
    f.write("E_t = lambda * Y_{t-1} + (1 - lambda) * E_{t-1}\n\n")

    f.write("Mejores lambdas:\n")
    f.write(mejores_lambdas.to_string(index=False))
    f.write("\n\n")

    f.write("Formación de expectativas:\n")
    f.write(formacion_expectativas[["variable", "lambda", "peso_expectativa_pasada", "interpretacion"]].to_string(index=False))
    f.write("\n\n")

    f.write("Coeficientes de modelos:\n")
    f.write(tabla_coeficientes_exp.to_string(index=False))
    f.write("\n\n")

    f.write("Métricas de modelos:\n")
    f.write(metricas_modelos_exp.to_string(index=False))
    f.write("\n\n")

    for variable in variables_expectativas:
        f.write("=" * 70 + "\n")
        f.write(f"Resumen modelo para {variable}:\n")
        f.write(str(modelos_exp[variable].summary()))
        f.write("\n\n")

print(f"Resumen TXT guardado en: {ruta_txt_exp}")


# ------------------------------------------------------------
# 16. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nMejores lambdas")
display(mejores_lambdas)

print("\nFormación de expectativas")
display(formacion_expectativas[["variable", "lambda", "peso_expectativa_pasada", "interpretacion"]])

print("\nCoeficientes de modelos con expectativas")
display(tabla_coeficientes_exp)

print("\nMétricas de modelos")
display(metricas_modelos_exp)
