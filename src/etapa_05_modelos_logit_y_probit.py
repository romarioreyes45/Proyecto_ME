"""
MODELOS LOGIT Y PROBIT
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 5. MODELOS LOGIT Y PROBIT**
######################################################################



# %% [notebook cell 19]
# ============================================================
# ETAPA 5. MODELOS LOGIT Y PROBIT
# Variable dependiente: Defunción
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_5 = Path("resultados_etapa_5")
OUT_DIR_5.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar base de trabajo
# ------------------------------------------------------------

df_lp = df_covid.copy()

# Variables sugeridas por el proyecto
variable_dependiente = "defuncion"

variables_predictoras = [
    "edad",
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion"
]

# Verificar variables disponibles
variables_predictoras = [v for v in variables_predictoras if v in df_lp.columns]

columnas_modelo = [variable_dependiente] + variables_predictoras

df_modelo_lp = df_lp[columnas_modelo].copy()

# Convertir todo a numérico
for col in columnas_modelo:
    df_modelo_lp[col] = pd.to_numeric(df_modelo_lp[col], errors="coerce")

# Filtrar edades razonables
if "edad" in df_modelo_lp.columns:
    df_modelo_lp = df_modelo_lp[
        (df_modelo_lp["edad"] >= 0) & 
        (df_modelo_lp["edad"] <= 120)
    ]

# Mantener solamente defunción 0/1
df_modelo_lp = df_modelo_lp[df_modelo_lp[variable_dependiente].isin([0, 1])]

# Eliminar valores faltantes
df_modelo_lp = df_modelo_lp.dropna()

# Quitar variables sin variación, si existieran
variables_sin_variacion = []

for var in variables_predictoras:
    if df_modelo_lp[var].nunique() < 2:
        variables_sin_variacion.append(var)

if variables_sin_variacion:
    print("Variables eliminadas por no tener variación:")
    print(variables_sin_variacion)
    variables_predictoras = [v for v in variables_predictoras if v not in variables_sin_variacion]

print("Observaciones utilizadas:", len(df_modelo_lp))
print("Variable dependiente:", variable_dependiente)
print("Variables predictoras:", variables_predictoras)

display(df_modelo_lp.head())


# ------------------------------------------------------------
# 2. Definir Y y X
# ------------------------------------------------------------

Y = df_modelo_lp[variable_dependiente]

X = df_modelo_lp[variables_predictoras]
X = sm.add_constant(X)

print("Dimensiones de X:", X.shape)
print("Dimensiones de Y:", Y.shape)


# ------------------------------------------------------------
# 3. Estimar modelo Logit
# ------------------------------------------------------------

modelo_logit = sm.Logit(Y, X).fit(disp=False, maxiter=200)

print(modelo_logit.summary())


# ------------------------------------------------------------
# 4. Estimar modelo Probit
# ------------------------------------------------------------

modelo_probit = sm.Probit(Y, X).fit(disp=False, maxiter=200)

print(modelo_probit.summary())


# ------------------------------------------------------------
# 5. Tablas de coeficientes Logit y Probit
# ------------------------------------------------------------

tabla_logit = pd.DataFrame({
    "coeficiente": modelo_logit.params,
    "error_estandar": modelo_logit.bse,
    "z": modelo_logit.tvalues,
    "p_valor": modelo_logit.pvalues,
    "IC_95_inf": modelo_logit.conf_int()[0],
    "IC_95_sup": modelo_logit.conf_int()[1]
})

tabla_probit = pd.DataFrame({
    "coeficiente": modelo_probit.params,
    "error_estandar": modelo_probit.bse,
    "z": modelo_probit.tvalues,
    "p_valor": modelo_probit.pvalues,
    "IC_95_inf": modelo_probit.conf_int()[0],
    "IC_95_sup": modelo_probit.conf_int()[1]
})

print("Tabla de coeficientes Logit")
display(tabla_logit)

print("Tabla de coeficientes Probit")
display(tabla_probit)



# %% [notebook cell 20]
# ------------------------------------------------------------
# 6. Odds Ratio del modelo Logit
# ------------------------------------------------------------

# En Logit, el Odds Ratio se obtiene como exp(coeficiente).
# OR > 1: aumenta la razón de momios de defunción.
# OR < 1: reduce la razón de momios de defunción.

odds_ratio_logit = pd.DataFrame({
    "coeficiente_logit": modelo_logit.params,
    "odds_ratio": np.exp(modelo_logit.params),
    "OR_IC_95_inf": np.exp(modelo_logit.conf_int()[0]),
    "OR_IC_95_sup": np.exp(modelo_logit.conf_int()[1]),
    "p_valor": modelo_logit.pvalues
})

print("Odds Ratio - Modelo Logit")
display(odds_ratio_logit)


# ------------------------------------------------------------
# 7. Efectos marginales
# ------------------------------------------------------------

# Efectos marginales promedio:
# indican cuánto cambia la probabilidad de defunción ante un cambio
# de una unidad en cada variable explicativa.

marginales_logit = modelo_logit.get_margeff(at="overall").summary_frame()
marginales_probit = modelo_probit.get_margeff(at="overall").summary_frame()

print("Efectos marginales promedio - Logit")
display(marginales_logit)

print("Efectos marginales promedio - Probit")
display(marginales_probit)


# ------------------------------------------------------------
# 8. Probabilidades predichas
# ------------------------------------------------------------

df_predicciones = df_modelo_lp.copy()

df_predicciones["prob_logit"] = modelo_logit.predict(X)
df_predicciones["prob_probit"] = modelo_probit.predict(X)

print("Probabilidades predichas")
display(df_predicciones.head())


# ------------------------------------------------------------
# 9. Clasificación usando umbral de 0.5
# ------------------------------------------------------------

df_predicciones["pred_logit_05"] = np.where(df_predicciones["prob_logit"] >= 0.5, 1, 0)
df_predicciones["pred_probit_05"] = np.where(df_predicciones["prob_probit"] >= 0.5, 1, 0)

def metricas_clasificacion(y_real, y_pred):
    tp = int(((y_real == 1) & (y_pred == 1)).sum())
    tn = int(((y_real == 0) & (y_pred == 0)).sum())
    fp = int(((y_real == 0) & (y_pred == 1)).sum())
    fn = int(((y_real == 1) & (y_pred == 0)).sum())
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    return {
        "verdaderos_positivos": tp,
        "verdaderos_negativos": tn,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

metricas_logit_clasif = metricas_clasificacion(
    df_predicciones["defuncion"],
    df_predicciones["pred_logit_05"]
)

metricas_probit_clasif = metricas_clasificacion(
    df_predicciones["defuncion"],
    df_predicciones["pred_probit_05"]
)

tabla_clasificacion = pd.DataFrame([
    {"modelo": "Logit", **metricas_logit_clasif},
    {"modelo": "Probit", **metricas_probit_clasif}
])

print("Métricas de clasificación con umbral 0.5")
display(tabla_clasificacion)


# ------------------------------------------------------------
# 10. Comparación Logit vs Probit
# ------------------------------------------------------------

comparacion_logit_probit = pd.DataFrame({
    "modelo": ["Logit", "Probit"],
    "observaciones": [int(modelo_logit.nobs), int(modelo_probit.nobs)],
    "log_likelihood": [modelo_logit.llf, modelo_probit.llf],
    "pseudo_r2_mcfadden": [modelo_logit.prsquared, modelo_probit.prsquared],
    "aic": [modelo_logit.aic, modelo_probit.aic],
    "bic": [modelo_logit.bic, modelo_probit.bic],
    "accuracy_umbral_05": [
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Logit", "accuracy"].iloc[0],
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Probit", "accuracy"].iloc[0]
    ],
    "precision_umbral_05": [
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Logit", "precision"].iloc[0],
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Probit", "precision"].iloc[0]
    ],
    "recall_umbral_05": [
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Logit", "recall"].iloc[0],
        tabla_clasificacion.loc[tabla_clasificacion["modelo"] == "Probit", "recall"].iloc[0]
    ]
})

print("Comparación Logit vs Probit")
display(comparacion_logit_probit)



# %% [notebook cell 21]
# ------------------------------------------------------------
# 11. Probabilidades predichas por perfiles
# ------------------------------------------------------------

# Creamos perfiles hipotéticos para interpretar el modelo.
# Esto ayuda mucho en el reporte.

edad_promedio = df_modelo_lp["edad"].mean()
edad_mediana = df_modelo_lp["edad"].median()

perfiles = pd.DataFrame([
    {
        "perfil": "Paciente base",
        "edad": edad_mediana,
        "sexo_masculino": 0,
        "diabetes": 0,
        "hipertension": 0,
        "obesidad": 0,
        "neumonia": 0,
        "inmunosupresion": 0
    },
    {
        "perfil": "Paciente mayor sin comorbilidades",
        "edad": 65,
        "sexo_masculino": 0,
        "diabetes": 0,
        "hipertension": 0,
        "obesidad": 0,
        "neumonia": 0,
        "inmunosupresion": 0
    },
    {
        "perfil": "Paciente mayor con diabetes e hipertensión",
        "edad": 65,
        "sexo_masculino": 0,
        "diabetes": 1,
        "hipertension": 1,
        "obesidad": 0,
        "neumonia": 0,
        "inmunosupresion": 0
    },
    {
        "perfil": "Paciente mayor con neumonía y comorbilidades",
        "edad": 65,
        "sexo_masculino": 1,
        "diabetes": 1,
        "hipertension": 1,
        "obesidad": 1,
        "neumonia": 1,
        "inmunosupresion": 0
    }
])

# Mantener solo columnas usadas en el modelo
columnas_perfil = ["perfil"] + variables_predictoras
perfiles = perfiles[[c for c in columnas_perfil if c in perfiles.columns]]

X_perfiles = perfiles[variables_predictoras].copy()
X_perfiles = sm.add_constant(X_perfiles, has_constant="add")

# Asegurar mismo orden de columnas que X
X_perfiles = X_perfiles[X.columns]

perfiles["prob_logit"] = modelo_logit.predict(X_perfiles)
perfiles["prob_probit"] = modelo_probit.predict(X_perfiles)

print("Probabilidades predichas por perfil")
display(perfiles)


# ------------------------------------------------------------
# 12. Gráfica: Odds Ratio Logit
# ------------------------------------------------------------

or_plot = odds_ratio_logit.drop(index="const", errors="ignore").copy()

plt.figure(figsize=(9, 5))
plt.bar(or_plot.index, or_plot["odds_ratio"])
plt.axhline(1, linestyle="--", linewidth=1)
plt.title("Odds Ratio del modelo Logit")
plt.xlabel("Variables explicativas")
plt.ylabel("Odds Ratio")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_or = OUT_DIR_5 / "odds_ratio_logit.png"
plt.savefig(ruta_or, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_or}")


# ------------------------------------------------------------
# 13. Gráfica: comparación de probabilidades predichas
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df_predicciones["prob_logit"],
    df_predicciones["prob_probit"],
    alpha=0.5
)

plt.title("Comparación de probabilidades predichas: Logit vs Probit")
plt.xlabel("Probabilidad predicha Logit")
plt.ylabel("Probabilidad predicha Probit")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_probs = OUT_DIR_5 / "comparacion_probabilidades_logit_probit.png"
plt.savefig(ruta_probs, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_probs}")


# ------------------------------------------------------------
# 14. Gráfica: probabilidades predichas por perfil
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

x = np.arange(len(perfiles))
width = 0.35

plt.bar(x - width/2, perfiles["prob_logit"], width, label="Logit")
plt.bar(x + width/2, perfiles["prob_probit"], width, label="Probit")

plt.title("Probabilidades predichas de defunción por perfil")
plt.xlabel("Perfil")
plt.ylabel("Probabilidad predicha de defunción")
plt.xticks(x, perfiles["perfil"], rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

ruta_perfiles = OUT_DIR_5 / "probabilidades_predichas_perfiles.png"
plt.savefig(ruta_perfiles, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_perfiles}")



# %% [notebook cell 22]
# ------------------------------------------------------------
# 15. Guardar resultados en Excel y TXT
# ------------------------------------------------------------

ruta_excel_lp = OUT_DIR_5 / "resultados_logit_probit.xlsx"

with pd.ExcelWriter(ruta_excel_lp, engine="openpyxl") as writer:
    tabla_logit.to_excel(writer, sheet_name="coeficientes_logit")
    tabla_probit.to_excel(writer, sheet_name="coeficientes_probit")
    odds_ratio_logit.to_excel(writer, sheet_name="odds_ratio_logit")
    marginales_logit.to_excel(writer, sheet_name="marginales_logit")
    marginales_probit.to_excel(writer, sheet_name="marginales_probit")
    comparacion_logit_probit.to_excel(writer, sheet_name="comparacion_modelos", index=False)
    tabla_clasificacion.to_excel(writer, sheet_name="clasificacion_05", index=False)
    perfiles.to_excel(writer, sheet_name="probabilidades_perfiles", index=False)
    df_predicciones.to_excel(writer, sheet_name="predicciones", index=False)
    df_modelo_lp.to_excel(writer, sheet_name="datos_modelo", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_lp}")

ruta_txt_logit = OUT_DIR_5 / "summary_logit.txt"
with open(ruta_txt_logit, "w", encoding="utf-8") as f:
    f.write(str(modelo_logit.summary()))

ruta_txt_probit = OUT_DIR_5 / "summary_probit.txt"
with open(ruta_txt_probit, "w", encoding="utf-8") as f:
    f.write(str(modelo_probit.summary()))

print(f"Resumen Logit guardado en: {ruta_txt_logit}")
print(f"Resumen Probit guardado en: {ruta_txt_probit}")


# ------------------------------------------------------------
# 16. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nModelo Logit")
for var in odds_ratio_logit.index:
    if var != "const":
        coef = tabla_logit.loc[var, "coeficiente"]
        pval = tabla_logit.loc[var, "p_valor"]
        or_val = odds_ratio_logit.loc[var, "odds_ratio"]
        print(
            f"{var}: coef = {coef:.6f}, OR = {or_val:.4f}, "
            f"p-valor = {pval:.6f}"
        )

print("\nEfectos marginales Logit")
for var in marginales_logit.index:
    dy_dx = marginales_logit.loc[var, "dy/dx"]
    pval = marginales_logit.loc[var, "Pr(>|z|)"]
    print(
        f"{var}: efecto marginal = {dy_dx:.6f}, "
        f"cambio en puntos porcentuales = {dy_dx*100:.4f}, "
        f"p-valor = {pval:.6f}"
    )

print("\nComparación Logit vs Probit")
display(comparacion_logit_probit)

print("\nProbabilidades predichas por perfil")
display(perfiles)
