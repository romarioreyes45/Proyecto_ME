"""
VARIABLES DUMMY
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 4. VARIABLES DUMMY**
######################################################################



# %% [notebook cell 15]
# ============================================================
# ETAPA 4. VARIABLES DUMMY
# Proyecto: Modelos Econométricos - COVID-19 en México
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

# Carpeta de salida para esta etapa
OUT_DIR_4 = Path("resultados_etapa_4")
OUT_DIR_4.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Preparar base de trabajo
# ------------------------------------------------------------

df_dummy = df_covid.copy()

# Asegurar que edad sea numérica
if "edad" in df_dummy.columns:
    df_dummy["edad"] = pd.to_numeric(df_dummy["edad"], errors="coerce")

# Si alguna dummy no existe, se intenta reconstruir a partir de la variable original

# Sexo: normalmente 1 = Mujer, 2 = Hombre
if "sexo_masculino" not in df_dummy.columns and "sexo" in df_dummy.columns:
    df_dummy["sexo_masculino"] = pd.to_numeric(df_dummy["sexo"], errors="coerce").map({2: 1, 1: 0})

# Hospitalización: normalmente 1 = Ambulatorio, 2 = Hospitalizado
if "hospitalizado" not in df_dummy.columns and "tipo_paciente" in df_dummy.columns:
    df_dummy["hospitalizado"] = pd.to_numeric(df_dummy["tipo_paciente"], errors="coerce").map({2: 1, 1: 0})

# Defunción: si existe fecha de defunción válida, defuncion = 1
if "defuncion" not in df_dummy.columns and "fecha_def" in df_dummy.columns:
    df_dummy["defuncion"] = np.where(df_dummy["fecha_def"].notna(), 1, 0)

# Función para convertir variables Sí/No:
# 1 = Sí, 2 = No, otros valores se toman como NaN
def convertir_si_no(serie):
    serie_num = pd.to_numeric(serie, errors="coerce")
    return serie_num.map({1: 1, 2: 0})

# Variables clínicas
variables_clinicas = ["diabetes", "hipertension", "obesidad", "neumonia", "inmunosupresion"]

# Si inmunosupresión viene como inmusupr, renombrar
if "inmusupr" in df_dummy.columns and "inmunosupresion" not in df_dummy.columns:
    df_dummy = df_dummy.rename(columns={"inmusupr": "inmunosupresion"})

for col in variables_clinicas:
    if col in df_dummy.columns:
        # Si tiene valores distintos de 0/1, se convierte con catálogo 1/2
        valores = set(pd.to_numeric(df_dummy[col], errors="coerce").dropna().unique())
        if not valores.issubset({0, 1}):
            df_dummy[col] = convertir_si_no(df_dummy[col])

# Número de comorbilidades
comorbilidades_base = ["diabetes", "hipertension", "obesidad", "neumonia", "inmunosupresion"]
comorbilidades_existentes = [c for c in comorbilidades_base if c in df_dummy.columns]

df_dummy["num_comorbilidades"] = df_dummy[comorbilidades_existentes].sum(axis=1, skipna=True)

print("Dimensiones de la base para variables dummy:", df_dummy.shape)
display(df_dummy.head())


# ------------------------------------------------------------
# 2. Tabla de variables dummy
# ------------------------------------------------------------

dummies_obligatorias = [
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "hospitalizado",
    "defuncion"
]

# Variables extra útiles para el proyecto y el punto 11
dummies_extra = [
    "neumonia",
    "inmunosupresion"
]

dummies_disponibles = [v for v in dummies_obligatorias + dummies_extra if v in df_dummy.columns]

tabla_dummies = []

for var in dummies_disponibles:
    serie = pd.to_numeric(df_dummy[var], errors="coerce")
    
    tabla_dummies.append({
        "variable_dummy": var,
        "categoria_0": int((serie == 0).sum()),
        "categoria_1": int((serie == 1).sum()),
        "valores_perdidos": int(serie.isna().sum()),
        "porcentaje_1": (serie.eq(1).mean() * 100)
    })

tabla_dummies_etapa4 = pd.DataFrame(tabla_dummies)

print("Tabla de distribución de variables dummy")
display(tabla_dummies_etapa4)


# ------------------------------------------------------------
# 3. Modelo A: Hospitalización explicada por dummies clínicas
# ------------------------------------------------------------

variables_modelo_hosp = [
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion"
]

variables_modelo_hosp = [v for v in variables_modelo_hosp if v in df_dummy.columns]

df_modelo_hosp = df_dummy[["hospitalizado"] + variables_modelo_hosp].dropna().copy()

Y_hosp = df_modelo_hosp["hospitalizado"]
X_hosp = df_modelo_hosp[variables_modelo_hosp]
X_hosp = sm.add_constant(X_hosp)

modelo_dummy_hosp = sm.OLS(Y_hosp, X_hosp).fit()

print(modelo_dummy_hosp.summary())


# ------------------------------------------------------------
# 4. Tabla de resultados del modelo A
# ------------------------------------------------------------

tabla_dummy_hosp = pd.DataFrame({
    "coeficiente": modelo_dummy_hosp.params,
    "error_estandar": modelo_dummy_hosp.bse,
    "t": modelo_dummy_hosp.tvalues,
    "p_valor": modelo_dummy_hosp.pvalues,
    "IC_95_inf": modelo_dummy_hosp.conf_int()[0],
    "IC_95_sup": modelo_dummy_hosp.conf_int()[1]
})

print("Modelo A: Hospitalización explicada por variables dummy")
display(tabla_dummy_hosp)



# %% [notebook cell 16]
# ------------------------------------------------------------
# 5. Modelo B: Defunción explicada por dummies clínicas y hospitalización
# ------------------------------------------------------------

variables_modelo_def = [
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion",
    "hospitalizado"
]

variables_modelo_def = [v for v in variables_modelo_def if v in df_dummy.columns]

df_modelo_def = df_dummy[["defuncion"] + variables_modelo_def].dropna().copy()

Y_def = df_modelo_def["defuncion"]
X_def = df_modelo_def[variables_modelo_def]
X_def = sm.add_constant(X_def)

modelo_dummy_def = sm.OLS(Y_def, X_def).fit()

print(modelo_dummy_def.summary())


# ------------------------------------------------------------
# 6. Tabla de resultados del modelo B
# ------------------------------------------------------------

tabla_dummy_def = pd.DataFrame({
    "coeficiente": modelo_dummy_def.params,
    "error_estandar": modelo_dummy_def.bse,
    "t": modelo_dummy_def.tvalues,
    "p_valor": modelo_dummy_def.pvalues,
    "IC_95_inf": modelo_dummy_def.conf_int()[0],
    "IC_95_sup": modelo_dummy_def.conf_int()[1]
})

print("Modelo B: Defunción explicada por variables dummy")
display(tabla_dummy_def)


# ------------------------------------------------------------
# 7. Métricas generales de ambos modelos
# ------------------------------------------------------------

metricas_dummy = pd.DataFrame({
    "modelo": [
        "Modelo A: Hospitalización",
        "Modelo B: Defunción"
    ],
    "variable_dependiente": [
        "hospitalizado",
        "defuncion"
    ],
    "observaciones": [
        int(modelo_dummy_hosp.nobs),
        int(modelo_dummy_def.nobs)
    ],
    "r_cuadrada": [
        modelo_dummy_hosp.rsquared,
        modelo_dummy_def.rsquared
    ],
    "r_cuadrada_ajustada": [
        modelo_dummy_hosp.rsquared_adj,
        modelo_dummy_def.rsquared_adj
    ],
    "estadistico_f": [
        modelo_dummy_hosp.fvalue,
        modelo_dummy_def.fvalue
    ],
    "p_valor_f": [
        modelo_dummy_hosp.f_pvalue,
        modelo_dummy_def.f_pvalue
    ],
    "aic": [
        modelo_dummy_hosp.aic,
        modelo_dummy_def.aic
    ],
    "bic": [
        modelo_dummy_hosp.bic,
        modelo_dummy_def.bic
    ]
})

display(metricas_dummy)


# ------------------------------------------------------------
# 8. Síntesis técnica de coeficientes
# ------------------------------------------------------------

def construir_sintesis_coeficientes_dummy(tabla, nombre_modelo):
    sintesis = []
    
    for var in tabla.index:
        if var == "const":
            continue
        
        coef = tabla.loc[var, "coeficiente"]
        pvalor = tabla.loc[var, "p_valor"]
        ic_inf = tabla.loc[var, "IC_95_inf"]
        ic_sup = tabla.loc[var, "IC_95_sup"]
        
        sintesis.append({
            "modelo": nombre_modelo,
            "variable": var,
            "coeficiente": coef,
            "cambio_en_puntos_porcentuales": coef * 100,
            "p_valor": pvalor,
            "IC_95_inf": ic_inf,
            "IC_95_sup": ic_sup,
            "significativa_5": "Sí" if pvalor < 0.05 else "No",
            "lectura_tecnica": (
                f"Pasar de 0 a 1 en {var} se asocia con un cambio promedio de "
                f"{coef*100:.2f} puntos porcentuales en la variable dependiente."
            )
        })
    
    return pd.DataFrame(sintesis)

sintesis_hosp = construir_sintesis_coeficientes_dummy(
    tabla_dummy_hosp,
    "Modelo A: Hospitalización"
)

sintesis_def = construir_sintesis_coeficientes_dummy(
    tabla_dummy_def,
    "Modelo B: Defunción"
)

sintesis_dummy = pd.concat([sintesis_hosp, sintesis_def], ignore_index=True)

display(sintesis_dummy)



# %% [notebook cell 17]
# ------------------------------------------------------------
# 9. Gráfica de coeficientes del Modelo A
# ------------------------------------------------------------

coef_hosp = tabla_dummy_hosp.drop(index="const", errors="ignore").copy()

plt.figure(figsize=(9, 5))
plt.bar(coef_hosp.index, coef_hosp["coeficiente"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Coeficientes de variables dummy - Modelo A: Hospitalización")
plt.xlabel("Variables dummy")
plt.ylabel("Coeficiente estimado")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_coef_hosp = OUT_DIR_4 / "coeficientes_dummy_hospitalizacion.png"
plt.savefig(ruta_coef_hosp, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_coef_hosp}")


# ------------------------------------------------------------
# 10. Gráfica de coeficientes del Modelo B
# ------------------------------------------------------------

coef_def = tabla_dummy_def.drop(index="const", errors="ignore").copy()

plt.figure(figsize=(9, 5))
plt.bar(coef_def.index, coef_def["coeficiente"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Coeficientes de variables dummy - Modelo B: Defunción")
plt.xlabel("Variables dummy")
plt.ylabel("Coeficiente estimado")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ruta_coef_def = OUT_DIR_4 / "coeficientes_dummy_defuncion.png"
plt.savefig(ruta_coef_def, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_coef_def}")


# ------------------------------------------------------------
# 11. Guardar resultados en Excel y TXT
# ------------------------------------------------------------

ruta_excel_dummy = OUT_DIR_4 / "resultados_variables_dummy.xlsx"

with pd.ExcelWriter(ruta_excel_dummy, engine="openpyxl") as writer:
    tabla_dummies_etapa4.to_excel(writer, sheet_name="distribucion_dummies", index=False)
    tabla_dummy_hosp.to_excel(writer, sheet_name="modelo_hospitalizacion")
    tabla_dummy_def.to_excel(writer, sheet_name="modelo_defuncion")
    metricas_dummy.to_excel(writer, sheet_name="metricas_modelos", index=False)
    sintesis_dummy.to_excel(writer, sheet_name="sintesis_coeficientes", index=False)
    df_modelo_hosp.to_excel(writer, sheet_name="datos_modelo_hosp", index=False)
    df_modelo_def.to_excel(writer, sheet_name="datos_modelo_def", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_dummy}")

ruta_txt_hosp = OUT_DIR_4 / "summary_modelo_dummy_hospitalizacion.txt"
with open(ruta_txt_hosp, "w", encoding="utf-8") as f:
    f.write(str(modelo_dummy_hosp.summary()))

ruta_txt_def = OUT_DIR_4 / "summary_modelo_dummy_defuncion.txt"
with open(ruta_txt_def, "w", encoding="utf-8") as f:
    f.write(str(modelo_dummy_def.summary()))

print(f"Resumen Modelo A guardado en: {ruta_txt_hosp}")
print(f"Resumen Modelo B guardado en: {ruta_txt_def}")


# ------------------------------------------------------------
# 12. Resumen rápido para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nModelo A: Hospitalización")
for var in tabla_dummy_hosp.index:
    if var != "const":
        coef = tabla_dummy_hosp.loc[var, "coeficiente"]
        pval = tabla_dummy_hosp.loc[var, "p_valor"]
        print(f"{var}: coef = {coef:.6f}, cambio = {coef*100:.2f} puntos porcentuales, p-valor = {pval:.6f}")

print("\nModelo B: Defunción")
for var in tabla_dummy_def.index:
    if var != "const":
        coef = tabla_dummy_def.loc[var, "coeficiente"]
        pval = tabla_dummy_def.loc[var, "p_valor"]
        print(f"{var}: coef = {coef:.6f}, cambio = {coef*100:.2f} puntos porcentuales, p-valor = {pval:.6f}")
