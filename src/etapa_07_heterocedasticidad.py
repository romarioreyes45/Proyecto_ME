"""
HETEROCEDASTICIDAD
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 7. HETEROCEDASTICIDAD**
######################################################################



# %% [notebook cell 28]
# ============================================================
# ETAPA 7. HETEROCEDASTICIDAD
# Pruebas de Breusch-Pagan, White y errores robustos
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Carpeta de salida para esta etapa
OUT_DIR_7 = Path("resultados_etapa_7")
OUT_DIR_7.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Modelo de regresión lineal simple
# Defunciones diarias ~ Hospitalizaciones diarias
# ------------------------------------------------------------

df_rls_het = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias"
]].copy()

for col in df_rls_het.columns:
    df_rls_het[col] = pd.to_numeric(df_rls_het[col], errors="coerce")

df_rls_het = df_rls_het.dropna()

Y_rls = df_rls_het["defunciones_diarias"]
X_rls = df_rls_het[["hospitalizaciones_diarias"]]
X_rls = sm.add_constant(X_rls)

modelo_rls_het = sm.OLS(Y_rls, X_rls).fit()

print("Resumen modelo RLS")
print(modelo_rls_het.summary())


# ------------------------------------------------------------
# 2. Modelo de regresión lineal múltiple
# Defunciones diarias ~ Hospitalizaciones diarias + Casos diarios
# ------------------------------------------------------------

df_rlm_het = df_ts[[
    "defunciones_diarias",
    "hospitalizaciones_diarias",
    "casos_diarios"
]].copy()

for col in df_rlm_het.columns:
    df_rlm_het[col] = pd.to_numeric(df_rlm_het[col], errors="coerce")

df_rlm_het = df_rlm_het.dropna()

Y_rlm = df_rlm_het["defunciones_diarias"]
X_rlm = df_rlm_het[[
    "hospitalizaciones_diarias",
    "casos_diarios"
]]
X_rlm = sm.add_constant(X_rlm)

modelo_rlm_het = sm.OLS(Y_rlm, X_rlm).fit()

print("Resumen modelo RLM")
print(modelo_rlm_het.summary())


# ------------------------------------------------------------
# 3. Funciones auxiliares
# ------------------------------------------------------------

def pruebas_heterocedasticidad(modelo, nombre_modelo):
    """
    Aplica Breusch-Pagan y White a un modelo OLS.
    """
    residuos = modelo.resid
    exog = modelo.model.exog
    
    # Breusch-Pagan
    bp_test = het_breuschpagan(residuos, exog)
    
    # White
    white_test = het_white(residuos, exog)
    
    resultados = pd.DataFrame({
        "modelo": [nombre_modelo, nombre_modelo],
        "prueba": ["Breusch-Pagan", "White"],
        "estadistico_lm": [bp_test[0], white_test[0]],
        "p_valor_lm": [bp_test[1], white_test[1]],
        "estadistico_f": [bp_test[2], white_test[2]],
        "p_valor_f": [bp_test[3], white_test[3]]
    })
    
    resultados["decision_5"] = np.where(
        resultados["p_valor_lm"] < 0.05,
        "Rechazar H0: existe evidencia de heterocedasticidad",
        "No rechazar H0: no hay evidencia suficiente de heterocedasticidad"
    )
    
    return resultados


def tabla_coeficientes_modelo(modelo, nombre_modelo, tipo_error):
    """
    Genera tabla ordenada de coeficientes para un modelo OLS.
    """
    nombres = modelo.model.exog_names
    
    tabla = pd.DataFrame({
        "modelo": nombre_modelo,
        "tipo_error": tipo_error,
        "variable": nombres,
        "coeficiente": modelo.params,
        "error_estandar": modelo.bse,
        "t": modelo.tvalues,
        "p_valor": modelo.pvalues,
        "IC_95_inf": modelo.conf_int()[:, 0] if isinstance(modelo.conf_int(), np.ndarray) else modelo.conf_int()[0],
        "IC_95_sup": modelo.conf_int()[:, 1] if isinstance(modelo.conf_int(), np.ndarray) else modelo.conf_int()[1]
    })
    
    return tabla


def comparar_errores(modelo_original, modelo_robusto, nombre_modelo):
    """
    Compara errores estándar y p-valores originales contra errores robustos.
    """
    nombres = modelo_original.model.exog_names
    
    comparacion = pd.DataFrame({
        "modelo": nombre_modelo,
        "variable": nombres,
        "coeficiente_ols": modelo_original.params,
        "error_estandar_ols": modelo_original.bse,
        "p_valor_ols": modelo_original.pvalues,
        "error_estandar_robusto_HC3": modelo_robusto.bse,
        "p_valor_robusto_HC3": modelo_robusto.pvalues
    })
    
    comparacion["cambio_error_estandar"] = (
        comparacion["error_estandar_robusto_HC3"] -
        comparacion["error_estandar_ols"]
    )
    
    comparacion["cambio_p_valor"] = (
        comparacion["p_valor_robusto_HC3"] -
        comparacion["p_valor_ols"]
    )
    
    return comparacion



# %% [notebook cell 29]
# ------------------------------------------------------------
# 4. Pruebas para ambos modelos
# ------------------------------------------------------------

hetero_rls = pruebas_heterocedasticidad(
    modelo_rls_het,
    "Regresión lineal simple"
)

hetero_rlm = pruebas_heterocedasticidad(
    modelo_rlm_het,
    "Regresión lineal múltiple"
)

resultados_hetero = pd.concat(
    [hetero_rls, hetero_rlm],
    ignore_index=True
)

print("Resultados de pruebas de heterocedasticidad")
display(resultados_hetero)


# ------------------------------------------------------------
# 5. Modelos con errores robustos HC3
# ------------------------------------------------------------

modelo_rls_robusto = modelo_rls_het.get_robustcov_results(cov_type="HC3")
modelo_rlm_robusto = modelo_rlm_het.get_robustcov_results(cov_type="HC3")

print("Modelo RLS con errores robustos HC3")
print(modelo_rls_robusto.summary())

print("\nModelo RLM con errores robustos HC3")
print(modelo_rlm_robusto.summary())


# ------------------------------------------------------------
# 6. Tablas de coeficientes originales y robustos
# ------------------------------------------------------------

tabla_rls_ols = tabla_coeficientes_modelo(
    modelo_rls_het,
    "Regresión lineal simple",
    "OLS"
)

tabla_rls_robusto = tabla_coeficientes_modelo(
    modelo_rls_robusto,
    "Regresión lineal simple",
    "Robusto HC3"
)

tabla_rlm_ols = tabla_coeficientes_modelo(
    modelo_rlm_het,
    "Regresión lineal múltiple",
    "OLS"
)

tabla_rlm_robusto = tabla_coeficientes_modelo(
    modelo_rlm_robusto,
    "Regresión lineal múltiple",
    "Robusto HC3"
)

tabla_coeficientes_hetero = pd.concat(
    [
        tabla_rls_ols,
        tabla_rls_robusto,
        tabla_rlm_ols,
        tabla_rlm_robusto
    ],
    ignore_index=True
)

print("Coeficientes OLS vs errores robustos")
display(tabla_coeficientes_hetero)


# ------------------------------------------------------------
# 7. Comparación directa de errores estándar y p-valores
# ------------------------------------------------------------

comparacion_rls = comparar_errores(
    modelo_rls_het,
    modelo_rls_robusto,
    "Regresión lineal simple"
)

comparacion_rlm = comparar_errores(
    modelo_rlm_het,
    modelo_rlm_robusto,
    "Regresión lineal múltiple"
)

comparacion_errores = pd.concat(
    [comparacion_rls, comparacion_rlm],
    ignore_index=True
)

print("Comparación de errores estándar y p-valores")
display(comparacion_errores)


# ------------------------------------------------------------
# 8. Gráfica residuos vs predichos - RLS
# ------------------------------------------------------------

df_graf_rls = pd.DataFrame({
    "predichos": modelo_rls_het.fittedvalues,
    "residuos": modelo_rls_het.resid
})

plt.figure(figsize=(8, 5))
plt.scatter(df_graf_rls["predichos"], df_graf_rls["residuos"], alpha=0.7)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Residuos vs valores predichos - Regresión lineal simple")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_residuos_rls = OUT_DIR_7 / "heterocedasticidad_residuos_rls.png"
plt.savefig(ruta_residuos_rls, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_residuos_rls}")


# ------------------------------------------------------------
# 9. Gráfica residuos vs predichos - RLM
# ------------------------------------------------------------

df_graf_rlm = pd.DataFrame({
    "predichos": modelo_rlm_het.fittedvalues,
    "residuos": modelo_rlm_het.resid
})

plt.figure(figsize=(8, 5))
plt.scatter(df_graf_rlm["predichos"], df_graf_rlm["residuos"], alpha=0.7)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Residuos vs valores predichos - Regresión lineal múltiple")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.grid(alpha=0.3)
plt.tight_layout()

ruta_residuos_rlm = OUT_DIR_7 / "heterocedasticidad_residuos_rlm.png"
plt.savefig(ruta_residuos_rlm, dpi=300)
plt.show()

print(f"Gráfica guardada en: {ruta_residuos_rlm}")



# %% [notebook cell 30]
# ------------------------------------------------------------
# 10. Conclusión con base en pruebas
# ------------------------------------------------------------

def conclusion_hetero(tabla_resultados, nombre_modelo):
    sub = tabla_resultados[tabla_resultados["modelo"] == nombre_modelo]
    
    existe_hetero = (sub["p_valor_lm"] < 0.05).any()
    
    if existe_hetero:
        return (
            f"En el modelo {nombre_modelo} se detecta evidencia de heterocedasticidad "
            f"al menos en una de las pruebas aplicadas. Por ello, se recomienda interpretar "
            f"los resultados utilizando errores estándar robustos HC3."
        )
    else:
        return (
            f"En el modelo {nombre_modelo} no se detecta evidencia estadística suficiente "
            f"de heterocedasticidad al 5% de significancia. Los errores estándar OLS "
            f"pueden considerarse adecuados bajo este diagnóstico."
        )


conclusion_rls = conclusion_hetero(
    resultados_hetero,
    "Regresión lineal simple"
)

conclusion_rlm = conclusion_hetero(
    resultados_hetero,
    "Regresión lineal múltiple"
)

print(conclusion_rls)
print(conclusion_rlm)


# ------------------------------------------------------------
# 11. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_hetero = OUT_DIR_7 / "resultados_heterocedasticidad.xlsx"

with pd.ExcelWriter(ruta_excel_hetero, engine="openpyxl") as writer:
    resultados_hetero.to_excel(writer, sheet_name="pruebas_hetero", index=False)
    tabla_coeficientes_hetero.to_excel(writer, sheet_name="coeficientes_ols_robustos", index=False)
    comparacion_errores.to_excel(writer, sheet_name="comparacion_errores", index=False)
    df_graf_rls.to_excel(writer, sheet_name="residuos_rls", index=False)
    df_graf_rlm.to_excel(writer, sheet_name="residuos_rlm", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_hetero}")


# ------------------------------------------------------------
# 12. Guardar resúmenes en TXT
# ------------------------------------------------------------

ruta_txt_hetero = OUT_DIR_7 / "resumen_heterocedasticidad.txt"

with open(ruta_txt_hetero, "w", encoding="utf-8") as f:
    f.write("ETAPA 7. HETEROCEDASTICIDAD\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Pruebas de heterocedasticidad:\n")
    f.write(resultados_hetero.to_string(index=False))
    f.write("\n\n")
    
    f.write("Conclusiones:\n")
    f.write(conclusion_rls + "\n")
    f.write(conclusion_rlm + "\n\n")
    
    f.write("=" * 70 + "\n\n")
    f.write("Resumen modelo RLS OLS:\n")
    f.write(str(modelo_rls_het.summary()))
    f.write("\n\n")
    
    f.write("Resumen modelo RLS con errores robustos HC3:\n")
    f.write(str(modelo_rls_robusto.summary()))
    f.write("\n\n")
    
    f.write("=" * 70 + "\n\n")
    f.write("Resumen modelo RLM OLS:\n")
    f.write(str(modelo_rlm_het.summary()))
    f.write("\n\n")
    
    f.write("Resumen modelo RLM con errores robustos HC3:\n")
    f.write(str(modelo_rlm_robusto.summary()))

print(f"Resumen TXT guardado en: {ruta_txt_hetero}")


# ------------------------------------------------------------
# 13. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

display(resultados_hetero)

print("\nConclusión RLS:")
print(conclusion_rls)

print("\nConclusión RLM:")
print(conclusion_rlm)

print("\nComparación de errores estándar y p-valores:")
display(comparacion_errores)
