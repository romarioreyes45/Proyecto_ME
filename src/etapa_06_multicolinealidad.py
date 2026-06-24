"""
MULTICOLINEALIDAD
Código fuente extraído del notebook_proyecto_final_ME.ipynb.

Nota:
Este archivo corresponde a una sección del notebook.
Algunas variables pueden depender de etapas anteriores. Para ejecutar todo
el flujo completo, usa codigo_fuente_completo.py desde la raíz del proyecto.
"""


######################################################################
# ## **ETAPA 6. MULTICOLINEALIDAD**
######################################################################



# %% [notebook cell 24]
# ============================================================
# ETAPA 6. MULTICOLINEALIDAD
# Matriz de correlación y VIF
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Carpeta de salida
OUT_DIR_6 = Path("resultados_etapa_6")
OUT_DIR_6.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Funciones auxiliares
# ------------------------------------------------------------

def preparar_variables_para_vif(df_base, variables):
    """
    Selecciona variables, convierte a numérico, elimina NA
    y quita variables sin variación.
    """
    variables_disponibles = [v for v in variables if v in df_base.columns]
    df_vif = df_base[variables_disponibles].copy()
    
    for col in df_vif.columns:
        df_vif[col] = pd.to_numeric(df_vif[col], errors="coerce")
    
    df_vif = df_vif.dropna()
    
    # Quitar variables sin variación
    variables_sin_variacion = [
        col for col in df_vif.columns 
        if df_vif[col].nunique() < 2
    ]
    
    if variables_sin_variacion:
        print("Variables eliminadas por no tener variación:")
        print(variables_sin_variacion)
        df_vif = df_vif.drop(columns=variables_sin_variacion)
    
    return df_vif


def calcular_vif(df_variables):
    """
    Calcula VIF para cada variable explicativa.
    """
    X = sm.add_constant(df_variables)
    
    resultados = []
    
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        
        resultados.append({
            "variable": col,
            "VIF": vif,
            "interpretacion": interpretar_vif(vif)
        })
    
    return pd.DataFrame(resultados)


def interpretar_vif(vif):
    """
    Clasificación práctica del VIF.
    """
    if pd.isna(vif):
        return "No calculable"
    elif vif < 5:
        return "Bajo: no sugiere multicolinealidad relevante"
    elif vif < 10:
        return "Moderado: revisar posible multicolinealidad"
    else:
        return "Alto: posible multicolinealidad severa"


def obtener_correlaciones_altas(matriz_corr, umbral=0.70):
    """
    Obtiene pares de variables con correlación absoluta alta.
    """
    pares = []
    cols = matriz_corr.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = matriz_corr.iloc[i, j]
            
            if abs(corr) >= umbral:
                pares.append({
                    "variable_1": cols[i],
                    "variable_2": cols[j],
                    "correlacion": corr,
                    "correlacion_absoluta": abs(corr)
                })
    
    return pd.DataFrame(pares)


def graficar_matriz_correlacion(matriz_corr, titulo, ruta_salida):
    """
    Grafica y guarda matriz de correlación.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matriz_corr, aspect="auto")
    plt.colorbar(label="Correlación")
    
    plt.xticks(
        range(len(matriz_corr.columns)), 
        matriz_corr.columns, 
        rotation=90
    )
    plt.yticks(
        range(len(matriz_corr.index)), 
        matriz_corr.index
    )
    
    # Agregar valores dentro de la matriz
    for i in range(len(matriz_corr.index)):
        for j in range(len(matriz_corr.columns)):
            plt.text(
                j, i, 
                f"{matriz_corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )
    
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.show()
    
    print(f"Gráfica guardada en: {ruta_salida}")



# %% [notebook cell 25]
# ------------------------------------------------------------
# 2. Variables clínicas usadas en Logit y Probit
# ------------------------------------------------------------

variables_clinicas_vif = [
    "edad",
    "sexo_masculino",
    "diabetes",
    "hipertension",
    "obesidad",
    "neumonia",
    "inmunosupresion"
]

df_vif_clinico = preparar_variables_para_vif(
    df_covid,
    variables_clinicas_vif
)

print("Observaciones usadas para diagnóstico clínico:", len(df_vif_clinico))
display(df_vif_clinico.head())


# ------------------------------------------------------------
# 3. Matriz de correlación clínica
# ------------------------------------------------------------

matriz_corr_clinica = df_vif_clinico.corr()

print("Matriz de correlación - variables clínicas")
display(matriz_corr_clinica)

ruta_corr_clinica = OUT_DIR_6 / "matriz_correlacion_clinica.png"

graficar_matriz_correlacion(
    matriz_corr_clinica,
    "Matriz de correlación - Variables clínicas",
    ruta_corr_clinica
)


# ------------------------------------------------------------
# 4. Correlaciones clínicas altas
# ------------------------------------------------------------

correlaciones_altas_clinicas = obtener_correlaciones_altas(
    matriz_corr_clinica,
    umbral=0.70
)

print("Pares de variables clínicas con correlación absoluta >= 0.70")
display(correlaciones_altas_clinicas)


# ------------------------------------------------------------
# 5. VIF variables clínicas
# ------------------------------------------------------------

vif_clinico = calcular_vif(df_vif_clinico)

print("VIF - Variables clínicas")
display(vif_clinico)


# ------------------------------------------------------------
# 6. Variables temporales usadas en regresión múltiple
# ------------------------------------------------------------

variables_temporales_vif = [
    "hospitalizaciones_diarias",
    "casos_diarios"
]

df_vif_temporal = preparar_variables_para_vif(
    df_ts,
    variables_temporales_vif
)

print("Observaciones usadas para diagnóstico temporal:", len(df_vif_temporal))
display(df_vif_temporal.head())


# ------------------------------------------------------------
# 7. Matriz de correlación temporal
# ------------------------------------------------------------

matriz_corr_temporal = df_vif_temporal.corr()

print("Matriz de correlación - variables temporales")
display(matriz_corr_temporal)

ruta_corr_temporal = OUT_DIR_6 / "matriz_correlacion_temporal.png"

graficar_matriz_correlacion(
    matriz_corr_temporal,
    "Matriz de correlación - Variables temporales",
    ruta_corr_temporal
)


# ------------------------------------------------------------
# 8. Correlaciones temporales altas
# ------------------------------------------------------------

correlaciones_altas_temporales = obtener_correlaciones_altas(
    matriz_corr_temporal,
    umbral=0.70
)

print("Pares de variables temporales con correlación absoluta >= 0.70")
display(correlaciones_altas_temporales)


# ------------------------------------------------------------
# 9. VIF variables temporales
# ------------------------------------------------------------

vif_temporal = calcular_vif(df_vif_temporal)

print("VIF - Variables temporales")
display(vif_temporal)



# %% [notebook cell 26]
# ------------------------------------------------------------
# 10. Recomendaciones según resultados VIF
# ------------------------------------------------------------

def generar_recomendaciones_vif(tabla_vif, nombre_bloque):
    recomendaciones = []
    
    for _, row in tabla_vif.iterrows():
        variable = row["variable"]
        vif = row["VIF"]
        
        if pd.isna(vif):
            recomendacion = "Revisar la variable, ya que el VIF no pudo calcularse."
        elif vif < 5:
            recomendacion = "No se requiere corrección por multicolinealidad."
        elif vif < 10:
            recomendacion = (
                "Revisar correlaciones con otras variables. "
                "La multicolinealidad es moderada, pero no necesariamente crítica."
            )
        else:
            recomendacion = (
                "Existe posible multicolinealidad severa. "
                "Considerar eliminar variables redundantes, combinar variables relacionadas "
                "o usar una especificación alternativa."
            )
        
        recomendaciones.append({
            "bloque": nombre_bloque,
            "variable": variable,
            "VIF": vif,
            "diagnostico": row["interpretacion"],
            "recomendacion": recomendacion
        })
    
    return pd.DataFrame(recomendaciones)


recomendaciones_clinicas = generar_recomendaciones_vif(
    vif_clinico,
    "Variables clínicas"
)

recomendaciones_temporales = generar_recomendaciones_vif(
    vif_temporal,
    "Variables temporales"
)

recomendaciones_vif = pd.concat(
    [recomendaciones_clinicas, recomendaciones_temporales],
    ignore_index=True
)

print("Recomendaciones generales según VIF")
display(recomendaciones_vif)


# ------------------------------------------------------------
# 11. Guardar resultados en Excel
# ------------------------------------------------------------

ruta_excel_multicolinealidad = OUT_DIR_6 / "resultados_multicolinealidad.xlsx"

with pd.ExcelWriter(ruta_excel_multicolinealidad, engine="openpyxl") as writer:
    matriz_corr_clinica.to_excel(writer, sheet_name="corr_clinica")
    vif_clinico.to_excel(writer, sheet_name="vif_clinico", index=False)
    correlaciones_altas_clinicas.to_excel(writer, sheet_name="corr_altas_clinicas", index=False)
    
    matriz_corr_temporal.to_excel(writer, sheet_name="corr_temporal")
    vif_temporal.to_excel(writer, sheet_name="vif_temporal", index=False)
    correlaciones_altas_temporales.to_excel(writer, sheet_name="corr_altas_temporales", index=False)
    
    recomendaciones_vif.to_excel(writer, sheet_name="recomendaciones", index=False)

print(f"Archivo Excel guardado en: {ruta_excel_multicolinealidad}")


# ------------------------------------------------------------
# 12. Guardar resumen en TXT
# ------------------------------------------------------------

ruta_txt_multicolinealidad = OUT_DIR_6 / "resumen_multicolinealidad.txt"

with open(ruta_txt_multicolinealidad, "w", encoding="utf-8") as f:
    f.write("ETAPA 6. MULTICOLINEALIDAD\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Variables clínicas analizadas:\n")
    f.write(", ".join(df_vif_clinico.columns) + "\n\n")
    
    f.write("VIF - Variables clínicas:\n")
    f.write(vif_clinico.to_string(index=False))
    f.write("\n\n")
    
    f.write("Correlaciones clínicas altas:\n")
    if len(correlaciones_altas_clinicas) > 0:
        f.write(correlaciones_altas_clinicas.to_string(index=False))
    else:
        f.write("No se detectaron correlaciones absolutas mayores o iguales a 0.70.")
    
    f.write("\n\n" + "-" * 60 + "\n\n")
    
    f.write("Variables temporales analizadas:\n")
    f.write(", ".join(df_vif_temporal.columns) + "\n\n")
    
    f.write("VIF - Variables temporales:\n")
    f.write(vif_temporal.to_string(index=False))
    f.write("\n\n")
    
    f.write("Correlaciones temporales altas:\n")
    if len(correlaciones_altas_temporales) > 0:
        f.write(correlaciones_altas_temporales.to_string(index=False))
    else:
        f.write("No se detectaron correlaciones absolutas mayores o iguales a 0.70.")
    
    f.write("\n\n" + "-" * 60 + "\n\n")
    
    f.write("Recomendaciones:\n")
    f.write(recomendaciones_vif.to_string(index=False))

print(f"Resumen TXT guardado en: {ruta_txt_multicolinealidad}")


# ------------------------------------------------------------
# 13. Síntesis de resultados para el reporte
# ------------------------------------------------------------

print("Resumen para el reporte")
print("---------------------------------------------")

print("\nVariables clínicas")
display(vif_clinico)

max_vif_clinico = vif_clinico["VIF"].max()
print(f"Mayor VIF clínico: {max_vif_clinico:.4f}")

if max_vif_clinico < 5:
    print("Lectura técnica: No se observa multicolinealidad relevante en las variables clínicas.")
elif max_vif_clinico < 10:
    print("Lectura técnica: Se observa multicolinealidad moderada en algunas variables clínicas.")
else:
    print("Lectura técnica: Se observa posible multicolinealidad severa en variables clínicas.")

print("\nVariables temporales")
display(vif_temporal)

max_vif_temporal = vif_temporal["VIF"].max()
print(f"Mayor VIF temporal: {max_vif_temporal:.4f}")

if max_vif_temporal < 5:
    print("Lectura técnica: No se observa multicolinealidad relevante en las variables temporales.")
elif max_vif_temporal < 10:
    print("Lectura técnica: Se observa multicolinealidad moderada en algunas variables temporales.")
else:
    print("Lectura técnica: Se observa posible multicolinealidad severa en variables temporales.")

print("\nCorrelaciones clínicas altas:")
display(correlaciones_altas_clinicas)

print("\nCorrelaciones temporales altas:")
display(correlaciones_altas_temporales)
