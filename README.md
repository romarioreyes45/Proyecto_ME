# Análisis Econométrico Integral de la Mortalidad, Hospitalización y Evolución Temporal por COVID-19 en México

## Descripción del proyecto

Este repositorio contiene el desarrollo de un proyecto econométrico aplicado al análisis de COVID-19 en México. El objetivo principal es estudiar la relación entre variables demográficas, clínicas y temporales con la hospitalización, la defunción y la evolución diaria de la pandemia.

El proyecto utiliza datos abiertos oficiales de la Secretaría de Salud de México y aplica diferentes técnicas econométricas vistas durante el curso, incluyendo modelos lineales, variables dummy, modelos probabilísticos, modelos dinámicos y series de tiempo.

El análisis se enfoca en identificar asociaciones estadísticas relevantes, evaluar supuestos econométricos y generar pronósticos de corto plazo para las defunciones diarias por COVID-19.

---

## Objetivo general

Aplicar técnicas econométricas estáticas, dinámicas y predictivas para analizar los factores asociados a la mortalidad, hospitalización y evolución temporal del COVID-19 en México utilizando datos abiertos oficiales.

---

## Fuente de datos

Los datos utilizados provienen del portal oficial de Datos Abiertos COVID-19 de la Secretaría de Salud de México:

https://www.gob.mx/salud/documentos/datos-abiertos-152127

La base original contiene registros relacionados con enfermedades respiratorias. Para este proyecto se filtraron únicamente los casos confirmados de COVID-19, utilizando como criterio principal la variable `CLASIFICACION_FINAL_COVID`.

---

## Muestra utilizada

* Registros totales cargados: 223,952
* Casos confirmados por clasificación final: 7,666
* Casos confirmados por PCR: 7,666
* Registros considerados como casos confirmados de COVID-19: 7,666

---

## Pregunta de investigación

¿En qué medida las características demográficas, clínicas y temporales de los pacientes con COVID-19 en México explican la probabilidad de hospitalización y defunción, y cómo la evolución diaria de los casos, hospitalizaciones y muertes permite generar pronósticos mediante modelos econométricos?

---

## Variables principales

Las variables utilizadas se organizaron en los siguientes grupos:

| Grupo de variables | Variables incluidas                                           | Uso en el análisis                                               |
| ------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Demográficas       | Edad, sexo masculino                                          | Explicar diferencias individuales en hospitalización y defunción |
| Clínicas           | Diabetes, hipertensión, obesidad, neumonía, inmunosupresión   | Identificar factores de riesgo asociados con gravedad            |
| Resultado          | Hospitalización, defunción                                    | Medir desenlaces clínicos del paciente                           |
| Temporales         | Casos diarios, hospitalizaciones diarias, defunciones diarias | Modelar evolución, rezagos y pronósticos                         |
| Construidas        | Número de comorbilidades, rezagos, expectativas adaptativas   | Capturar carga clínica y dinámica temporal                       |

---

## Metodología

El proyecto se desarrolló en varias etapas:

1. Carga y preparación de datos.
2. Limpieza de valores faltantes.
3. Corrección de tipos de datos.
4. Construcción de variables binarias.
5. Estadística descriptiva.
6. Regresión lineal simple.
7. Regresión lineal múltiple.
8. Modelos con variables dummy.
9. Modelos Logit y Probit.
10. Diagnósticos econométricos.
11. Modelos con rezagos distribuidos.
12. Modelo de ajuste parcial.
13. Modelo de expectativas adaptativas.
14. Modelos de series de tiempo.
15. Pronóstico de defunciones diarias.

---

## Modelos aplicados

### Regresión lineal simple

Se estimó un modelo para analizar la relación entre hospitalizaciones diarias y defunciones diarias:

```text
DefuncionesDiarias_t = β_0 + β_1 HospitalizacionesDiarias_t + u_t
```

Este modelo permitió observar una primera asociación positiva entre presión hospitalaria y mortalidad diaria.

---

### Regresión lineal múltiple

Se incorporaron casos diarios y hospitalizaciones diarias como variables explicativas:

```text
DefuncionesDiarias_t = β_0 + β_1 HospitalizacionesDiarias_t + β_2 CasosDiarios_t + u_t
```

El modelo permitió analizar el efecto conjunto de contagios y hospitalizaciones sobre las defunciones diarias.

---

### Variables dummy

Se construyeron variables binarias para representar características clínicas y demográficas como:

* Sexo masculino
* Diabetes
* Hipertensión
* Obesidad
* Neumonía
* Inmunosupresión
* Hospitalización
* Defunción

Estas variables permitieron analizar diferencias promedio entre grupos de pacientes.

---

### Modelos Logit y Probit

Se estimaron modelos probabilísticos para analizar la probabilidad de defunción. La variable dependiente fue `defuncion`, que toma el valor de 1 si el paciente falleció y 0 en caso contrario.

Entre los principales resultados del modelo Logit se encontraron los siguientes Odds Ratio:

| Variable       | Odds Ratio |
| -------------- | ---------: |
| Edad           |      1.033 |
| Sexo masculino |      1.637 |
| Diabetes       |      1.515 |
| Neumonía       |      5.389 |

La neumonía fue el factor clínico con mayor asociación con la defunción.

---

### Modelos con rezagos distribuidos

Se estimaron modelos para analizar si los casos diarios y hospitalizaciones tenían efectos contemporáneos y rezagados sobre las defunciones.

Modelo con casos diarios:

```text
DefuncionesDiarias_t = β_0 + β_1 CasosDiarios_t + β_2 CasosDiarios_t-1 + β_3 CasosDiarios_t-7 + β_4 CasosDiarios_t-14 + u_t
```

Modelo con hospitalizaciones diarias:

```text
DefuncionesDiarias_t = β_0 + β_1 HospitalizacionesDiarias_t + β_2 HospitalizacionesDiarias_t-1 + β_3 HospitalizacionesDiarias_t-7 + β_4 HospitalizacionesDiarias_t-14 + u_t
```

El rezago de casos a 14 días resultó relevante dentro del análisis temporal.

---

### Modelo de ajuste parcial

Se estimó un modelo de ajuste parcial para analizar la persistencia temporal de las defunciones:

```text
DefuncionesDiarias_t = α + β HospitalizacionesDiarias_t + ρ DefuncionesDiarias_t-1 + u_t
```

A partir del coeficiente de persistencia se calculó la velocidad de ajuste:

```text
λ = 1 - ρ
```

La velocidad de ajuste estimada fue alta, lo que sugiere un ajuste rápido hacia el equilibrio en la especificación utilizada.

---

### Expectativas adaptativas

Se construyeron expectativas adaptativas para casos diarios, hospitalizaciones diarias y defunciones diarias mediante la siguiente expresión:

```text
E_t = λY_t-1 + (1 - λ)E_t-1
```

Los mejores valores de lambda fueron:

| Variable                  | Lambda |
| ------------------------- | -----: |
| Casos diarios             |   0.20 |
| Hospitalizaciones diarias |   0.20 |
| Defunciones diarias       |   0.10 |

Estos resultados indican que las expectativas se ajustan lentamente y dependen en gran medida de la trayectoria pasada.

---

### Series de tiempo

Se construyeron series diarias de:

* Casos diarios
* Hospitalizaciones diarias
* Defunciones diarias

Se aplicó la prueba Dickey-Fuller Aumentada para evaluar estacionariedad. Las series originales no fueron estacionarias, pero al aplicar primeras diferencias se volvieron estacionarias.

Se compararon modelos AR, MA, ARMA y ARIMA. El mejor modelo para defunciones diarias fue:

```text
ARIMA(1,1,1)
```

Este modelo fue seleccionado con base en los criterios AIC y BIC.

---

## Diagnósticos econométricos

Se aplicaron pruebas para evaluar los supuestos de los modelos:

| Diagnóstico                | Resultado principal                                          | Interpretación                             |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| Multicolinealidad clínica  | VIF máximo de 1.528                                          | No se observa multicolinealidad relevante  |
| Multicolinealidad temporal | VIF de 9.395 entre casos y hospitalizaciones                 | Existe multicolinealidad moderada          |
| Heterocedasticidad         | Breusch-Pagan y White con p-valores menores a 0.05           | Se detecta heterocedasticidad              |
| Autocorrelación            | Durbin-Watson cercano a 2 y Breusch-Godfrey no significativo | No se observa autocorrelación importante   |
| Normalidad                 | Jarque-Bera y Shapiro-Wilk rechazan normalidad               | Los residuos no siguen distribución normal |

Debido a la presencia de heterocedasticidad, se recomienda interpretar los modelos lineales considerando errores estándar robustos.

---

## Resultados principales

Los principales hallazgos del proyecto fueron:

* La edad se asocia positivamente con la probabilidad de defunción.
* El sexo masculino presentó mayor riesgo relativo de defunción.
* La diabetes se asoció con mayor probabilidad de defunción.
* La neumonía fue el factor clínico con mayor asociación con mortalidad.
* Las hospitalizaciones diarias se asociaron positivamente con las defunciones diarias.
* Los efectos rezagados fueron relevantes, especialmente el rezago de casos a 14 días.
* El modelo ARIMA(1,1,1) fue el mejor modelo de series de tiempo para pronóstico de defunciones diarias según AIC y BIC.

---

## Interpretación económica

Desde una perspectiva econométrica y de salud pública, los resultados permiten identificar variables relevantes para el análisis de mortalidad y presión hospitalaria.

La edad, diabetes y neumonía ayudan a identificar grupos vulnerables. Las hospitalizaciones funcionan como un indicador agregado de gravedad y presión sobre el sistema de salud. Además, los efectos rezagados muestran que las defunciones no dependen únicamente del comportamiento actual de los casos, sino también de la evolución previa de la pandemia.

El pronóstico mediante ARIMA puede apoyar la planeación de corto plazo, aunque debe interpretarse con cautela debido a la incertidumbre y a posibles cambios en las condiciones epidemiológicas.

---

## Limitaciones

Este proyecto presenta algunas limitaciones importantes:

* Los resultados dependen de la calidad del registro oficial.
* La base puede contener valores faltantes o cambios en la estructura de variables.
* Los modelos identifican asociaciones estadísticas, no causalidad directa.
* Algunas variables relevantes, como vacunación, variantes del virus o saturación hospitalaria, no fueron incorporadas.
* Los pronósticos de series de tiempo son aproximaciones de corto plazo y aumentan su incertidumbre conforme crece el horizonte.

---

## Tecnologías utilizadas

El análisis fue realizado principalmente con Python y librerías orientadas a análisis de datos, modelado econométrico y visualización.

Principales librerías utilizadas:

```text
pandas
numpy
matplotlib
seaborn
statsmodels
scipy
scikit-learn
openpyxl
```

---

---

## Notas importantes

Los resultados del proyecto deben interpretarse como evidencia estadística de asociación. No se afirma causalidad directa entre las variables explicativas y los desenlaces analizados.

El análisis tiene fines académicos y busca demostrar la aplicación de modelos econométricos a un problema real de salud pública.

---

## Autores

* Gutiérrez Ramírez Alana Sofia
* Reyes Maldonado Oscar Romario
* Rivera García Axel Maximiliano

Instituto Politécnico Nacional
Escuela Superior de Cómputo
Licenciatura en Ciencia de Datos

---

## Profesor

Jiménez Alcantar Daniel

---

## Licencia

Este proyecto fue desarrollado con fines académicos.
El uso, modificación o distribución del contenido debe citar adecuadamente a los autores y respetar la fuente oficial de los datos.
