# Definición del problema y dataset

## Formulación

Para cada estación $s$ y hora $t$, la entrada es la ventana de las últimas $L = 168$ horas de $d = 4$ variables
(velocidad, seno y coseno de la dirección, y temperatura):

$$\mathbf X_{s,t} = \big[\mathbf x_{s,t-L+1}, \dots, \mathbf x_{s,t}\big]^\top \in \mathcal X = \mathbb R^{L \times d}$$

La salida es la velocidad observada 24 horas después del final de la ventana:

$$y_{s,t} = v_{s,t+24} \in \mathcal Y = \mathbb R_{\ge 0}$$

Se busca $f_\theta : \mathcal X \to \mathcal Y$ que minimice el error cuadrático medio de entrenamiento

$$\mathcal L(\theta) = \frac{1}{n} \sum_{(s,t)} \big(y_{s,t} - f_\theta(\mathbf X_{s,t})\big)^2 .$$

El pronóstico es **directo a un horizonte** ($h = 24$): una sola salida, sin realimentar predicciones.

**Tipo de problema: 3, series de tiempo.** La entrada contiene la historia del propio objetivo (los rezagos de la
velocidad) y el orden temporal no puede alterarse sin fuga de información. Hay además una dimensión espacial (varias
estaciones), pero en este entregable cada estación se trata como una serie más de un modelo global; la estructura
espacio-temporal queda para entregables posteriores.

**Hipótesis.** $H_0$: los tres modelos base tienen el mismo RMSE esperado en 2025. $H_1$: al menos uno difiere. Se
contrasta además cada modelo contra la persistencia $\hat v_{t+24} = v_t$.

## Dataset

**Origen.** Mediciones horarias de las estaciones automáticas del IDEAM publicadas en el Portal Nacional de Datos
Abiertos (datos.gov.co) con licencia CC BY-SA 4.0. El panel se construyó en el proyecto de Machine Learning
{cite}`rollo2026ml`, cuyo ETL limpia, alinea por hora y une las cuatro variables sin usar información futura.

**Relevancia.** El pronóstico del viento a un día sirve para planear la operación de parques eólicos (La Guajira
concentra el mayor potencial eólico del país), puertos y aeropuertos de la región.

| Característica | Valor |
|---|---|
| Archivo | `panel_multivariado_causal.csv` (CSV, 238 MB) |
| SHA-256 | `dd0fdfad3408627d7dad77ba072e0060a9a72e1ed2177cf5ff4445c8e86c12fb` |
| Filas | 1 308 726 estación-horas |
| Estaciones | 76, en 7 departamentos del Caribe |
| Periodo | 1 de enero de 2020 a 31 de diciembre de 2025 |
| Frecuencia | Horaria |
| Variables usadas | Velocidad (m/s), dirección (°, codificada como seno y coseno), temperatura (°C) |
| Variable descartada | Presión (hPa), por faltante no aleatorio (ver EDA) |

**Estructura de dependencia.** Fuerte autocorrelación dentro de cada estación (ciclo diario dominante) y correlación
espacial entre estaciones cercanas. Por eso la estación, y no la hora, es la unidad de independencia en las pruebas.

**Ruido de medición.** Los anemómetros tienen un umbral de arranque (típicamente del orden de 0.5 m/s) y una resolución de
décimas, lo que pone un piso al error alcanzable en vientos débiles.

**Ventanas resultantes** (salida de `src/datos.py`):

| Conjunto | Años | Ventanas | Estaciones |
|---|---|---|---|
| Entrenamiento | 2020–2023 | 387 770 | 60 |
| Validación | 2024 | 78 012 | 45 |
| Prueba | 2025 | 211 408 | 62 |
