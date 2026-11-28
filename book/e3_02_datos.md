# Datos: EDA y auditoría

Los datos del panel principal son los mismos del E1 y el E2: el EDA completo está en el capítulo de análisis
exploratorio del E1 y la ficha, la partición, la predictibilidad y la auditoría de fuga, en el capítulo de datos del
E2. Aquí resumimos lo esencial, agregamos un análisis exploratorio específico del modelo original (la estructura de
análogos diarios) y la ficha de los tres conjuntos adicionales. La auditoría de fuga actualizada está en el capítulo de
trazabilidad.

## Data Card del panel principal

| Campo | Contenido |
|---|---|
| Fuente y licencia | Estaciones automáticas del IDEAM, Portal de Datos Abiertos de Colombia {cite}`ideam2025` (CC BY-SA 4.0) |
| Construcción | Panel horario del proyecto de Machine Learning {cite}`rollo2026ml`: velocidad, dirección y temperatura alineadas por estación y hora, con limpieza causal |
| Cobertura | 76 estaciones de 7 departamentos del Caribe colombiano, enero de 2020 a diciembre de 2025; 62 estaciones con ventanas válidas en las particiones del E2 |
| Entrada y objetivo | Ventana de 168 h de velocidad (m/s), seno y coseno de la dirección y temperatura (°C); objetivo: velocidad 24 h después |
| Partición | Origen móvil trimestral con embargo de 24 h: desarrollo y validación en 2024, prueba en los 4 trimestres de 2025 (215 286 ventanas) |
| Sesgos conocidos | La red cambia mucho entre años (de 10–16 estaciones en el primer semestre de 2024 a 56 a finales de 2025); el viento fuerte es raro (ver abajo); una estación se excluye por anemómetro bloqueado |
| Usos no previstos | No sustituye un pronóstico oficial; no se validó para eventos extremos (huracanes) |

## EDA específico del modelo original: la estructura de análogos diarios

PADRE-TCN usa como anclas la misma hora de los seis días anteriores. Esta decisión tiene que estar justificada por los
datos: si el viento de mañana a esta hora no se pareciera más al de los días anteriores a esta hora que al de otras
horas, las anclas no tendrían sentido. Lo medimos en el periodo de entrenamiento (2020–2024), sin tocar la prueba
(`src/eda_e3.py`).

```{figure} figuras/e3_acf_anclas.png
:name: fig-acf-anclas
:width: 100%

Autocorrelación de la velocidad horaria del viento en función del rezago, para las 66 estaciones con al menos 120 días
de datos antes de 2025. La línea es la mediana entre estaciones y la banda, el rango intercuartílico. Las líneas
punteadas marcan los múltiplos de 24 h, que son las horas de las anclas diarias.
```

La {numref}`fig-acf-anclas` muestra un **ciclo diario muy marcado**: la autocorrelación cae casi a 0 a las 12 h
(`−0.003` en la mediana) y vuelve a subir a `0.63` a las 24 h, `0.57` a las 48 h y alrededor de `0.50` hasta los 7
días. Entre los picos, en las horas que no son múltiplos de 24, la correlación es nula o negativa. Es el ciclo de la
brisa del Caribe, que ya habíamos encontrado en la descomposición STL del E1.

La comparación directa con el objetivo lo confirma: correlación de Pearson entre el objetivo v(t + 24) y el valor de
cada ancla, en la mediana de las estaciones.

| Candidato a análogo | Ancla 1 (t) | Ancla 2 (t − 24) | Ancla 3 | Ancla 4 | Ancla 5 | Ancla 6 | Ancla 7 | t − 12 | t − 36 |
|---|---|---|---|---|---|---|---|---|---|
| Correlación con v(t + 24) | `0.63` | `0.57` | `0.54` | `0.52` | `0.51` | `0.50` | `0.50` | `−0.04` | `−0.07` |

**Interpretación y consecuencias para el diseño.** (1) La misma hora de los días anteriores es, con diferencia, el
mejor análogo disponible dentro de la ventana; una hora a medio día de distancia no aporta nada. (2) La información
decae con la antigüedad del día pero no desaparece: el día 6 todavía tiene correlación `0.50`. Eso justifica usar
**K = 6 anclas** y que el modelo aprenda a **pesarlas** (el ancla más reciente debería pesar más, que es lo que
aprendió: capítulo de implementación). (3) La correlación no llega a 1: los días anteriores discrepan entre sí, y esa
discrepancia es justamente la que el modelo usa como medida de incertidumbre.

**El viento fuerte es raro.** En la mediana de las estaciones, menos del 0.01 % de las horas tienen viento de 6 m/s o
más; en las estaciones más ventosas (La Guajira) llega al 17 %. En la prueba de 2025 son 4 136 de las 215 286
ventanas (1.9 %). Con tan pocos casos, la pérdida cuadrática apenas los pesa, lo que explica la subestimación que
comparten todos los modelos (capítulo de evaluación ampliada).

## Resumen del EDA del E1 y el E2 que sigue vigente

* **Ciclo diario:** explica cerca de dos tercios de la varianza (descomposición STL), y la autocorrelación es
  significativa hasta 168 h; de ahí la ventana de 168 h y la persistencia a 24 h como baseline obligatorio.
* **Faltantes:** la presión falta en el 18 % de las horas y se descartó; las demás variables se rellenan hacia
  adelante como máximo 3 h y las ventanas incompletas no se usan.
* **Estacionariedad:** casi todas las estaciones caen en el escenario 3 de ADF y KPSS (sin raíz unitaria, pero con una
  media que cambia con el ciclo diario y la época del año), así que no se diferencia y se estandariza.
* **Heterocedasticidad:** efecto ARCH en todas las estaciones; por eso las pruebas usan Diebold–Mariano con HLN y
  bootstrap por bloques, y los intervalos de PADRE-TCN se escalan por ventana.
* **Quiebres de nivel:** más de la mitad de las estaciones tiene un cambio de nivel local (sensores o entorno), no
  regional; la ventana larga le da al modelo el nivel reciente.
* **Modelo global:** las estaciones son heterogéneas (de 0.5 a 5.7 m/s de media), pero cada una tiene pocos datos
  para una red propia; se usa un solo modelo y la estación es la unidad de independencia en las pruebas.
* **Predictibilidad:** la predictibilidad espectral Ω de cada estación anticipa su dificultad (ρ = `−0.80` con el
  error relativo de la persistencia): las estaciones con ciclo diario marcado son las fáciles.

## Ficha de los conjuntos adicionales

| Campo | ETTh1 | SDWPF | NOAA ISD-Lite, Kansas |
|---|---|---|---|
| Fuente | Zhou et al. (2021) {cite}`zhou2021`, repositorio ETDataset | Zhou et al. (2024), *Scientific Data*; figshare | NOAA Integrated Surface Database, AWS Open Data |
| Licencia | CC BY-ND 4.0 (solo evaluación, sin redistribuir versiones modificadas) | CC BY 4.0 | Dominio público |
| Tamaño | 1 serie, 7 variables, 17 420 h | 134 turbinas, 245 días cada 10 min (agregado a horario) | 10 de 34 estaciones, 2021–2024 |
| Objetivo | Temperatura del aceite (OT) | Potencia activa (kW) | Velocidad del viento (m/s) |
| Huella | SHA-256 en `experiments/e3_sha256.csv` | Ídem (y MD5 igual al publicado) | Ídem (204 archivos) |
| Sesgo conocido | No estacionaria: la media baja de `15.9` en el entrenamiento a `2.3` en la prueba | Potencia acotada por la curva de la turbina | Solo 10 estaciones por cómputo, elegidas al azar |

Los detalles de la elección (antes de ver resultados) y de las desviaciones están en
`experiments/conjuntos_adicionales.md` y en el capítulo de evaluación ampliada.
