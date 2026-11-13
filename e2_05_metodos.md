# Métodos (Entregable 2)

Este capítulo describe cómo se entrenaron y compararon los modelos. Todo lo que aquí se afirma quedó fijado en el
preregistro antes de ver resultados de prueba (`prerregistro_e2.md`, publicado con la etiqueta `prerregistro-e2`), y
cada cambio posterior está registrado como una enmienda fechada con su motivo.

## Preregistro y enmiendas

El preregistro fija las preguntas, las hipótesis, las métricas, la partición, el presupuesto de ajuste, la búsqueda de
arquitectura y las pruebas estadísticas. Durante el trabajo hubo siete enmiendas. Las seis primeras se hicieron antes de elegir
finalistas y sin abrir el conjunto de prueba; la séptima agrega un análisis secundario (la ablación) después de ver
los resultados de prueba, y así se declara:

| Enmienda | Fecha | Qué cambió | Motivo |
|---|---|---|---|
| 1 | 7-nov | Ajuste previo y fase 1 con el 10 % de las ventanas de ajuste; Mamba en espera | Costo medido de una época |
| 2 | 8-nov | GRU y LSTM: muestra aleatoria de 200 de las 1 000 arquitecturas (semilla 11) | Las recurrentes anchas tardan hasta 670 s |
| 3 | 9-nov | La misma muestra para TCN, KAN y Transformer; fase 2 con k = 10 y un origen; uso de Colab | Presupuesto fijo de cómputo para el E2 y el E3 |
| 4 | 9-nov | Mamba se sustituye por PatchTST (11.º del ranking) | Mamba no tiene núcleos para Windows y en PyTorch puro es inviable |
| 5 | 9-nov | Configuración de Informer | Adaptaciones al panel (sin calendario) |
| 6 | 9-nov | Configuración de STGCN con subgrafo de 8 vecinas | El panel no es una red sincronizada |
| 7 | 12-nov | Estudio de ablación de TCN (sin dilatación, sin residuales, sin covariables) | Análisis secundario, declarado después de ver la prueba |

## Partición con origen móvil

La serie se parte cronológicamente con **origen móvil**: cuatro orígenes trimestrales de
validación en 2024 (1-ene, 1-abr, 1-jul y 1-oct) y cuatro de prueba en 2025, con ventana de entrenamiento expansiva y
un **embargo de 24 h** (igual al horizonte) entre el último objetivo usado para entrenar y el primer pronóstico
evaluado. En cada origen los modelos **se reentrenan desde cero** y el escalado se ajusta solo con el entrenamiento de
ese origen, de modo que ningún dato posterior al origen influye en el pronóstico. La batería de pruebas de fuga
(`tests/test_fuga.py`, 9 pruebas) verifica estas propiedades con datos sintéticos.

## Modelos

| Modelo | Familia | Arquitectura | Referencia |
|---|---|---|---|
| Transformer | Atención | Búsqueda (sección 14) | {cite}`vaswani2017` |
| KAN | MLP profundo | Búsqueda | {cite}`liu2024kan` |
| TCN | Convolucional | Búsqueda | {cite}`bai2018` |
| STGCN | Grafos | Publicada (enmienda 6) | {cite}`yu2018` |
| Chronos | Preentrenado | Pesos oficiales | {cite}`ansari2024` |
| GRU | Recurrente | Búsqueda | {cite}`cho2014` |
| Informer | Atención | Publicada (enmienda 5) | {cite}`zhou2021` |
| DLinear | MLP profundo | Publicada | {cite}`zeng2023` |
| Moirai | Preentrenado | Pesos oficiales | {cite}`woo2024` |
| PatchTST | Atención | Publicada (enmienda 4) | {cite}`nie2023` |
| MLP, LSTM, CNN 1D | Base del E1 | Búsqueda | E1 |

Los baselines son la persistencia (que con h = 24 coincide con el naive estacional diario), el naive estacional
semanal, la media de la misma hora en los últimos 7 días, ETS por estación, y como clásicos fuertes una regresión
ridge y gradient boosting sobre rezagos de la velocidad, reentrenados en cada origen.

Todas las arquitecturas propias (atención ProbSparse de Informer, bloques ST-Conv de STGCN, parches de PatchTST, capas
de KAN) se programaron desde su formulación matemática en `src/modelos_e2.py`; ninguna se importa de una biblioteca.

## Ajuste de hiperparámetros y búsqueda de arquitectura

Todos los modelos entrenables reciben el **mismo presupuesto de ajuste previo**: 20 configuraciones aleatorias
(semilla 7) de tasa de aprendizaje (log-uniforme en [1e-4, 3e-3]), weight decay ({0, 1e-5, 1e-4}) y tamaño de lote
({512, 1 024, 2 048}), evaluadas con la arquitectura de referencia (64, 64, 64) con dropout (0, 1, 0) en el último origen
de validación.

Los modelos sin arquitectura definida en su artículo pasan por la búsqueda en dos fases de la sección 14 del
enunciado. El espacio es de 3 capas con anchos {16, 32, 64, 128, 256} y dropout de 0.2 en cualquier posición:
5³ × 2³ = 1 000 arquitecturas.

* **Fase 1 (cribado):** una semilla (2 000), 12 épocas como máximo con paciencia 3. MLP y CNN evaluaron las 1 000;
  GRU, LSTM, TCN, KAN y Transformer una **muestra aleatoria simple de 200**, la misma para los cinco (enmiendas 2 y 3).
  La búsqueda aleatoria sobre un espacio fijo es un método reconocido de ajuste {cite}`bergstra2012`.
* **Fase 2 (confirmación):** las 10 mejores de la fase 1 más la referencia, con presupuesto completo (50 épocas,
  paciencia 5) y 5 semillas (2 001–2 005), en el último origen de validación.
* **Regla de selección (fijada antes):** menor RMSE de validación medio en la fase 2; si otra arquitectura queda a menos
  de δ/2 = 0.01 m/s, gana la de menos parámetros.

**Efecto del muestreo.** Como MLP y CNN tienen la fase 1 completa, se les aplicó la misma muestra de 200 para medir qué
se pierde: la mejor arquitectura de la muestra quedó a **0.001 m/s** (MLP) y **0.004 m/s** (CNN) de la mejor de las
1 000, muy por debajo del margen de relevancia δ = 0.02 m/s. Los 10 finalistas sí cambian (comparten 1 y 0), lo que es
esperable con una muestra del 20 % y muestra que el ranking dentro del cribado es ruidoso; por eso la fase 2 confirma
con 5 semillas (`resultados/e2/sensibilidad_muestra.json`).

## Corridas finales

Con la configuración elegida solo en validación, cada modelo se reentrena en los cuatro orígenes de prueba de 2025
con 10 semillas (1 000–1 009). Las conclusiones del E2 usan las 6 declaradas en el preregistro (1 000–1 005), y las
otras 4 se reportan aparte (sirven al E3, que exige al menos 10). Se guardan las predicciones de cada ventana y los
modelos entrenados.

## Inferencia estadística

La **estación es la unidad de independencia**. Las familias confirmatorias son F1 (cada modelo del Top 10 contra el
mejor clásico, ridge) y F2 (contra la persistencia), con corrección de Holm {cite}`holm1979` dentro de cada familia.

* **Prueba principal:** Wilcoxon de rangos con signo sobre la diferencia de RMSE por estación, con la diferencia media
  y la correlación biserial de rangos como tamaño del efecto. Un modelo "supera con relevancia" si el p ajustado es
  menor que 0.05 **y** la diferencia media es mayor que δ = 0.02 m/s.
* **Diebold–Mariano** con la corrección de Harvey, Leybourne y Newbold {cite}`diebold1995,harvey1997` sobre la serie
  horaria de diferencias de pérdida, con varianza HAC de 23 rezagos (h − 1).
* **Giacomini–White** condicional {cite}`giacomini2006` y **Clark–West** para modelos anidados frente a la persistencia
  {cite}`clark2007`.
* **Model Confidence Set** {cite}`hansen2011` con bootstrap por bloques de una semana, y **Friedman** {cite}`friedman1937`
  con post-hoc de Nemenyi para el ranking global.
* **Equivalencia (RQ2):** TOST con ±δ {cite}`schuirmann1987`.

## Limitaciones de cómputo

El cómputo se planificó con un presupuesto fijo, que también debía alcanzar para el Entregable 3:

| Recurso | Uso |
|---|---|
| PC del equipo: NVIDIA RTX 4060 Ti (8 GB) | Solo de noche y en horarios disponibles; fase 1 de MLP, CNN, GRU, LSTM y parte de la TCN |
| Google Colab Pro+: NVIDIA A100 (40 GB) | 600 unidades de cómputo por un mes, sin posibilidad de comprar más; ≈ 6.7 unidades por hora de A100 |

Cada resultado guarda el equipo y la GPU que lo produjo (`resultados/e2/busqueda/hardware.jsonl`). El presupuesto de
cada modelo se mide en arquitecturas, épocas y semillas, no en horas, de modo que no depende del equipo; los tiempos de
entrenamiento se reportan junto con su GPU. Las decisiones tomadas por el presupuesto (muestra de 200 arquitecturas,
fase 2 con 10 finalistas y un origen) se declararon antes de ver los resultados que podían afectar, y su efecto se
midió con el análisis de sensibilidad descrito arriba. El consumo real de unidades y horas por modelo se reporta en el
capítulo de resultados.
