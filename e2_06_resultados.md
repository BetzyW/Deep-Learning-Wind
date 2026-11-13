# Resultados (Entregable 2)

Todos los números de este capítulo salen de las predicciones de prueba de 2025: 215 286 ventanas de 62 estaciones en
cuatro orígenes trimestrales, con cada modelo reentrenado desde cero en cada origen. El pronóstico de un modelo de
Deep Learning en cada ventana es la media de sus 6 semillas declaradas (1000–1005). Las tablas se generan con
`src/tablas_libro.py` a partir de los archivos de resultados, así que no hay números copiados a mano.

## Replicación del paper guía

La replicación de DLinear en ETTh1 se presenta en el capítulo de revisión: con las condiciones originales del
artículo obtuvimos un MSE de `0.379` (IC 95 % [`0.370`, `0.389`], 5 semillas) frente al `0.375` reportado, y se
cumplió el criterio de éxito fijado antes de correrla.

## Benchmark

```{include} tablas/e2_principal.md
```

El RMSE agregado y su intervalo usan todas las ventanas; el intervalo remuestrea estaciones (2 000 réplicas), porque
la estación es la unidad de independencia. Los intervalos son anchos y se solapan casi por completo: las estaciones
difieren mucho entre sí (de `0.53` a más de `1.0` m/s), y esa variación domina. Por eso las comparaciones entre
modelos se hacen pareadas por estación, que es lo que hacen Wilcoxon y Friedman. El MASE divide el MAE de cada modelo
entre el de la persistencia.

Tres grupos se distinguen con claridad:

1. **Los modelos de Deep Learning entrenados desde cero y los clásicos fuertes** quedan todos entre `0.692` y `0.704`
   m/s, una franja de `0.012` m/s.
2. **Las líneas base triviales** (media de la misma hora, ETS por estación, persistencia y naive semanal) quedan
   entre `0.04` y `0.29` m/s por encima.
3. **Los preentrenados con el modelo base congelado** quedan lejos: Chronos (`0.839`) apenas mejora a la persistencia
   y Moirai (`0.989`) es el peor modelo de la tabla.

## Pruebas por contraste (familias F1 y F2)

Cada modelo del Top 10 y los tres modelos base se comparan con el mejor clásico (Ridge, familia F1) y con la
persistencia (familia F2) usando tres pruebas: Wilcoxon de rangos con signo sobre la diferencia de RMSE por estación
(no paramétrica, la principal), Diebold–Mariano con la corrección de Harvey, Leybourne y Newbold sobre la serie horaria
de diferencias de pérdida cuadrática (paramétrica, con varianza de Newey–West) y Clark–West (para modelos anidados),
más Holm dentro de cada familia {cite}`diebold1995,harvey1997,clark2007,holm1979`. Un Δ negativo favorece al modelo.

```{include} tablas/e2_familias.md
```

**RQ1.** Siete modelos le ganan a Ridge con significancia tras Holm (Transformer, TCN, Informer, MLP, STGCN, KAN y
PatchTST), y las tres pruebas coinciden en la dirección. Pero la mejor diferencia es de `0.0073` m/s, menos de la
mitad del margen de relevancia declarado (δ = `0.02` m/s). **Ningún modelo supera a Ridge por más de δ**, así que
H1 se rechaza: las ganancias son reales pero pequeñas. Frente a la persistencia, todos los modelos entrenados desde
cero ganan por unos `0.15` m/s (cerca del 18 % del RMSE), con p < `0.001` en todas las pruebas.

La prueba de Giacomini–White condicional rechaza la igualdad en todos los contrastes: la ventaja relativa de cada
modelo depende de la hora anterior, es decir, cambia en el tiempo. La sección de estabilidad del capítulo siguiente
lo muestra con la prueba de fluctuación.

## Procedimientos globales

**Friedman con Nemenyi** (estaciones como bloques, 15 modelos): χ² = `524.6`, p < `0.001`; la diferencia crítica de
Nemenyi es `2.72` rangos. TCN (`3.65`) y Transformer (`3.68`) encabezan; KAN, Informer, STGCN y MLP quedan dentro de
la diferencia crítica del primero; Ridge (`8.13`) queda fuera de ella.

**Model Confidence Set** (bootstrap por bloques de 168 h, α = 0.10) {cite}`hansen2011`: el conjunto de modelos
indistinguibles del mejor contiene 9 modelos (Transformer, TCN, Informer, MLP, STGCN, KAN, LSTM, GRU y PatchTST).
Ridge queda fuera por poco (p = `0.054`), y CNN, DLinear, Chronos, Moirai y la persistencia quedan fuera.

## RQ2: preentrenados frente a modelos desde cero

El mejor preentrenado (Chronos) es peor que el mejor modelo desde cero (Transformer) por `0.127` m/s de RMSE medio por
estación, con IC 90 % de [`0.087`, `0.166`]. El TOST con ±δ no declara equivalencia (p ≈ `1.0`) y Wilcoxon confirma
la dirección (p < `0.001`) {cite}`schuirmann1987`. **H2 se rechaza**: en nuestra adaptación, los preentrenados no son
equivalentes a los modelos desde cero, sino claramente peores. El capítulo de discusión explica por qué esto refleja
sobre todo la forma de resumir su representación y no necesariamente a los modelos.

## RQ3: la limitación del paper guía

En las 11 estaciones con al menos 30 ventanas con viento observado de 6 m/s o más (4 091 ventanas), todos los
modelos subestiman: el sesgo es negativo en todas las estaciones y Wilcoxon unilateral lo confirma tras Holm.

```{include} tablas/e2_f3.md
```

**H3 se cumple**: la subestimación de los vientos fuertes no es propia de DLinear (`−1.17` m/s), aparece en todos los
modelos con magnitudes casi iguales (de `−1.15` a `−1.28` m/s en los entrenados desde cero) y también en Ridge
(`−1.21`). Que hasta la persistencia subestime (`−0.76`) muestra que parte del efecto se debe a condicionar en la
observación: cualquier pronóstico que se acerque a la media queda por debajo cuando el valor observado es extremo.
Es una limitación de todos los modelos entrenados con error cuadrático, y es el punto de partida del modelo del E3.

## Análisis del espacio de arquitecturas

```{include} tablas/e2_busqueda.md
```

"Factor principal" es el factor con la mayor proporción de la suma de cuadrados en el modelo lineal de efectos
principales e interacciones de dropout; "DE entre arquitecturas / entre semillas" divide la desviación estándar del
RMSE entre arquitecturas en la fase 1 entre la desviación entre semillas en la fase 2; "Equivalentes (TOST)" cuenta los
finalistas equivalentes al mejor con ±δ. Los mapas de calor de la fase 1 están en
`resultados/e2/busqueda_analisis/{modelo}_mapa_fase1.png`.

Lo que dicen estos resultados:

- **La arquitectura importa poco.** En GRU la dispersión entre arquitecturas es igual a la dispersión entre semillas
  (razón `1.0`), y en KAN, LSTM, Chronos y Moirai apenas la supera (`1.2`–`1.9`): para esos modelos la elección de
  anchos y dropout es casi irrelevante en este problema. Solo en CNN (`18.1`), MLP (`6.1`) y TCN (`5.0`) la
  arquitectura pesa claramente, sobre todo por el ancho de la primera o segunda capa.
- **Casi todos los finalistas son equivalentes al mejor.** En 7 de los 9 modelos, 10 u 11 de los 11 finalistas son
  equivalentes al mejor con ±δ. El ganador es "el mejor entre equivalentes", no un óptimo claro.
- **El cribado de la fase 1 fue poco informativo.** La correlación de rangos entre fases en los finalistas es baja o
  negativa en 6 de 9 modelos (en TCN, ρ = `−0.90`): con una semilla y 12 épocas, el orden de la fase 1 no anticipa
  el de la fase 2. Lo declaramos como limitación: la fase 1 fue un cribado, como estaba preregistrado, y el ganador es
  el mejor entre los finalistas, no necesariamente el mejor de las 1 000.
- **Ganadores en el borde.** Los ganadores de MLP, LSTM, TCN, KAN y Chronos tienen al menos una capa en 16 o 256;
  ninguno tiene todas las capas en un extremo, así que no hay señal de que el espacio se quedara corto.
- **El costo real** de las fases de búsqueda fue de `44.6` horas de GPU entre el PC y Colab (la estimación previa era
  de 40 horas para las fases 1 y 2), sin contar las corridas finales.

## Techo de desempeño, potencia y varianza

**Techo.** Usamos dos cotas: el promedio de los pronósticos de los 13 modelos diversos entrenados desde cero y
clásicos (RMSE `0.690`), y un oráculo que elige el mejor modelo en cada estación mirando la prueba (`0.688`, cota
optimista). Con la persistencia como base, la distancia normalizada del mejor modelo al techo es γ = `0.011` con el
conjunto y γ = `0.023` con el oráculo: **el mejor modelo ya cubrió el 98–99 % del margen alcanzable** y cualquier
modelo nuevo con estas entradas solo puede ganar milésimas. Además, el RMSE del mejor modelo relativo a la persistencia
se correlaciona con la predictibilidad intrínseca de cada estación (Ω de la entropía de permutación; Spearman `0.46`,
p = `0.001`, 47 estaciones): las estaciones más predecibles son las que más ganan.

**Potencia.** Con 62 estaciones, Holm sobre 10 contrastes y potencia de 0.80, la diferencia mínima detectable en el
contraste principal (Transformer frente a Ridge, desviación de las diferencias por estación `0.0096` m/s) es de
`0.0045` m/s, cuatro veces menor que δ. El estudio tenía potencia de sobra para detectar una diferencia relevante; si
ninguna aparece, es porque no existe. La dispersión entre semillas de una corrida final es de `0.0043` m/s, de modo que
el RMSE promedio de 6 semillas tiene un error estándar de `0.0017` m/s.

**Componentes de varianza.** En el RMSE de las corridas finales de los modelos entrenados desde cero, el origen
(la época del año) explica el `99.5` % de la variación, el modelo el `0.27` %, la interacción el `0.06` % y las
semillas el `0.19` %. Elegir un modelo u otro cambia mucho menos el error que la estación del año en que se
pronostica.

## Errores y casos difíciles

Tomamos los errores grandes (el 10 % de mayor |error| de cada modelo) y etiquetamos 160 por modelo, estratificados por
trimestre, con categorías fijas y excluyentes aplicadas por reglas (`src/errores_e2.py`).

```{include} tablas/e2_errores.md
```

Las rampas (cambios de 2 m/s o más en 24 h) y los vientos fuertes son menos del 2 % de las ventanas, pero cerca del 10
al 17 % de los errores grandes cada una: están sobrerrepresentadas entre 6 y 10 veces. Las calmas, en cambio, son el
29 % de las ventanas y solo el 9–12 % de los errores grandes. Casi la mitad de los errores grandes no cae en ninguna
categoría, lo que indica variabilidad que las entradas no explican.

Los modelos se equivocan en los mismos casos: el 86 % de los errores grandes de TCN también lo es de Transformer y
KAN, y el índice de Jaccard entre los errores grandes de los modelos profundos y los de Ridge es de `0.77`–`0.79`.
Cuando todos fallan en las mismas ventanas, el límite está en los datos (dificultad intrínseca), no en la
arquitectura, lo que coincide con el γ cercano a cero.

## Estudio de ablación

*(Se completa con `python src/ablacion.py --analizar` al terminar las corridas: TCN sin dilatación, sin conexiones
residuales y sin covariables, 4 orígenes × 6 semillas, con Wilcoxon, Diebold–Mariano y Holm.)*

## Reproducción independiente

Recalculamos el RMSE y el MAE de las 520 corridas finales de los 13 modelos de Deep Learning directamente desde las predicciones guardadas y los datos
del panel, por un camino de código distinto al del entrenamiento: la discrepancia máxima es de `1.3e-07` m/s, solo
redondeo. *(Se agrega el reentrenamiento en otro equipo de corridas hechas en Colab, `src/reproducir.py
--reentrenar`.)*
