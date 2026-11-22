# Resultados

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

La misma tabla con el formato de la plantilla A.2 del enunciado agrega la variación entre semillas, la calibración,
la robustez y el costo. La columna de media ± DE trata cada semilla por separado, mientras que la del IC usa el
pronóstico promedio de las 6 semillas, que es un poco mejor. Para la calibración damos la pendiente b de
Mincer–Zarnowitz y el p conjunto de a = 0 y b = 1. El índice de robustez es el aumento relativo medio del RMSE ante
las tres perturbaciones; solo se midió en los tres mejores. La latencia está medida en CPU con 4 hilos, por ventana,
con lote 1 y con lote 1 024 (`src/costo_inferencia.py`); entre dos mediciones varía hasta un 30 % con lote 1, así que
solo importa el orden de magnitud. En Chronos y Moirai, el costo es solo el del cabezal. Como
referencia del modelo base, su pronóstico sin entrenar tarda unos 3 ms por ventana (Chronos, en una A100) y de 0.3 a
0.5 ms (Moirai, en una L4).

```{include} tablas/e2_a2.md
```

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
equivalentes a los modelos desde cero, sino claramente peores.

### Chronos y Moirai sin entrenar (análisis secundario)

Para saber si el resultado se debe a los modelos o a nuestra adaptación, evaluamos Chronos y Moirai con sus pesos
oficiales y su propio decodificador, sin ningún entrenamiento (*zero-shot*, enmienda 9). Usan el mismo contexto de
168 h de la velocidad, y el pronóstico es la mediana de sus trayectorias. Moirai corrió en una L4 de Colab (de 13 a 25 s por
origen) y Chronos en una A100 (de 1.5 a 4 min por origen), sobre las mismas ventanas que el benchmark (`src/preentrenados.py`, `src/cero_disparos.py`).
La tabla da el RMSE (m/s) de cada origen de prueba de 2025; en los modelos de Deep Learning se promedia el pronóstico
de las 6 semillas declaradas:

```{include} tablas/e2_cero_rmse.md
```

Las pruebas, con la estación como unidad. La diferencia es la del modelo sin entrenar menos la del rival, así que un
valor negativo lo favorece. Es un análisis exploratorio, así que no aplicamos Holm:

```{include} tablas/e2_cero_pruebas.md
```

- **Sin entrenar son mejores que con nuestro cabezal**, en los 4 orígenes. En Moirai la mejora es grande y clara
  (`0.238` m/s por estación). En Chronos el RMSE global baja en todos los orígenes, pero por estación la diferencia
  no es significativa (p = `0.219`), porque en algunas estaciones el cabezal sí le ayuda.
- **Los dos le ganan a la persistencia**, que Moirai con cabezal no lograba.
- **Siguen lejos de los modelos entrenados.** Moirai sin entrenar, el mejor de los dos, queda `0.038` m/s por encima
  del Transformer y `0.031` m/s por encima de Ridge. El TOST no declara equivalencia con ninguno. **H2 se rechaza
  también con los modelos en su uso original.**

Los dos modelos también dan cuantiles, así que medimos su intervalo del 80 % (cuantiles 0.1 a 0.9) y el sesgo en las
4 136 ventanas con viento observado de 6 m/s o más:

```{include} tablas/e2_cero_cobertura.md
```

El intervalo de Moirai sale bien calibrado sin ningún ajuste (`81.3` %). Es incluso más cercano al 80 % que los
intervalos conformales de los modelos entrenados, que cubren cerca del `77` % (sección de intervalos). El de Chronos
es demasiado estrecho. En vientos
fuertes, Moirai subestima `0.37` m/s menos que los modelos entrenados con error cuadrático, que coinciden en `−1.23`
m/s. El capítulo de discusión explica qué dice esto sobre la adaptación y qué sugiere para el E3.

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

Los coeficientes de cada factor (anchos y dropout por capa, con interacciones de dropout), con su IC 95 % y su p
ajustado por Holm, están en `resultados/e2/busqueda_analisis/{modelo}_coeficientes.csv` y `{modelo}_anova.csv`.

**Frente de Pareto.** La figura muestra el costo y el error de cada arquitectura de la fase 1 (`src/pareto_ganador.py`).
La línea es el frente de Pareto: las arquitecturas que ninguna otra mejora a la vez en parámetros y en error.

```{figure} figuras/e2_pareto.png
:name: fig-pareto
RMSE de validación de la fase 1 frente al número de parámetros (escala logarítmica). Los círculos son los finalistas
y la estrella es la arquitectura elegida. En GRU y Transformer la elegida es la referencia (64, 64, 64) / (0, 1, 0),
que entra directo a la fase 2 y por eso no aparece en la fase 1.
```

- **De las 7 elegidas que pasaron por la fase 1, 5 están sobre el frente** (MLP, LSTM, TCN, Chronos y Moirai). Las de
  CNN y KAN quedan algo por encima. La regla de desempate, que prefiere la de menos parámetros entre las que empatan
  a menos de δ/2, empuja la elección hacia la parte barata del frente.
- **En CNN el frente sigue bajando hasta los modelos más grandes**, pero la ganancia final es de milésimas. Pagar diez
  veces más parámetros por menos de `0.01` m/s no compensa.
- **En Chronos y Moirai el frente es casi plano.** Un cabezal pequeño rinde igual que uno grande, porque el límite lo
  pone la representación congelada.

**Sesgo del ganador.** La "mejor" de la fase 2 se eligió por tener el menor RMSE en un solo origen, y el mínimo de
varios valores ruidosos siempre es optimista. Lo medimos con la validación de la sección siguiente. Comparamos la
ventaja de la mejor frente a la elegida y a la peor, primero en el origen donde se eligió y luego en los otros tres
(`resultados/e2/busqueda_analisis/sesgo_ganador.csv`). En promedio, la ventaja baja de `0.011` m/s a `0.008` m/s, es
decir, cerca de un tercio era suerte. La ventaja se mantiene en 13 de los 16 contrastes. En CNN, Transformer y Moirai
se invierte frente a una de las rivales. Por eso no reportamos el RMSE de validación del ganador como estimación de su
desempeño: el desempeño se mide solo en la prueba, con las corridas finales.

## Validación de la selección con origen móvil

La fase 2 eligió cada arquitectura con un solo origen de validación (2024-10-01, enmienda 3). Para medir qué tanto
depende la elección de ese trimestre, volvimos a entrenar tres finalistas de cada modelo en los otros tres orígenes
de validación de 2024, con las semillas 2001 y 2002 y el mismo presupuesto de la fase 2 (enmienda 8). Las tres
arquitecturas se fijaron por regla antes de correr: la **elegida**, la **mejor** de la fase 2 y la **peor** de los
10 finalistas. En KAN la mejor de la fase 2 es la elegida. En Transformer la elegida es la referencia
(64, 64, 64) / (0, 1, 0), que gana por tener menos parámetros y es a la vez la peor de la fase 2. Por eso esos dos
modelos tienen dos filas. Son 150 entrenamientos y `12.2` horas de GPU, la
mitad del Transformer en el PC. La tabla da el RMSE de validación (m/s), con la media de las 2 semillas en cada origen:

```{include} tablas/e2_origenes.md
```

Lo que muestra:

- **La elección con un solo origen se sostiene en 8 de 9 modelos.** En la media de los 4 orígenes, la elegida queda
  a menos de `0.013` m/s de la mejor de las tres, y en los 7 modelos entrenados desde cero a menos de `0.006` m/s,
  lejos de δ = 0.02 m/s. En KAN y en Transformer la elegida es incluso la mejor. En el Transformer esto significa que
  la peor de la fase 2 resultó la mejor en los otros trimestres.
- **El orden entre arquitecturas cambia de un trimestre a otro.** En CNN, la peor de la fase 2 es la mejor en el
  primer trimestre de 2024, y en GRU y TCN las tres se alternan. Las diferencias son de milésimas, así que esos
  cambios son ruido y no un error de selección.
- **El trimestre pesa mucho más que la arquitectura.** Entre orígenes, el RMSE de una misma arquitectura varía entre
  `0.11` y `0.16` m/s. Entre las tres arquitecturas de un modelo, la diferencia de la media de los 4 orígenes es de
  `0.03` m/s como mucho. Es el
  mismo patrón que vemos en la prueba, donde el origen explica casi toda la varianza.
- **La excepción son los cabezales de Moirai.** Su elegida queda `0.027` m/s por encima de la mejor de las tres, que es
  la peor de la fase 2. Con un solo origen, la selección del cabezal de Moirai fue ruidosa. No cambia ninguna
  conclusión: cualquier cabezal de Moirai queda muy por detrás de Moirai sin entrenar (sección de RQ2) y de los
  modelos desde cero.
- **El equipo no cambia el resultado.** Las 9 corridas de GRU que hicimos primero en el PC difieren en milésimas de
  las mismas corridas en Colab (`resultados/e2/busqueda/GRU_origenes_pc_rtx4060.csv`), y el Transformer combina
  6 corridas del PC y 6 de Colab. Lo que sí cambia es el tiempo: una corrida del Transformer tardó de 37 a 104 minutos
  en el PC y de 3 a 5 minutos en la A100 de Colab.

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

## Validación cruzada con forward chaining (sugerencia del profesor)

Al revisar el E1, el profesor Rubio observó que las curvas de aprendizaje de entrenamiento y de validación estaban muy
separadas. Sugirió aplicar la validación cruzada para series de tiempo de la sección 2.21 del curso con una muestra
reducida, para que el cómputo cupiera. La muestra es una sola estación, la de menos datos faltantes (Aeropuerto Los
Garzones, Montería), con las últimas 12 000 h (19-ago-2024 a 31-dic-2025). Reservamos el 10 % final como holdout y
partimos el resto con *forward chaining* de la librería `tsxv`, con salto de 24 h, lo que da entre 334 y 401 pliegues
según la entrada. Probamos una rejilla de MLP (1 a 3 capas de {32, 64, 128, 256} neuronas, con dropout de 0.5
activable por capa; 584 configuraciones) con entradas de 24, 48, 96 y 192 h. Seleccionamos con el criterio del curso,
gap_ratio = |RMSE de entrenamiento final − mejor RMSE de validación| / mejor RMSE de validación < 0.15
(`src/timeseries_cv.py`, `src/tsxv_extras.py`).

| Entrada | Configuraciones con gap_ratio < 0.15 | Ganadora | gap_ratio | RMSE validación (media por pliegue) | RMSE holdout | K-Fold | Group K-Fold (meses) |
|---|---|---|---|---|---|---|---|
| 24 h | `398` de 584 | w256-32-256 | `0.074` | `0.583` | `0.528` | `0.608` | `0.630` |
| 48 h | `364` de 584 | w256-32-256 | `0.107` | `0.586` | `0.541` | `0.609` | `0.634` |
| 96 h | `312` de 584 | w256-256-128 | `0.146` | `0.590` | `0.529` | `0.607` | `0.639` |
| 192 h | **`0`** de 584 | (la de menor gap) w64-32-32 | `0.226` | `0.611` | `0.544` | `0.630` | `0.649` |

```{figure} figuras/tsxv_curvas.png
:name: fig-tsxv-curvas
Curvas de aprendizaje de la ganadora de cada entrada (promedio sobre los pliegues). La línea punteada marca la época de
mejor validación.
```

Lo que muestra:

- **La brecha crece con la longitud de la entrada.** Con 24 h pasa el filtro el 68 % de las configuraciones; con
  192 h, ninguna. Con una sola estación, cada pliegue tiene como mucho unas 400 ventanas de entrenamiento. Con 192
  entradas por ventana, una red de miles de parámetros memoriza el entrenamiento y las curvas se separan. Es el
  fenómeno que el profesor vio en el E1.
- **Más historia no ayuda en esta muestra.** La mejor entrada es la de 24 h, y el RMSE empeora al alargarla.
- **Los residuos no son ruido blanco.** La prueba BDS sobre los residuos de validación encadenados rechaza la
  independencia en todas las configuraciones (p < `0.001`). El MAPE (solo con viento > 0.5 m/s) es de cerca del 35 %.
- **El esquema de validación cambia el número.** K-Fold barajado da un RMSE parecido al de *forward chaining* agregado
  (`0.61`), pero con poca dispersión, porque cada pliegue mezcla todas las épocas del año. Group K-Fold por meses deja
  fuera bloques enteros de tiempo: da un error mayor (`0.63`–`0.65`) y una dispersión entre pliegues cuatro veces
  mayor, que es la estimación más honesta para un pronóstico futuro.

**Qué pasa con el criterio del profesor en el benchmark completo.** Aplicamos el mismo gap_ratio a los entrenamientos
de la fase 2 del E2, que usan las 62 estaciones (decenas de miles de ventanas por origen) y una entrada de 168 h. La
brecha mediana es de solo 1–6 % según el modelo, y pasan el filtro entre el 96 y el 100 % de los entrenamientos:

| Modelo | MLP | CNN | LSTM | GRU | TCN | KAN | Transformer |
|---|---|---|---|---|---|---|---|
| gap_ratio mediano | `0.009` | `0.022` | `0.022` | `0.056` | `0.014` | `0.031` | `0.013` |
| Pasan < 0.15 | 100 % | 100 % | 100 % | 96 % | 100 % | 98 % | 100 % |

Con todo el panel, las curvas de entrenamiento y validación quedan juntas incluso con 168 h de entrada. La brecha que
aparece con 192 h en la muestra reducida viene del poco número de ventanas, no de la longitud de la entrada en sí.
Esto respalda dos decisiones del E2: entrenar con todas las estaciones y evaluar con origen móvil, que es *forward
chaining* por trimestres. Para el E3, la muestra reducida sirve para explorar barato, pero las conclusiones deben
confirmarse con el panel completo.

## Diagnóstico de los tres mejores modelos

Esta sección analiza a fondo los tres primeros del ranking de Friedman (TCN, Transformer y KAN), con Ridge como
referencia cuando aplica (`src/f2a_analisis.py` y `src/f2a_gpu.py`).

### Residuos y calibración

En casi todas las estaciones los residuos de los tres modelos y de Ridge rechazan las pruebas de ruido blanco: están
autocorrelacionados (Ljung–Box a 24 rezagos, 100 % de las estaciones), son heterocedásticos (ARCH-LM, 98 %), no son
normales (Jarque–Bera, 92 %) y no son independientes (BDS, 100 %). La regresión de Mincer–Zarnowitz
{cite}`mincer1969` del valor observado sobre el pronóstico, con todas las ventanas, no rechaza que TCN (a = `−0.004`,
b = `1.006`, p = `0.22`) y Transformer (p = `0.61`) sean insesgados. Sí lo rechaza para KAN (b = `1.026`,
p < `0.001`) y Ridge (p = `0.001`). Por estación, el sesgo se rechaza en cerca de la mitad de las estaciones en los
cuatro modelos.

**Estabilidad en el tiempo.** La prueba de fluctuación de Giacomini y Rossi {cite}`giacomini2010` sobre la diferencia
de pérdida frente a Ridge rechaza la estabilidad en TCN (máximo `5.34`) y Transformer (`7.46`), con valor crítico de
`3.43`, pero no en KAN (`3.27`).

**Error fuera y dentro de la muestra.** En el último origen de prueba, el RMSE de prueba es `0.83` veces el de las
ventanas de ajuste en los tres modelos y en las tres semillas evaluadas. El error de prueba es menor porque el
trimestre octubre–diciembre es el de vientos más suaves, no porque el modelo generalice mejor de lo que ajusta; lo
que sí muestra es que no hay sobreajuste visible.

### Intervalos de predicción

Construimos intervalos conformales por estación con los residuos absolutos del último origen de validación (2024-10-01)
y los evaluamos en la prueba de 2025. La cobertura condicional se mide por terciles del valor pronosticado.

```{include} tablas/e2_intervalos.md
```

La cobertura marginal queda entre 1 y 3 puntos por debajo de la nominal, porque los residuos de validación vienen de
un solo trimestre y 2025 tiene más variabilidad. El problema está en la **cobertura condicional**: cuando el modelo
pronostica viento alto, el intervalo al 90 % cubre solo el 82 %, y cuando pronostica viento bajo cubre el 93 %. Es la
heterocedasticidad que detectó ARCH-LM: el error crece con el nivel del viento, y un intervalo de ancho fijo por
estación no la captura. La PIT tampoco es uniforme (Kolmogorov–Smirnov, p < `0.001`). Los tres modelos tienen el mismo
CRPS (`0.358`–`0.361` m/s), así que tampoco en incertidumbre hay un ganador.

### Robustez

Perturbamos la entrada de prueba del último origen de tres formas: ruido gaussiano, horas faltantes (puestas en la
media del escalado) y un quiebre de nivel de la velocidad en las últimas 72 h. Los niveles están en unidades de
desviación estándar del escalado, y cada valor es la media de 3 semillas. El índice de degradación es el área bajo la
curva del aumento relativo de RMSE frente al nivel, promediada entre las tres perturbaciones (menor es mejor).

```{include} tablas/e2_robustez.md
```

Los tres modelos toleran bien el ruido (menos del 5 % con σ = 0.5). Son sensibles a las horas faltantes y, sobre
todo, a un cambio de nivel reciente: un quiebre de 1 σ en las últimas 72 h aumenta el RMSE en un 58–68 %, porque
los tres modelos se apoyan en el nivel más reciente. **La TCN es la más robusta** en las tres perturbaciones (índice
`0.291` frente a `0.321` de KAN y `0.338` del Transformer). En la práctica, la sensibilidad a los faltantes importa
porque el panel del IDEAM tiene muchos huecos.

### Interpretabilidad

Calculamos gradientes integrados sobre 2 000 ventanas de prueba por modelo y semilla, por hora de la ventana y por
variable. Para comprobar que las atribuciones son fieles, borramos el 10 % de las horas más importantes y lo
comparamos con borrar el 10 % al azar:

| Modelo | Cambio del pronóstico al borrar las horas más importantes | Al borrar horas al azar | Estabilidad entre semillas (Spearman) | Peso de la velocidad |
|---|---|---|---|---|
| TCN | `0.55`–`0.65` | `0.08` | `0.86`–`0.91` | `53` % |
| Transformer | `0.68`–`0.74` | `0.09`–`0.11` | `0.53`–`0.68` | `57` % |
| KAN | `0.56`–`0.62` | `0.09` | `0.76`–`0.77` | `49` % |

Las atribuciones son **fieles**: borrar las horas que señalan cambia el pronóstico entre 6 y 8 veces más que borrar
horas al azar. Coinciden con lo que se sabe del viento en la región. Las horas más importantes en los tres
modelos son la más reciente (t) y las de la misma hora del día en los días anteriores (t − 24, t − 48, …, t − 144);
es decir, los modelos usan el nivel actual y el **ciclo diario** de la brisa, que es el patrón dominante en el
Caribe. La velocidad pesa cerca de la mitad, y la dirección y la temperatura el resto. Las atribuciones de la TCN son
las más estables entre semillas, y las del Transformer las menos.

## Estudio de ablación

Tomamos la TCN, primera en el ranking de Friedman, con su configuración final fija, y le quitamos un componente a la
vez. Cada variante se reentrenó en los cuatro orígenes de prueba con las 6 semillas declaradas, y se comparó con la
TCN completa con Wilcoxon por estación, Diebold–Mariano y Holm sobre las tres variantes (`src/ablacion.py`). Como se
declaró en la enmienda 7, es un análisis secundario hecho después de ver la prueba.

```{include} tablas/e2_ablacion.md
```

- **La dilatación es lo que más importa.** Sin ella, el RMSE sube `0.025` m/s, más que δ, y empeora en 60 de las 62
  estaciones (biserial `0.95`). Con dilataciones 1, 4 y 16 y núcleo de 7, el campo receptivo de la TCN es de 253 h y
  cubre la ventana de 168 h. Sin dilatación baja a 37 h. El modelo todavía ve t − 24, pero pierde t − 48, …, t − 144,
  que los gradientes integrados señalan entre las horas más importantes. Es la única diferencia de todo el E2 que supera el margen de relevancia, y no viene de elegir otra
  arquitectura sino de quitarle a una su acceso al ciclo diario.
- **Las covariables ayudan poco.** Sin dirección ni temperatura el RMSE sube `0.006` m/s: es significativo, pero
  menor que δ.
- **Las conexiones residuales no importan** en una red de tres capas: `0.001` m/s, apenas significativo.

## Reproducción independiente

**Desde las predicciones.** Recalculamos el RMSE y el MAE de las 520 corridas finales de los 13 modelos de Deep Learning
directamente desde las predicciones guardadas y los datos del panel, por un camino de código distinto al del
entrenamiento. La discrepancia máxima es de `1.3e-07` m/s, que es solo redondeo.

**Reentrenando en otro equipo.** Volvimos a entrenar desde cero en el PC del equipo (RTX 4060 Ti) cinco corridas que
se habían hecho en Colab (A100), con la misma semilla y la misma configuración:

```{include} tablas/e2_reproduccion.md
```

Las cinco paran en la misma época. TCN y DLinear reproducen el RMSE exacto, Transformer y KAN difieren en menos de
`0.0001` m/s y el MLP en `0.0009` m/s, todos muy por debajo del error estándar entre semillas (`0.0017` m/s). Las
diferencias vienen de operaciones no deterministas de la GPU, que cambian con el hardware; los resultados del E2 no
dependen del equipo en que se corrieron.
