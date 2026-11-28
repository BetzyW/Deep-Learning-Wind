# Evaluación ampliada y robustez

El capítulo de validación respondió si PADRE-TCN supera a la familia en el panel del IDEAM. Este capítulo amplía la
mirada: cómo le va en tres conjuntos de datos de otra naturaleza (condición 4 de R3), y cómo se comporta frente a los
benchmarks en los estudios del cuadro 5 del enunciado: sensibilidad a los hiperparámetros, estabilidad entre semillas,
perturbaciones, cambio de distribución, calibración e incertidumbre, eficiencia y análisis de errores. Las cifras salen
de `src/adicionales_e3.py` (`resultados/e3/adicionales/analisis.json`) y `src/evaluacion_ampliada.py`
(`resultados/e3/ampliada.json`), y las tablas se generan con `src/tablas_e3.py`.

## Conjuntos de datos adicionales

Los tres conjuntos se eligieron el 21 de noviembre, antes de entrenar ningún modelo en ellos y antes del tag del
protocolo, con su fuente, licencia, versión y huella SHA-256 (`experiments/conjuntos_adicionales.md`). Cada uno difiere
del panel del IDEAM en al menos dos ejes:

| Conjunto | Tarea a 24 h | En qué difiere del panel del IDEAM |
|---|---|---|
| ETTh1 {cite}`zhou2021` | Temperatura del aceite de un transformador | Otro dominio (eléctrico), una sola serie, no estacionaria, sin estructura espacial |
| SDWPF (KDD Cup 2022) | Potencia activa de 134 turbinas (kW) | Otra variable (potencia, acotada por la curva de la turbina), un solo parque muy denso, 245 días |
| NOAA ISD-Lite, Kansas | Velocidad del viento (m/s) | Clima continental y llano, otra red, otro país |

**Protocolo.** El mismo del panel, adaptado a cada conjunto: ventana de 168 h, horizonte de 24 h, partición
cronológica con embargo, escalado con el entrenamiento y 10 semillas. Por cómputo, y declarado antes de entrenar, se
usan 2 orígenes de prueba por conjunto, 10 de las 34 estaciones de Kansas (elegidas al azar con semilla 11) y la
configuración elegida en el E2 para cada modelo, sin reajustar ninguno (tampoco PADRE-TCN).

**Desviaciones declaradas.**
* **Informer y STGCN con 3 semillas** en SDWPF y NOAA, por su costo en Colab (unos 50 y 43 minutos por corrida). El
  de Informer se decidió el 23-nov, antes del tag; el de STGCN, el 26-nov, después del tag y antes de ver sus
  resultados. Sus medias se reportan con el número de corridas que tienen.
* **Chronos y Moirai no se evaluaron en estos conjuntos.** Su versión sin entrenar necesita una sesión de Colab con
  bibliotecas propias y no alcanzó el tiempo. Es una excepción a la regla de evaluar a toda la familia; la condición 4
  se evalúa con los otros 10 rivales. No cambia el veredicto de R3, que ya falla en el panel principal.
* **DLinear corregido** para desescalar su salida, como en el artículo original (23-nov, antes del tag).

### Resultados por conjunto

```{include} tablas/e3_adicionales_resumen.md
```

*RMSE medio de los 2 orígenes de prueba, en las unidades de cada conjunto. Los contrastes usan t pareado y Wilcoxon
sobre las corridas pareadas por origen y semilla, con Holm sobre los 10 rivales de cada conjunto; un contraste es
significativo si los dos p ajustados son menores que 0.05.*

```{include} tablas/e3_adicionales_contrastes.md
```

**Lectura.**

* **ETTh1: PADRE-TCN queda 4.º de 11.** Supera con significancia a 5 rivales (la TCN, KAN, STGCN, Informer y gradient
  boosting), por un 17 % a un 39 %; frente al Transformer es 19 % mejor, pero solo una de las dos pruebas rechaza. La razón es que la serie no es estacionaria (su nivel cae de `15.9` en el entrenamiento a `2.3`
  en la prueba) y PADRE-TCN se ancla a los valores recientes de la ventana: su error es igual al de la persistencia
  (`2.063` frente a `2.062`). Los modelos que predicen el valor absoluto arrastran el nivel del entrenamiento. Solo
  Ridge, que trabaja sobre rezagos, lo supera.
* **SDWPF: PADRE-TCN queda 10.º de 11.** En potencia de turbinas, GRU, Informer, KAN y Ridge son entre un 7 % y un 14 %
  mejores, con significancia. La potencia está acotada por la curva de la turbina, y la corrección aditiva sobre los días
  anteriores es un mal supuesto cuando la turbina está saturada o detenida.
* **NOAA Kansas: PADRE-TCN queda 8.º de 11.** Es el caso más parecido al panel principal (viento horario) y el
  resultado también se parece: todos los modelos quedan a menos de 5 % entre sí, y la TCN, GRU, Ridge y gradient
  boosting son entre un 1 % y un 2.4 % mejores que PADRE-TCN.

### Prueba entre conjuntos

```{include} tablas/e3_friedman.md
```

*Rangos de los 11 modelos (PADRE-TCN, los 8 del Top 10 con corridas, Ridge y gradient boosting) en los 4 conjuntos:
el panel del IDEAM y los tres adicionales. No se promedian métricas de escalas distintas: cada conjunto aporta solo
rangos.*

La prueba de Friedman con la corrección de Iman–Davenport no rechaza que todos los modelos tengan el mismo rango medio
(p = `0.73`). Con 4 conjuntos y 11 modelos la potencia es baja, como advierte el enunciado: para que dos modelos fueran
distintos según Nemenyi harían falta `7.55` rangos de diferencia, y la mayor diferencia observada es de `4.75`. Por eso
la conclusión se apoya también en la consistencia por conjunto: **ningún modelo gana en todos**. Ridge tiene el mejor
rango medio y gana en dos conjuntos, pero queda 3.º en el panel del IDEAM; el Transformer es el mejor en el panel y
6.º o peor en los otros tres.

### Condición 4 de R3

La condición exige que PADRE-TCN sea mejor que todos los rivales en al menos 2 de los 3 conjuntos y que no sea
significativamente peor que ninguno en ninguno. **No se cumple**: no es el mejor en ningún conjunto, y es
significativamente peor que Ridge en ETTh1 y que 4 rivales en SDWPF y en NOAA. **Si se retirara el conjunto más
desfavorable (SDWPF), la conclusión no cambiaría**, porque NOAA y ETTh1 ya la incumplen.

## Sensibilidad a los hiperparámetros

Todos los modelos probaron las mismas 20 configuraciones de (tasa de aprendizaje, decaimiento, lote) en el ajuste
previo, en el origen de validación 2024-10. El rango del RMSE de validación entre esas 20 configuraciones mide cuánto
depende cada modelo de acertar con los hiperparámetros.

```{include} tablas/e3_sensibilidad.md
```

**PADRE-TCN es el modelo menos sensible a los hiperparámetros**: su RMSE varía solo `0.006` m/s entre las 20
configuraciones, menos que la TCN (`0.007`), unas seis veces menos que el Transformer (`0.035`) y muy lejos de
Informer y DLinear, que con la peor configuración casi duplican y casi triplican su error. Todas las configuraciones de PADRE-TCN quedan
a menos de δ de la mejor, así que **conserva su desempeño en toda la región explorada**. Una explicación plausible es
que el ancla 1 (la persistencia) le da un punto de partida razonable desde el inicio del entrenamiento.

## Estabilidad entre semillas

```{figure} figuras/e3_semillas.png
:name: fig-semillas
:width: 100%

RMSE de prueba de cada una de las 10 semillas (media de los 4 orígenes de 2025) para los modelos entrenados de la
familia y PADRE-TCN. Cada punto es una semilla; la caja va del primer al tercer cuartil y la línea es la mediana. La
banda sombreada es ±δ alrededor de la media de PADRE-TCN. Chronos y Moirai se omiten porque su escala (0.90 y 1.04 m/s)
aplastaría el resto.
```

```{include} tablas/e3_estabilidad.md
```

La {numref}`fig-semillas` muestra la conclusión del capítulo de validación de otra forma: los 9 modelos entrenados
caben dentro de la banda de ±δ alrededor de PADRE-TCN. PADRE-TCN es de los modelos más estables entre semillas, con un
coeficiente de variación de `0.14` %; solo DLinear y PatchTST, con muchos menos grados de libertad, varían menos. La
prueba de Friedman entre semillas, con los orígenes como bloques, no rechaza en ningún modelo: ninguna semilla es
sistemáticamente mejor que otra, así que la variación entre semillas es ruido de la optimización y no un efecto de la
inicialización.

## Perturbaciones controladas

Aplicamos a la entrada las mismas tres perturbaciones del E2, con tres severidades cada una, sobre todas las ventanas
de prueba del origen 2025-10-01 y con 3 semillas por modelo.

```{include} tablas/e3_perturbaciones.md
```

*RMSE medio de las 3 semillas y aumento relativo frente a la entrada sin perturbar. Los valores del Transformer y la TCN
son los del E2 (`resultados/e2/f2a/gpu_resultados.json`); los de PADRE-TCN, de `src/interpretabilidad_padre.py`.*

PADRE-TCN se degrada como los benchmarks: ante horas faltantes y quiebres de nivel queda en general entre la TCN y
el Transformer (algo peor que ambos solo con el quiebre más leve). Ante ruido fuerte (σ = 0.5) es el que más se degrada (`+8` %
frente a `+4` y `+5` %): el ruido afecta directamente a los objetivos de las anclas, que el modelo toma tal cual de la
ventana. Un quiebre de nivel de 2 desviaciones triplica el error en los tres modelos; ninguno es robusto a ese tipo de
cambio.

## Cambio de distribución

Hay dos evidencias de cambio de distribución fuera de toda decisión de diseño: los conjuntos adicionales (otro
dominio, otro clima) y los cuatro trimestres de 2025, que el diseño nunca vio.

```{include} tablas/e3_trimestres.md
```

El error de todos los modelos cambia mucho entre trimestres (de `0.81` a `0.61` m/s en PADRE-TCN), porque el régimen
de vientos cambia con la época del año. La diferencia entre PADRE-TCN y el Transformer se mantiene entre
`0.003` y `0.010` m/s en los cuatro (la mayor, en el primer trimestre): no hay un trimestre en el que el modelo se degrade de forma particular. **La
cobertura del intervalo al 90 % se mantiene entre `0.881` y `0.913`** aunque el régimen cambie, porque el intervalo
conformal se calibra con los otros trimestres y se escala con la dispersión de cada ventana. En los conjuntos
adicionales, en cambio, el modelo no se adapta: el supuesto de corregir aditivamente sobre los días anteriores
funciona en el viento, pero no en la potencia saturada de SDWPF.

## Calibración e incertidumbre

Los resultados principales están en el capítulo de validación (H2): con el mismo método de intervalo para todos,
PADRE-TCN tiene el menor *interval score* de Winkler de la familia (`2.991` frente a `3.153` del mejor rival), una regla
de puntuación propia para intervalos {cite}`gneiting2007`, con cobertura `0.898` e intervalos más estrechos. Además:

* **Cobertura por subgrupo.** Por trimestre, entre `0.881` y `0.913` (tabla anterior). Con viento fuerte (≥ 6 m/s), la
  cobertura baja a `0.74`: el intervalo es simétrico alrededor de un pronóstico que subestima esos casos.
* **Calibración del pronóstico puntual.** Mincer–Zarnowitz no rechaza a = 0 y b = 1 en el agregado (p = `0.92`), mejor
  que la TCN (p = `0.17`).
* **Limitación de la predicción conformal.** El método supone intercambiabilidad, que los datos temporales no
  cumplen del todo. Lo mitigamos calibrando con orígenes distintos del evaluado y verificando la cobertura por
  trimestre; la cobertura condicional con viento fuerte muestra que la garantía es marginal, no condicional.
* No calculamos el CRPS de PADRE-TCN: el modelo da un intervalo, no una distribución completa.

## Eficiencia y frontera de Pareto

El costo de cada modelo (parámetros, FLOPs y latencia en la misma CPU) está en la tabla de equidad del capítulo de
validación. Con esos valores y el desempeño, buscamos los modelos **no dominados**: aquellos para los que ningún otro
es a la vez más barato y mejor.

* **Latencia en CPU frente al Winkler:** la frontera la forman **DLinear, KAN y PADRE-TCN**. PADRE-TCN es el único con
  el mejor Winkler, y para mejorarlo no hace falta pagar más que con la TCN o el Transformer: su latencia es de `175` µs
  por ventana con lote 1 024, frente a `189` de la TCN y `770` del Transformer.
* **FLOPs frente al RMSE:** la frontera la forman el Transformer, la TCN, KAN, DLinear y PatchTST. PADRE-TCN queda
  apenas fuera: tiene el costo de la TCN (`33.0` frente a `32.9` MFLOPs) con un RMSE `0.003` m/s mayor.

Chronos y Moirai se excluyen de la frontera porque solo se mide su cabezal; su modelo base congelado cuesta órdenes de
magnitud más.

## Análisis de errores

**Por régimen de viento.** El pronóstico medio de las 10 semillas, por rango de la velocidad observada:

```{include} tablas/e3_regimenes.md
```

Los tres modelos neuronales se comportan casi igual en todos los regímenes: sobreestiman levemente el viento flojo
(`+0.19` m/s por debajo de 2 m/s) y subestiman cada vez más el fuerte (`−1.21` m/s por encima de 6 m/s). **Con viento
fuerte, la persistencia tiene menos error que los tres modelos** (`1.629` frente a `1.64`–`1.67` m/s) y mucho menos
sesgo (`−0.78`). Es la limitación del paper guía que el diseño intentó corregir y no corrigió: el error cuadrático
empuja el pronóstico hacia la media, y la combinación convexa de las anclas no lo saca de ahí (Proposición 2).

**Errores compartidos.**

```{include} tablas/e3_errores.md
```

Los residuos de PADRE-TCN están correlacionados en `0.99` con los de la TCN y el Transformer, y el 90 % de sus peores
ventanas son también las peores de esos modelos. Los modelos fallan en los mismos casos: los cambios bruscos de
régimen que ninguna variable de entrada anticipa. Es la misma evidencia del techo de desempeño del E2 vista desde los
errores: con estas entradas, los modelos extraen la misma información y comparten los mismos puntos ciegos. La
persistencia comparte menos (correlación `0.81`), porque falla de otra manera.

**Lo que no se estudió.** La curva de aprendizaje al variar el tamaño de la muestra (10 %, 25 %, 50 % y 100 %) no se
corrió para PADRE-TCN; la sensibilidad a la muestra de los benchmarks está en el E2. El tiempo y los FLOPs al variar el
tamaño de la entrada están en el capítulo de diseño.
