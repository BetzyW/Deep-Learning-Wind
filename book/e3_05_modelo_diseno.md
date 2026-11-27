# Modelo original: diseño

Este capítulo presenta el modelo original del E3, **PADRE-TCN**: una regresión por diferencias con anclas diarias
causales. Seguimos el orden de la sección 7 del enunciado: la limitación que motiva el modelo, la fuente del mecanismo
y por qué se puede transferir, la hipótesis con sus predicciones, la formulación matemática, el pseudocódigo, la
demostración de dos propiedades con su verificación numérica, la complejidad y la bitácora de diseño. La
implementación, las pruebas unitarias, la ablación y la interpretabilidad van en el capítulo siguiente.

## Limitación identificada

En el E2 encontramos dos cosas que comparten todos los modelos del benchmark.

1. **Subestiman los vientos fuertes.** Con velocidad observada ≥ 6 m/s, el sesgo fue negativo en todos los modelos y en
   todas las estaciones, de `−1.15` a `−1.28` m/s (RQ3 del E2). El paper guía (DLinear) reporta la misma limitación, y
   nosotros la vimos también en la persistencia. La causa es la pérdida cuadrática: el modelo aprende la media
   condicional, que queda por debajo cuando el valor observado es extremo.
2. **Usan sobre todo el nivel actual y la misma hora de días anteriores.** Los gradientes integrados del E2 mostraron
   que la atribución se concentra en la última hora y en las horas τ − 24 j de la ventana. Los modelos ya "miran" esos
   días, pero los mezclan dentro de una representación y predicen el valor absoluto.

La idea del diseño fue predecir **cuánto cambia** el viento respecto de situaciones ya observadas (la misma hora de los
días anteriores), en vez de predecir el valor absoluto. Una diferencia entre dos casos parecidos parte de un nivel real
y no de la media, y su dispersión entre casos da una medida natural de la incertidumbre.

**Lo que no se cumplió.** En la partición de desarrollo, ninguna variante del mecanismo redujo el sesgo con viento
fuerte (tabla de la bitácora, más abajo). Por eso, **antes del tag del protocolo**, reformulamos el hallazgo alternativo
sobre la calibración de los intervalos, que sí mejoró en desarrollo. El cambio quedó declarado en
`experiments/estandar.md`. La limitación del sesgo se sigue reportando, sin pretender mejora.

## Fuente del mecanismo y justificación de la transferencia

El mecanismo es **transferido y adaptado** de tres trabajos (`experiments/procedencia.md`):

* **Regresión por diferencias entre pares (PADRE)**, de la química computacional {cite}`tynes2021`. Entrena un modelo
  sobre pares de moléculas para predecir la diferencia de la propiedad y promedia las predicciones respecto de varios
  puntos conocidos; la dispersión entre esas predicciones estima la incertidumbre.
* **Regresión con redes gemelas (TNNR)** {cite}`wetzel2022`. Una red F(xᵢ, xⱼ) aprende yᵢ − yⱼ, y el pronóstico de un
  punto nuevo es el promedio de F(x, xⱼ) + yⱼ sobre puntos de entrenamiento xⱼ. Los autores observan que la red
  cumple solo de forma aproximada la antisimetría F(xᵢ, xⱼ) = −F(xⱼ, xᵢ), y la refuerzan entrenando con los dos
  órdenes.
* **Ensamble de análogos** de la meteorología {cite}`dellemonache2013`. Pronostica a partir de las observaciones que
  siguieron a situaciones pasadas parecidas a la actual (análogos), y usa su dispersión como incertidumbre.

**Por qué se puede transferir.** La estructura que motiva PADRE y TNNR es que la diferencia entre dos casos parecidos
es más fácil de aprender y de acotar que el valor absoluto. En el viento esa estructura existe y es temporal: el ciclo
diario hace que la misma hora de días anteriores sea un análogo natural, y en el E2 los modelos ya la usaban.

**Qué tuvo que cambiar.** En química y física los pares se arman con puntos del conjunto de entrenamiento. Aquí eso
rompería la causalidad y la equidad de la comparación: los modelos del benchmark solo ven la ventana de 168 h. Las
modificaciones algorítmicas son seis:

| | Modificación | Fuente | En PADRE-TCN |
|---|---|---|---|
| M1 | Anclas | Puntos del conjunto de entrenamiento | Las ventanas que terminan a la misma hora de los K días anteriores, **dentro de la misma ventana de 168 h**: no hay fuga ni información extra |
| M2 | Codificador | Una red por par | Una sola pasada de una TCN causal da la representación de la consulta y de las K anclas; cada ancla ve solo su pasado |
| M3 | Antisimetría | Aproximada (entrenando con los dos órdenes) | **Exacta por construcción**, D(a, b) = g(a, b) − g(b, a), para cualquier valor de los parámetros |
| M4 | Fusión | Promedio uniforme | Pesos softmax por similitud en un espacio proyectado, con temperatura aprendida y un sesgo por día |
| M5 | Pérdida | Error de cada par | Error de la combinación más una pérdida por ancla. En el diseño final λ = 0: M5 quedó fuera y se evalúa en la ablación |
| M6 | Incertidumbre | Dispersión entre pares | Dispersión ponderada entre anclas, combinada con la variabilidad reciente del viento como escala de un intervalo conformal |

El análisis de novedad (búsqueda en bases de datos y verificación por un tercero) está en el capítulo de literatura.

## Hipótesis y predicciones

Las hipótesis se fijaron en el protocolo pre-registrado (`experiments/protocolo.md`, tag `e3-protocolo`), antes de
evaluar el modelo en la prueba de 2025.

**H1 (superioridad, condición R3).** PADRE-TCN tiene menor RMSE que cada uno de los 12 modelos de la familia de
contrastes, con una diferencia mayor que δ = `0.02` m/s y las seis condiciones de la sección 8.4 del enunciado.

**H2 (hallazgo alternativo).** *La dispersión ponderada entre anclas (M4 y M6) mejora el intervalo de predicción
porque mide cuánto discrepan entre sí los días análogos, que es información sobre la dificultad de cada ventana que no
contiene la variabilidad reciente del viento; por tanto, el interval score de Winkler al 90 % de PADRE-TCN es menor que
el de cada rival con el mismo método de intervalo (H2a), sin empeorar el RMSE frente a la TCN en más de δ (H2b).*

**Predicciones secundarias declaradas a priori.**

1. La dispersión anticipa el error: ρ de Spearman entre dispersión y |error| mayor que 0 (en desarrollo fue `0.48`).
2. La cobertura del intervalo al 90 % queda en 0.90 ± 0.02, en total y por trimestre.
3. Si se retira la dispersión del intervalo (normalización solo con std24), la ventaja en Winkler debe desaparecer.
   Esta predicción es la que conecta H2 con el mecanismo: si el Winkler mejorara igual sin la dispersión, la ventaja no
   vendría de las anclas.
4. El sesgo con viento fuerte no mejora (se reporta sin pretender mejora).

**Criterio de éxito y de fracaso**, fijado el 22-nov en la bitácora: H2 se acepta si las tres pruebas rechazan en los
12 contrastes del Winkler con Holm (H2a) y si el TOST concluye equivalencia con margen ±δ (H2b). Si falla cualquiera de
las dos, el diseño se declara fallido para el hallazgo alternativo.

## Formulación matemática

### Punto de partida: la regresión con redes gemelas

En TNNR {cite}`wetzel2022`, una red F aprende la diferencia entre dos puntos y el pronóstico de un punto nuevo x es el
promedio de las predicciones hechas desde n puntos conocidos (xⱼ, yⱼ):

```{math}
:label: eq-tnnr
F(x_i, x_j) \approx y_i - y_j, \qquad \hat y(x) = \frac{1}{n} \sum_{j=1}^{n} \left[ F(x, x_j) + y_j \right].
```

PADRE-TCN cambia tres cosas de la ecuación {eq}`eq-tnnr`: de dónde salen los puntos conocidos (M1 y M2), la forma de
F (M3) y el promedio (M4).

### Notación

Sea x ∈ ℝ^{L×d} la ventana de entrada escalada, con L = 168 h y d = 4 variables (velocidad, seno y coseno de la
dirección y temperatura); la fila L − 1 es la hora de emisión τ y la columna 0 es la velocidad. El objetivo es la
velocidad y = v_{τ+24} en m/s. Sea K el número de anclas (K = 6 en el diseño final) y μ_v, σ_v la media y la
desviación de la velocidad con que se escaló la entrada.

### Codificador compartido y anclas (M1, M2)

Una TCN causal de tres bloques residuales {cite}`bai2018` produce una representación por hora:

```{math}
:label: eq-codificador
h_t = \mathrm{TCN}(x)_t \in \mathbb{R}^{k}, \qquad t = 0, \dots, L-1,
```

donde h_t depende solo de x_0, …, x_t. La consulta y las anclas son

```{math}
:label: eq-anclas
q = h_{L-1}, \qquad a_j = h_{L-1-24j}, \qquad y_j^{\mathrm{a}} = \sigma_v \, x_{L-1-24(j-1),\,0} + \mu_v, \qquad j = 1, \dots, K.
```

El ancla j representa la situación de hace j días a la misma hora, y su objetivo conocido y_j^a es la velocidad
observada 24 h después de ella. Para j = 1 ese objetivo es la velocidad actual: el ancla 1 es la persistencia. Todo
sale de la misma ventana, así que el modelo no usa nada que no vean los modelos del benchmark.

### Diferencia antisimétrica (M3)

Con φ(a, b) = [a, b, a − b, a ⊙ b] ∈ ℝ^{4k} y g una red de una capa oculta sin sesgo en la salida,

```{math}
:label: eq-diferencia
D(a, b) = g\big(\varphi(a, b)\big) - g\big(\varphi(b, a)\big), \qquad g(z) = u^{\top} \mathrm{ReLU}(W z + c).
```

Cada ancla da un pronóstico, su objetivo conocido más la diferencia estimada:

```{math}
:label: eq-pronostico-ancla
\hat y_j = y_j^{\mathrm{a}} + D(q, a_j).
```

### Fusión aprendida (M4)

Con una proyección P ∈ ℝ^{p×k} (p = 8), una temperatura T = exp(θ_T) y un sesgo por día β ∈ ℝ^K:

```{math}
:label: eq-pesos
s_j = -\frac{\lVert P q - P a_j \rVert^2}{T} + \beta_j, \qquad w = \mathrm{softmax}(s), \qquad \hat y = \sum_{j=1}^{K} w_j \, \hat y_j.
```

### Dispersión e intervalo (M6)

La dispersión ponderada entre anclas es

```{math}
:label: eq-dispersion
\hat\sigma_{\mathrm{a}} = \Big( \sum_{j=1}^{K} w_j \, (\hat y_j - \hat y)^2 \Big)^{1/2}.
```

El intervalo al 90 % es conformal con el error normalizado {cite}`lei2018`. Con std24 la desviación de la velocidad en
las últimas 24 h de la ventana, la escala es

```{math}
:label: eq-escala
s = \sqrt{\hat\sigma_{\mathrm{a}}^2 + \mathrm{std}_{24}^2}, \qquad r_i = \frac{\lvert y_i - \hat y_i \rvert}{s_i + \varepsilon},
```

con ε = 0.1 veces la media de s en la calibración. Si q₀.₉ es el cuantil ⌈(n + 1)·0.9⌉/n de los r_i de calibración
(los otros tres orígenes de prueba, sección 4 del protocolo), el intervalo es ŷ ± q₀.₉ (s + ε). Los rivales usan el
mismo método con s = std24, es decir, sin la dispersión entre anclas. Los intervalos se comparan con el interval score
de Winkler {cite}`gneiting2007`, que suma el ancho y una penalización de 2/α por cada unidad fuera del intervalo.

### Pérdida y entrenamiento (M5)

```{math}
:label: eq-perdida
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \Big[ (\hat y_i - y_i)^2 + \lambda \sum_{j=1}^{K} \bar w_{ij} (\hat y_{ij} - y_i)^2 \Big],
```

donde w̄ es w sin gradiente. En el diseño final λ = 0, así que la pérdida es el error cuadrático, la misma que la de
todos los modelos del benchmark. El entrenamiento es el del E2: Adam con la tasa y el decaimiento del ajuste previo
(lr = `8.5e-4`, wd = `1e-5`, lote 1 024), como máximo 50 épocas y parada temprana con paciencia 5 sobre el RMSE del
bloque de parada; se conservan los pesos de la época con menor RMSE de parada.

### Gradientes

El gradiente de la pérdida (λ = 0) llega a los parámetros por dos caminos. Por el pronóstico de cada ancla,
∂ŷ/∂ŷ_j = w_j. Por los pesos, derivando el softmax,

```{math}
:label: eq-grad-pesos
\frac{\partial \hat y}{\partial s_k} = \sum_{j} \hat y_j \, w_j (\mathbb{1}_{jk} - w_k) = w_k \, (\hat y_k - \hat y).
```

La ecuación {eq}`eq-grad-pesos` dice que un ancla gana peso cuando su pronóstico está del lado correcto del pronóstico
combinado: si ŷ < y, el gradiente de la pérdida empuja hacia arriba los s_k de las anclas con ŷ_k > ŷ. Para la
diferencia, si θ son los parámetros de g,

```{math}
:label: eq-grad-diferencia
\frac{\partial D(a, b)}{\partial \theta} = \frac{\partial g(\varphi(a, b))}{\partial \theta} - \frac{\partial g(\varphi(b, a))}{\partial \theta}.
```

Esta ecuación explica una decisión de implementación: un sesgo en la última capa de g aparece en los dos términos y se
cancela, así que su gradiente es siempre 0. La prueba unitaria de flujo de gradientes lo detectó y la última capa quedó
sin sesgo.

## Pseudocódigo

```text
Algoritmo 1. Entrenamiento de PADRE-TCN en un origen
Entradas: ventanas de ajuste A y de parada S (índices τ), objetivos y, K, λ, lr, wd, B,
          máximo de épocas E = 50, paciencia P = 5, semilla s
Salida:   parámetros θ* con menor RMSE en S

 1. fijar la semilla s (Python, NumPy, PyTorch; cuDNN determinista)
 2. escalar las variables con la media y la desviación de las filas de entrada de A
 3. inicializar θ = (TCN, g, P, θ_T, β); mejor ← ∞; sin_mejora ← 0
 4. para época e = 1, …, E:
 5.     para cada lote de B ventanas x de A, en orden aleatorio:
 6.         h ← TCN(x)                                  # ec. (eq-codificador), una sola pasada
 7.         q ← h_{L−1};  a_j ← h_{L−1−24j};  y_j^a ← σ_v x_{L−1−24(j−1),0} + μ_v
 8.         ŷ_j ← y_j^a + g(φ(q, a_j)) − g(φ(a_j, q))   # ec. (eq-diferencia) y (eq-pronostico-ancla)
 9.         w ← softmax(−‖Pq − Pa_j‖²/exp(θ_T) + β_j);  ŷ ← Σ_j w_j ŷ_j
10.         ℒ ← media[(ŷ − y)² + λ Σ_j w̄_j (ŷ_j − y)²]   # ec. (eq-perdida)
11.         θ ← Adam(θ, ∇_θ ℒ, lr, wd)
12.     r ← RMSE de ŷ en S
13.     si r < mejor − 10⁻⁴: mejor ← r; θ* ← θ; sin_mejora ← 0
14.     si no: sin_mejora ← sin_mejora + 1; si sin_mejora ≥ P: terminar
15. devolver θ*

Algoritmo 2. Pronóstico con intervalo al 90 %
Entradas: ventana x, θ*, residuos normalizados de calibración r_1, …, r_n, ε
Salida:   ŷ y el intervalo [ŷ − q(s + ε), ŷ + q(s + ε)]

 1. calcular ŷ_j, w y ŷ con los pasos 6–9 del Algoritmo 1
 2. σ̂_a ← (Σ_j w_j (ŷ_j − ŷ)²)^{1/2};  s ← (σ̂_a² + std24(x)²)^{1/2}
 3. q ← cuantil ⌈(n + 1)·0.9⌉/n de r_1, …, r_n
 4. devolver ŷ, ŷ ± q (s + ε)
```

## Propiedades demostradas

Elegimos dos propiedades del tipo "invarianza o equivariancia" del cuadro 3 del enunciado: la antisimetría de la
diferencia ante el intercambio de sus argumentos y la acotación del pronóstico por las anclas.

```{prf:proposition} Antisimetría exacta
:label: prop-antisimetria

Para cualquier función g: ℝ^{4k} → ℝ y cualesquiera a, b ∈ ℝ^k, la diferencia de la ecuación {eq}`eq-diferencia`
cumple D(a, b) = −D(b, a) y, en particular, D(a, a) = 0.
```

*Demostración.* Por la definición,
D(b, a) = g(φ(b, a)) − g(φ(a, b)) = −[g(φ(a, b)) − g(φ(b, a))] = −D(a, b).
Con a = b queda D(a, a) = −D(a, a), así que D(a, a) = 0. La prueba no usa ninguna propiedad de g, ni de los datos,
ni del valor de los parámetros: la antisimetría vale en todo momento del entrenamiento, no solo al final. ∎

**Corolario en coma flotante.** La propiedad se mantiene **exactamente** en el computador, no solo hasta un error de
redondeo. Con la norma IEEE 754 y redondeo al más cercano, la resta es exactamente antisimétrica, fl(u − v) =
−fl(v − u), porque el redondeo es simétrico respecto de 0. Basta entonces que g devuelva el mismo número cada vez que
recibe la misma entrada, lo que se cumple en una pasada determinista.

```{prf:proposition} El pronóstico queda entre las anclas
:label: prop-convexa

Con los pesos de la ecuación {eq}`eq-pesos`, ŷ es una combinación convexa de ŷ_1, …, ŷ_K, así que
min_j ŷ_j ≤ ŷ ≤ max_j ŷ_j. Además, σ̂_a = 0 si y solo si ŷ_1 = ⋯ = ŷ_K.
```

*Demostración.* El softmax da w_j = exp(s_j) / Σ_l exp(s_l) > 0 y Σ_j w_j = 1, así que ŷ = Σ_j w_j ŷ_j es una
combinación convexa y queda en [min_j ŷ_j, max_j ŷ_j]. Para la segunda parte, σ̂_a² = Σ_j w_j (ŷ_j − ŷ)² es una suma
de términos no negativos. Si todas las ŷ_j son iguales a un valor c, entonces ŷ = c y cada término es 0. Recíprocamente,
si σ̂_a = 0, cada término w_j (ŷ_j − ŷ)² es 0, y como w_j > 0, ŷ_j = ŷ para todo j. ∎

**Supuestos y su plausibilidad.** Ninguna de las dos proposiciones supone algo sobre los datos. La segunda necesita
w_j > 0 estrictamente, lo que en coma flotante podría fallar si algún s_j fuera tan negativo que su exponencial se
anulara; en el modelo entrenado el peso mínimo observado fue `0.0139`, lejos de ese caso.

**Honestidad teórica: lo que la propiedad no garantiza.** La motivación de M3 es que la diferencia verdadera entre dos
situaciones, y(a) − y(b), es antisimétrica, así que imponerla restringe la red a la clase de funciones correcta y
debería reducir el error. Esa consecuencia **no** se sigue de la demostración: es una hipótesis empírica, y la
ablación no la apoya. Sin la antisimetría (variante V5) el RMSE cambia en `0.0003` m/s, con p ajustado máximo de
`0.28` en las tres pruebas (capítulo de implementación). La propiedad sí tiene una consecuencia observable: garantiza
la coherencia del mecanismo. Si se intercambian los papeles de la consulta y de un ancla, la corrección cambia de
signo y nunca se "corrige" una situación respecto de sí misma. La {prf:ref}`prop-convexa` también tiene una
consecuencia práctica, que es una limitación: el pronóstico no puede salir del rango de los pronósticos de las anclas.
Como cada ŷ_j suma una diferencia D sin cota, el rango no está atado al de los valores observados, pero en vientos
fuertes el modelo necesita que al menos un ancla proponga un valor alto, y en la prueba el sesgo con viento fuerte
siguió igual que el de la TCN.

### Verificación numérica

Las pruebas unitarias verifican las dos proposiciones con pesos aleatorios en float64 (antisimetría hasta `1e-12` en
1 000 pares y `gradcheck` de la diferencia; `tests/test_original_model.py`). Además las verificamos sobre el modelo
V6 **entrenado** (semilla 1000, origen 2025-10-01) y 20 000 ventanas reales de prueba, con las representaciones
reales de la consulta y de las anclas (`src/verificar_propiedades.py`):

```{include} tablas/e3_verificacion.md
```

La antisimetría da exactamente `0` en float32 y en float64, como predice el corolario, aunque la diferencia llega a
`2.46` m/s en valor absoluto. Ningún pronóstico sale del rango de sus anclas: en el peor caso queda `0.017` m/s por
dentro. Los pesos suman 1 hasta la precisión de la máquina, y la dispersión es exactamente 0 cuando se fuerzan todas
las ŷ_j iguales. En los datos reales nunca es 0 (mínimo `0.0155` m/s): las anclas nunca coinciden del todo.

## Complejidad

Sean L la longitud de la ventana, d el número de variables, c₁, c₂, c₃ los anchos de los tres bloques de la TCN
(c₀ = d), k_c = 7 el tamaño del núcleo, K las anclas, m el ancho oculto de g y p la dimensión de la proyección.

* **Codificador.** Cada bloque tiene dos convoluciones causales y una conexión residual, con costo
  O(L k_c (c_{i−1} c_i + c_i²)) por ventana. La dilatación no cambia el costo, solo el campo receptivo.
* **Diferencias.** 2K evaluaciones de g sobre vectores de dimensión 4c₃: O(K m c₃).
* **Pesos.** O(K p c₃) para las proyecciones y O(K) para el softmax.

En total, la inferencia cuesta

```{math}
:label: eq-complejidad
O\Big( L \, k_c \sum_{i=1}^{3} (c_{i-1} c_i + c_i^2) \; + \; K c_3 (m + p) \Big) \text{ por ventana},
```

y el entrenamiento, del orden de tres veces eso por ventana y época (la pasada hacia atrás cuesta unas dos veces la
hacia adelante). La memoria de activaciones es O(L Σ c_i) por ventana. El término de las anclas no depende de L y es
pequeño: con los valores de V6 (c = (64, 64, 16), m = 32, p = 8, K = 6) es menos del 0.1 % del total. Por eso PADRE-TCN
cuesta prácticamente lo mismo que la TCN del E2: `33.0` frente a `32.9` MFLOPs por ventana y `100 663` frente a
`98 433` parámetros. Frente a TNNR, que evalúa la red una vez por par contra n puntos de entrenamiento (costo n veces
el de la red), M2 reduce el costo a una sola pasada del codificador.

La predicción es que, con los anchos fijos, los FLOPs y el tiempo crecen linealmente con L y los parámetros no
cambian; y que, con L fija y un ancho c en todos los bloques, los FLOPs y los parámetros crecen como c². Lo medimos en
la CPU del PC (4 hilos, lote 1 024; `src/complejidad_padre.py`):

```{include} tablas/e3_complejidad.md
```

Las pendientes en escala log-log confirman el análisis: `1.00` para los FLOPs y el tiempo en función de L, y `1.96`
para los FLOPs en función del ancho. El tiempo crece más despacio que los FLOPs al aumentar el ancho (pendiente
`1.35`): con anchos pequeños domina el costo fijo de cada operación y del acceso a memoria, no la aritmética.

## Bitácora de diseño

El diseño se tomó en una **partición de desarrollo** (los orígenes de validación 2024-01, 2024-04 y 2024-07) que no se
reutiliza en la confirmación. El origen 2024-10 se reservó para el ajuste de hiperparámetros, igual que en los
benchmarks, y ni la prueba de 2025 ni los conjuntos adicionales intervinieron en ninguna decisión. La base fija fue la
TCN elegida en el E2 con sus hiperparámetros y la semilla 2001. La bitácora completa, con fechas, está en
`experiments/design_log.md`.

```{include} tablas/e3_diseno.md
```

*Medias de los tres orígenes de desarrollo (semilla 2001). Viento fuerte: velocidad observada ≥ 6 m/s, unas 213
ventanas por origen. La correlación no aplica a V0 ni a V3, que no tienen dispersión entre anclas.*

**Lectura y decisiones.**

* **Ronda 1 (20 y 21 de noviembre).** Ninguna variante mejoró el RMSE de la TCN; todas quedaron a menos de `0.009`
  m/s, del orden de la variación entre semillas (DE ≈ `0.004` m/s en el E2). Era lo que anticipaba el techo de
  desempeño medido en el E2. Como la hipótesis del mecanismo no era sobre el RMSE medio, repetimos las mismas 6
  variantes guardando las predicciones (ronda 2). Las repeticiones no suman variantes, y sus 21 RMSE salieron idénticos
  a los de la ronda 1: el entrenamiento es reproducible.
* **Ronda 2 (21 y 22 de noviembre).** El sesgo con viento fuerte siguió en unos −2 m/s en todas las variantes, así que
  la hipótesis original no se cumplió. La dispersión entre anclas sí traía información sobre el error (ρ ≈ `0.45`).
  Con intervalos conformales calibrados entre orígenes, la escala combinada √(dispersión² + std24²) de V6 dio un Winkler
  de `3.317` frente a `3.460` de la TCN con std24, un `4.1` % menos, con un RMSE `0.009` m/s mayor, dentro de δ.
* **Congelamiento (22 de noviembre).** Se fijó V6 con el intervalo de escala combinada, el criterio de éxito (Winkler
  al 90 % sin empeorar el RMSE en más de δ) y la ablación de cinco componentes. Riesgo declarado: la ventaja de V6
  sobre V1 salió de una sola semilla.

**Número de variantes (N_d = 9):** 6 de arquitectura más 3 formas de normalizar el intervalo (ancho fijo, solo la
dispersión y la combinada; std24 es la del rival). Con 9 variantes evaluadas en una semilla, la mejor puede parecer
mejor por azar; por eso la confirmación usa otros datos (2025), 10 semillas y la familia completa de contrastes.

**Esfuerzo de diseño frente a los benchmarks.** El diseño de PADRE-TCN costó 9 variantes y unas 40 corridas de
desarrollo; los benchmarks no tuvieron diseño propio porque son arquitecturas publicadas, pero recibieron en el E2 una
búsqueda de arquitectura de hasta 1 000 configuraciones por modelo. El **ajuste de hiperparámetros sí fue el mismo**:
PADRE-TCN probó exactamente las mismas 20 configuraciones de (lr, wd, lote), generadas con la semilla 7, en el mismo
origen 2024-10 y con el mismo presupuesto que los demás modelos (`src/ajuste_padre.py`).
