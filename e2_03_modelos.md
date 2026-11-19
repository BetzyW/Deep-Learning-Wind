# Los diez modelos del Top 10 (Entregable 2)

Este capítulo describe cada modelo del Top 10 tal como lo usamos: de dónde viene, cómo fluye la información, qué
calcula, qué aportó frente a lo anterior, qué problema resolvía en su artículo y cuánto cuesta. Todos reciben la misma
entrada: una ventana de `L = 168` horas con `d = 4` variables por hora (velocidad, seno y coseno de la dirección y
temperatura), escaladas con el entrenamiento de cada origen, y todos producen un solo número: la velocidad del viento
24 horas después del final de la ventana (pronóstico directo).

Los parámetros y las operaciones por ventana corresponden a la configuración final de cada modelo en nuestro problema
(arquitectura ganadora de la búsqueda o arquitectura publicada). Las operaciones se midieron con el contador de
operaciones de PyTorch sobre una ventana; en la LSTM se calcularon a mano, porque el contador no ve la celda
recurrente optimizada.

| Modelo | Familia | Configuración final | Parámetros | MFLOPs por ventana | Costo en la longitud L |
|---|---|---|---|---|---|
| Transformer | Atención | w = (64, 64, 64), dropout (0, 1, 0) | `100 801` | `40.3` | O(L² · w) |
| KAN | MLP profundo | w = (16, 32, 128), dropout (0, 1, 0) | `138 545` | `0.28` | O(L · d · w₁ · G) |
| TCN | Convolucional | w = (64, 64, 16), dropout (1, 1, 0) | `98 433` | `32.9` | O(L · k · w²) |
| STGCN | Grafos | Publicada (2 bloques ST, 8 vecinas) | `1 358 881` | `167.4` | O(L · K · \|E\| · c) |
| Chronos | Preentrenado | T5-small congelado + cabezal (32, 16, 256) | `21 553` (cabezal) | `0.04` (cabezal) | O(L²) en el modelo base |
| GRU | Recurrente | w = (64, 64, 64), dropout (0, 1, 0) | `63 425` | `20.9` | O(L · w²), secuencial |
| Informer | Atención | Publicada (D = 512, 2 + 1 capas) | `11 312 641` | `2 519.8` | O(L log L · D) |
| DLinear | MLP profundo | Publicada | `338` | ≈ 0 | O(L) |
| Moirai | Preentrenado | Moirai-1.0-R small congelado + cabezal (32, 128, 32) | `20 705` (cabezal) | `0.04` (cabezal) | O((L/P)²) en el modelo base |
| PatchTST | Atención | Publicada (parches de 16, paso 8) | `404 995` | `8.35` | O((L/P)² · D) |

Los modelos base del E1 (MLP, CNN 1D y LSTM) entran al benchmark con la misma búsqueda de arquitectura; se describieron
en el capítulo 4 y aquí solo se resumen en la tabla de resultados.

## Transformer

**Referencia.** Vaswani et al. (2017), *Attention is all you need*, NeurIPS {cite}`vaswani2017`.

```{mermaid}
flowchart LR
  X["Ventana 168 × 4"] --> E["Proyección lineal + codificación posicional"]
  E --> B1["Bloque 1: atención multicabeza + red densa"]
  B1 --> B2["Bloque 2 (dropout)"] --> B3["Bloque 3"]
  B3 --> U["Última hora"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Cada bloque aplica atención de producto escalado,
$\mathrm{Atención}(Q,K,V)=\mathrm{softmax}\!\left(QK^\top/\sqrt{d_k}\right)V$, con $Q = HW_Q$, $K = HW_K$, $V = HW_V$,
seguida de una red densa por posición, ambas con conexión residual y normalización previa. Usamos solo el codificador,
con `w_i / 16` cabezas, y la salida se lee en la última hora.

**Innovación.** Sustituye la recurrencia por atención: cada hora de la ventana se relaciona directamente con todas las
demás, sin pasar la información paso a paso.

**Problema original.** Traducción automática (WMT 2014 inglés–alemán e inglés–francés), con BLEU como métrica, en una
sola corrida por configuración y sin pruebas de significancia.

**Fortalezas y limitaciones.** Captura dependencias lejanas y se paraleliza bien; su costo crece con el cuadrado de la
ventana y necesita muchos datos. En nuestro problema la arquitectura ganadora fue la de referencia (64, 64, 64).

## KAN

**Referencia.** Liu et al. (2024), *KAN: Kolmogorov–Arnold Networks* {cite}`liu2024kan`.

```{mermaid}
flowchart LR
  X["Ventana 168 × 4 aplanada"] --> K1["Capa KAN 1: una función B-spline por arista"]
  K1 --> K2["Capa KAN 2 (dropout)"] --> K3["Capa KAN 3"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Inspirada en el teorema de representación de Kolmogorov–Arnold: cada arista aprende una función
univariada $\phi_{ij}(x) = w_b\,\mathrm{silu}(x) + w_s \sum_k c_k B_k(x)$, combinación de B-splines de orden 3 sobre una
rejilla de 5 intervalos, y cada nodo suma sus entradas: $h_j = \sum_i \phi_{ij}(x_i)$.

**Innovación.** Mueve la no linealidad de los nodos a las aristas y la hace aprendible, lo que permite inspeccionar la
forma de cada función.

**Problema original.** Ajuste de funciones simbólicas, solución de ecuaciones diferenciales parciales y problemas
científicos (teoría de nudos, localización de Anderson), con RMSE; no evalúa series de tiempo.

**Fortalezas y limitaciones.** Interpretable y eficiente en parámetros en problemas suaves; cada arista cuesta varias
evaluaciones de spline y su rendimiento en datos ruidosos es poco conocido.

## TCN

**Referencia.** Bai, Kolter y Koltun (2018), *An empirical evaluation of generic convolutional and recurrent networks
for sequence modeling* {cite}`bai2018`.

```{mermaid}
flowchart LR
  X["Ventana 168 × 4"] --> C1["Bloque 1: 2 conv. causales (k = 7, dilatación 1) + residual"]
  C1 --> C2["Bloque 2: dilatación 4 (dropout)"] --> C3["Bloque 3: dilatación 16"]
  C3 --> U["Última hora"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Convolución causal dilatada: $(x *_d f)(t) = \sum_{i=0}^{k-1} f(i)\,x_{t - d\,i}$. Con dilataciones
1, 4 y 16 y núcleo 7, el campo receptivo es $1 + 2(k-1)(1+4+16) = 253$ horas, mayor que la ventana. Cada bloque
suma su entrada (conexión residual).

**Innovación.** Muestra que una convolución causal con dilatación exponencial y residuales iguala o supera a las
recurrentes en tareas de secuencias, con memoria efectiva larga y entrenamiento paralelo.

**Problema original.** Tareas sintéticas y de lenguaje (problema de suma, MNIST secuencial, memoria de copia, música
polifónica, Penn Treebank, WikiText-103), con exactitud, pérdida y perplejidad.

**Fortalezas y limitaciones.** Estable, paralela y con memoria controlada por diseño; el campo receptivo es fijo y no
se adapta al contenido.

## STGCN

**Referencia.** Yu, Yin y Zhu (2018), *Spatio-temporal graph convolutional networks*, IJCAI {cite}`yu2018`.

```{mermaid}
flowchart LR
  X["Ventanas de la estación y sus 8 vecinas"] --> T1["Conv. temporal con compuerta"]
  T1 --> G1["Conv. de grafo Chebyshev (K = 3)"] --> T2["Conv. temporal"]
  T2 --> S2["2.º bloque espacio-temporal"] --> O["Capa de salida"] --> Y["ŷ de la estación"]
```

**Formulación.** La convolución de grafo aproxima un filtro espectral con polinomios de Chebyshev del laplaciano
reescalado: $g_\theta * x = \sum_{k=0}^{K-1} \theta_k T_k(\tilde L)\,x$, con $\tilde L = 2L/\lambda_{\max} - I$. El grafo
de cada estación son ella y sus 8 vecinas más cercanas, con pesos $w_{ij} = \exp(-d_{ij}^2/\sigma^2)$.

**Innovación.** Combina convoluciones temporales y de grafo en bloques, sin recurrencia, para pronosticar en redes de
sensores.

**Problema original.** Velocidad del tráfico en autopistas (PeMSD7 y BJER4), con MAE, MAPE y RMSE.

**Fortalezas y limitaciones.** Es el único modelo que usa información de estaciones vecinas; supone una red de sensores
completa y sincronizada, que el panel del IDEAM no es (adaptaciones en la enmienda 6).

## Chronos

**Referencia.** Ansari et al. (2024), *Chronos: learning the language of time series* {cite}`ansari2024`.

```{mermaid}
flowchart LR
  X["Velocidad (168 h)"] --> Q["Escalado por la media y cuantización en 4 096 fichas"]
  Q --> T5["Codificador T5-small preentrenado (congelado)"]
  T5 --> P["Promedio sobre las horas"] --> H["Cabezal denso de 3 capas"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Trata la serie como lenguaje: la escala por su media absoluta, la cuantiza en fichas y entrena un T5
con entropía cruzada. Nosotros congelamos el modelo base (`amazon/chronos-t5-small`, 46 millones de parámetros),
calculamos una vez la representación de cada ventana y solo entrenamos un cabezal de 3 capas densas con la misma
búsqueda de arquitectura.

**Innovación.** Un modelo de lenguaje sin cambios de arquitectura, preentrenado con muchas series reales y sintéticas,
que pronostica sin entrenar en el conjunto nuevo.

**Problema original.** Pronóstico probabilístico en 42 conjuntos de datos, dentro y fuera del entrenamiento, con WQL y
MASE agregados.

**Fortalezas y limitaciones.** Sirve para muchas series sin ajuste; es univariado (no ve dirección ni temperatura) y el
promedio de la representación que usamos pierde el orden de las horas.

## GRU

**Referencia.** Cho et al. (2014), *Learning phrase representations using RNN encoder–decoder for statistical machine
translation*, EMNLP {cite}`cho2014`.

```{mermaid}
flowchart LR
  X["Ventana 168 × 4"] --> R1["GRU capa 1"] --> R2["GRU capa 2 (dropout)"] --> R3["GRU capa 3"]
  R3 --> U["Estado final"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Con compuertas de actualización $z_t = \sigma(W_z x_t + U_z h_{t-1})$ y de reinicio
$r_t = \sigma(W_r x_t + U_r h_{t-1})$, el estado es $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh\!\left(W x_t + U(r_t \odot h_{t-1})\right)$.

**Innovación.** Simplifica la LSTM a dos compuertas y sin celda de memoria aparte, con desempeño parecido y menos
parámetros.

**Problema original.** Traducción automática estadística (WMT'14 inglés–francés), con BLEU.

**Fortalezas y limitaciones.** Pocos parámetros y buena memoria de corto plazo; procesa la ventana hora por hora, sin
paralelizar en el tiempo.

## Informer

**Referencia.** Zhou et al. (2021), *Informer: beyond efficient Transformer for long sequence time-series
forecasting*, AAAI {cite}`zhou2021`.

```{mermaid}
flowchart LR
  X["Ventana 168 × 4"] --> E["Codificador: atención ProbSparse + destilación (×2)"]
  D0["Últimas 48 h + 24 posiciones en cero"] --> Dd["Decodificador generativo"]
  E --> Dd --> Y["ŷ (t + 24)"]
```

**Formulación.** La atención ProbSparse solo calcula de forma completa las $u = c \ln L$ consultas más informativas,
elegidas por la medida $\bar M(q_i, K) = \max_j \frac{q_i k_j^\top}{\sqrt d} - \frac{1}{L}\sum_j \frac{q_i k_j^\top}{\sqrt d}$;
la destilación reduce a la mitad la longitud entre capas.

**Innovación.** Baja el costo de la atención de O(L²) a O(L log L) y genera todo el horizonte en una pasada.

**Problema original.** Pronóstico de secuencias largas (ETTh1, ETTh2, ETTm1, Weather, ECL), con MSE y MAE.

**Fortalezas y limitaciones.** Pensado para ventanas y horizontes largos; con D = 512 es el modelo más grande del
benchmark (11 millones de parámetros) y necesitó procesar los lotes por trozos para caber en memoria.

## DLinear (paper guía)

**Referencia.** Zeng et al. (2023), *Are Transformers effective for time series forecasting?*, AAAI
{cite}`zeng2023`.

```{mermaid}
flowchart LR
  X["Velocidad (168 h)"] --> MA["Media móvil k = 25: tendencia"]
  X --> R["Resto = serie − tendencia"]
  MA --> L1["Lineal 168 → 1"]
  R --> L2["Lineal 168 → 1"]
  L1 --> S(("+")) --> Y["ŷ (t + 24)"]
  L2 --> S
```

**Formulación.** $\hat y = W_t\,\mathrm{MA}_{25}(x) + W_r\,(x - \mathrm{MA}_{25}(x))$, dos capas lineales de 168 a 1.

**Innovación.** Muestra que un modelo lineal con descomposición supera a varios Transformers en pronóstico de largo
plazo, lo que cuestiona su efectividad en series de tiempo.

**Problema original.** Nueve conjuntos de pronóstico de largo plazo (ETT, Weather, Traffic, Electricity, Exchange,
ILI), con MSE y MAE. Lo replicamos en ETTh1 (capítulo de revisión).

**Fortalezas y limitaciones.** Muy barato e interpretable; univariado y lineal, no capta interacciones.

## Moirai

**Referencia.** Woo et al. (2024), *Unified training of universal time series forecasting Transformers*, ICML
{cite}`woo2024`.

```{mermaid}
flowchart LR
  X["Velocidad (168 h)"] --> P["Parches de 32 h + escalado"]
  P --> T["Transformer preentrenado Moirai-1.0-R small (congelado)"]
  T --> A["Promedio sobre los parches"] --> H["Cabezal denso de 3 capas"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Un Transformer de solo codificador sobre parches, con atención que distingue variables y una mezcla
de distribuciones como salida, preentrenado con el conjunto LOTSA. Igual que con Chronos, congelamos el modelo base
(`Salesforce/moirai-1.0-R-small`, 14 millones de parámetros) y solo entrenamos el cabezal.

**Innovación.** Un único modelo para cualquier frecuencia y número de variables, con parches de varios tamaños.

**Problema original.** Pronóstico probabilístico y de largo plazo en conjuntos fuera del entrenamiento (benchmark de
Monash y otros), con CRPS, MSIS, MSE y MAE.

**Fortalezas y limitaciones.** Universal y sin reentrenar; en nuestro uso, el promedio sobre los parches pierde el
último nivel observado, lo que explica en buena parte su mal desempeño (capítulo de discusión).

## PatchTST

**Referencia.** Nie et al. (2023), *A time series is worth 64 words: long-term forecasting with Transformers*, ICLR
{cite}`nie2023`.

```{mermaid}
flowchart LR
  X["Velocidad (168 h)"] --> N["Normalización por instancia"] --> P["Parches de 16 h, paso 8"]
  P --> T["Codificador Transformer"] --> F["Aplanar"] --> L["Lineal"] --> D["Desnormalizar"] --> Y["ŷ (t + 24)"]
```

**Formulación.** Divide la serie en parches que hacen de "palabras", los proyecta y aplica un codificador Transformer;
cada variable se procesa por separado (independencia de canales).

**Innovación.** Los parches reducen la longitud de la secuencia y conservan la información local, y la independencia de
canales evita el sobreajuste entre variables.

**Problema original.** Pronóstico de largo plazo (Weather, Traffic, Electricity, ILI, ETT), con MSE y MAE.

**Fortalezas y limitaciones.** Más barato que un Transformer por hora; al ser univariado no usa la dirección ni la
temperatura.

## Qué reportan los artículos

Según lo que revisamos, ninguno de los diez artículos compara modelos con pruebas de significancia; la mayoría reporta
una sola corrida o el promedio de pocas sin intervalos. Esa es una de las razones del diseño de este benchmark: varias
semillas, la estación como unidad de independencia y tres pruebas por contraste.
