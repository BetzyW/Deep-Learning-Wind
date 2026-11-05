# Estado del arte y modelos base

El objetivo de esta revisión es doble: saber **qué arquitecturas de Deep Learning se usan para pronosticar la
velocidad del viento** y **qué tan bien comparadas están**, para elegir con evidencia los tres modelos base del
benchmark. Se sigue la guía PRISMA 2020 {cite}`page2021`.

## Protocolo de búsqueda

| Elemento | Decisión |
|---|---|
| Pregunta | ¿Qué arquitecturas de Deep Learning se comparan para pronosticar la velocidad del viento a partir de series observadas, y cuál rinde mejor? |
| Fuentes | **Scopus** y **Web of Science Core Collection** (acceso institucional de la Universidad del Norte), **Semantic Scholar** (API de búsqueda masiva) y **arXiv** (API oficial) para preprints |
| Fecha de la búsqueda | 3 de noviembre de 2026 (Semantic Scholar y arXiv) y 4 de noviembre de 2026 (Scopus y Web of Science) |
| Periodo | 2018–2026 |
| Idioma | Inglés |
| Reproducibilidad | En `literatura/` están `s2_fetch.py`, `arxiv_fetch.py`, `tamizaje.py` (unión, duplicados y tamizaje), `decisiones_elegibilidad.csv` (decisión manual por registro) y `registros_tamizados.csv` (los 395 registros evaluados). Las exportaciones de Scopus (CSV) y Web of Science (texto delimitado, registro completo) no se publican porque sus términos no permiten redistribuir los resúmenes; se regeneran con las consultas de abajo y se guardan como `scopus.csv` y `wos.txt` |

**Cadena de búsqueda** (sintaxis de Semantic Scholar, donde `+` es AND y `|` es OR):

```text
"wind speed" + (forecast* | predict*) + ("deep learning" | lstm | "long short-term memory" | gru
  | "convolutional neural network" | cnn | "multilayer perceptron" | transformer)
```

Equivalentes en las otras fuentes:

```text
Scopus:          TITLE("wind speed" AND (forecast* OR predict*)) AND TITLE-ABS-KEY("deep learning" OR lstm
                 OR "long short-term memory" OR gru OR "convolutional neural network" OR cnn
                 OR "multilayer perceptron" OR mlp OR transformer) AND PUBYEAR > 2017 AND PUBYEAR < 2027
Web of Science:  TI=("wind speed" AND (forecast* OR predict*)) AND TS=("deep learning" OR lstm OR ...
                 OR transformer) AND PY=(2018-2026)
arXiv:           ti:"wind speed" AND (ti:forecast* OR ti:predict*) AND (abs:"deep learning" OR abs:LSTM ...)
```

En Scopus, Web of Science y arXiv la condición sobre el viento se aplicó al título, que es el mismo filtro del
primer tamizaje; en Semantic Scholar la búsqueda abarcó título y resumen y el filtro de título se aplicó después.

## Criterios de inclusión y exclusión

| Etapa | Criterio | Cómo se aplicó |
|---|---|---|
| Tamizaje 1 (título) | El título trata de **pronosticar la velocidad del viento** (contiene *wind speed* y *forecast/predict*); se excluye el viento solar | Automático (`tamizaje.py`) |
| Tamizaje 2 (resumen) | Hay resumen disponible y en él aparecen **al menos dos familias** de redes: MLP, recurrentes (LSTM/GRU/RNN), convolucionales (CNN/TCN) o Transformer | Automático, con expresiones regulares |
| Elegibilidad | El estudio **compara cuantitativamente al menos dos familias como modelos independientes** (no solo como piezas de un híbrido) sobre datos de viento atmosférico | Lectura manual del resumen; decisión registrada en `decisiones_elegibilidad.csv` |
| Síntesis | Los **20 estudios elegibles más citados** (máximo de citas entre las fuentes); el conteo de frecuencias usa los 65 elegibles | Orden por número de citas |

## Diagrama PRISMA

```{mermaid}
flowchart TD
    A["Identificados: 5 070<br/>Scopus: 1 186 · Web of Science: 734<br/>Semantic Scholar: 3 111 · arXiv: 39"] --> B["Duplicados eliminados: 1 570<br/>(mismo DOI o mismo título)"]
    B --> C["Registros únicos: 3 500"]
    C --> D["Excluidos por título o periodo: 2 205<br/>(no tratan de pronóstico de velocidad del viento)"]
    C --> E["Tamizados por resumen: 1 295"]
    E --> F["Sin resumen disponible: 31<br/>Menos de dos familias de redes: 869"]
    E --> G["Evaluados para elegibilidad: 395"]
    G --> H["Excluidos: 330<br/>(solo comparan híbridos, una sola familia<br/>como modelo independiente, viento no atmosférico<br/>o revisiones)"]
    G --> I["Elegibles: 65"]
    I --> J["Incluidos en la síntesis: 20<br/>(los más citados)"]
```

## Frecuencia de las familias en los estudios elegibles

| Familia | Estudios (de 65) | Proporción |
|---|---|---|
| Recurrentes (LSTM, GRU, RNN) | 60 | 92 % |
| Convolucionales (CNN 1D, TCN) | 39 | 60 % |
| Perceptrón multicapa (MLP, ANN, BPNN) | 30 | 46 % |
| Transformer y variantes | 13 | 20 % |

El conteo es automático sobre título y resumen, así que una familia cuenta aunque solo aparezca como comparador. Los
Transformers son minoría y aparecen sobre todo desde 2022.

## Tabla de síntesis

| Estudio | Datos | Horizonte | Familias comparadas | Mejor modelo |
|---|---|---|---|---|
| {cite}`bentsen2022` | Estaciones de la plataforma continental noruega | 10 min a 4 h | GNN con Transformer, LSTM, MLP | GNN con Transformers modificados |
| {cite}`joseph2023` | Estación objetivo y estaciones vecinas | Casi tiempo real | BiLSTM híbrido, LSTM, RNN, MLP, RF | BiLSTM híbrido |
| {cite}`zhu2018` | Arreglo de aerogeneradores (varios sitios) | Corto plazo | CNN + MLP (PDCNN), MLP, SVR, árbol | PDCNN |
| {cite}`duan2022` | Tres estaciones de un parque eólico, cuatro meses | Corto plazo | CNN, RNN y otras tres redes como predictores individuales; combinación con regresión lineal | La combinación supera a cada red individual |
| {cite}`liu2020` | Registros de un parque eólico | 10 min | SDAE-LSTM, MLP, LSTM, SAE | SDAE-LSTM |
| {cite}`yaghoubirad2023` | Zabol (Irán), multivariado | 6 meses a 5 años | LSTM, GRU, CNN, CNN-LSTM | GRU |
| {cite}`fukuoka2018` | Serie de una estación | Corto plazo | LSTM, CNN 1D, red densa | LSTM y CNN 1D superan a la densa |
| {cite}`jin2025` | Serie de velocidad, cuatro rangos de tiempo | Ultracorto, multipaso | Informer con doble descomposición, GRU, ViT, CNN-BiLSTM con atención | Informer con descomposición (−23 % a −33 % de RMSE) |
| {cite}`shivam2020` | Seis series reales | 50 pasos | CNN dilatada causal, LSTM, GRU | CNN residual con atención |
| {cite}`saxena2021` | LiDAR y mástil costa afuera (India), 80 a 180 m | Un paso | CNN, LSTM apilada, BiLSTM, CNN-LSTM, MLP, ConvLSTM 2D | Depende del sitio |
| {cite}`li2023` | Serie de velocidad | Corto plazo | ARIMA, GM, LR, RF, SVR, ANN, LSTM, CNN | CNN |
| {cite}`sarp2022` | Datos reales a gran escala | Multipaso | MLP, RNN, SVM con censura de datos | Censura reduce costo sin perder precisión |
| {cite}`ibrahim2020` | Parque eólico | Corto plazo | ANN, CNN, LSTM, ConvLSTM, SVM | SVM (más preciso); ConvLSTM elegido por costo |
| {cite}`qin2019` | Tres sitios de distinto terreno | 4 h, multipaso directo | EEMD-LSTM, BPNN, SVM, LSTM | EEMD-LSTM |
| {cite}`biswas2021` | Bahía de Bengala y mar Arábigo, diario 2006–2018 | Diario | LSTM, BiLSTM, red densa poco profunda | BiLSTM |
| {cite}`qiao2022` | Varios sitios en valles de terreno complejo | 1 paso y multipaso | Aprendizaje colaborativo multisitio, CNN, LSTM, CNN-LSTM | Multisitio (−19 % de RMSE frente a CNN) |
| {cite}`jaseena2020` | Serie de velocidad | Corto plazo | SAE-LSTM, SVR, ANN, RNN, LSTM | SAE-LSTM |
| {cite}`hittawe2024` | Diez puntos del mar Rojo | Corto plazo | Transformer apilado, LSTM, BiLSTM, GRU, BiGRU | Transformer apilado |
| {cite}`ullah2025` | Cuatro series de Alemania y China | Corto plazo | GRU y Transformer con doble descomposición, cinco modelos individuales | Híbrido GRU-Transformer |
| {cite}`namboodiri2025` | Serie de velocidad, con transferencia a otros sitios | Corto plazo | LSTM, BiLSTM, CNN, CNN-BiLSTM con ondículas | CNN-BiLSTM en ensamble |

Los datos de la tabla se extrajeron de los resúmenes; cuando un dato no aparece allí se describe de forma general
(por ejemplo, «corto plazo»).

## Fichas de los tres modelos base

Con $d = 4$ variables de entrada y una ventana de $L = 168$ horas (decisiones del EDA), la entrada de cada muestra es
$\mathbf X \in \mathbb R^{L \times d}$ y la salida es la velocidad $\hat v_{t+24}$.

### MLP (perceptrón multicapa)

Aplana la ventana en un vector de $Ld = 672$ entradas y aplica capas densas {cite}`rumelhart1986`:

$$\mathbf h_1 = \sigma(\mathbf W_1 \operatorname{vec}(\mathbf X) + \mathbf b_1), \quad
\mathbf h_2 = \sigma(\mathbf W_2 \mathbf h_1 + \mathbf b_2), \quad \hat y = \mathbf w_3^\top \mathbf h_2 + b_3$$

- **Parámetros:** $(Ld+1)h_1 + (h_1+1)h_2 + (h_2+1)$. Con la configuración elegida en el ajuste ($h_1 = h_2 = 128$):
  **102 785**.
- **Sesgo inductivo:** ninguno temporal; trata cada hora de la ventana como una entrada distinta.
- **Por qué se incluye:** es un comparador frecuente (46 % de los estudios) y el punto de partida para medir si la
  estructura temporal aporta algo. En {cite}`sari2020` llegó a superar a CNN y LSTM solas.

### LSTM (red recurrente de memoria larga y corta)

Recorre la ventana hora por hora y mantiene un estado de memoria controlado por compuertas {cite}`hochreiter1997`:

$$\begin{aligned}
\mathbf f_t &= \sigma(\mathbf W_f[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_f), \quad
\mathbf i_t = \sigma(\mathbf W_i[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_i), \quad
\mathbf o_t = \sigma(\mathbf W_o[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_o) \\
\mathbf c_t &= \mathbf f_t \odot \mathbf c_{t-1} + \mathbf i_t \odot \tanh(\mathbf W_c[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_c),
\quad \mathbf h_t = \mathbf o_t \odot \tanh(\mathbf c_t)
\end{aligned}$$

- **Parámetros:** $4h(d_{\text{in}}+h) + 8h$ por capa recurrente (PyTorch usa dos vectores de sesgo), con
  $d_{\text{in}} = d$ en la primera capa y $d_{\text{in}} = h$ en las siguientes, más $h+1$ en la salida. Con la
  configuración elegida (dos capas, $h = 32$): $4\,864 + 8\,448 + 33 =$ **13 345**.
- **Sesgo inductivo:** procesamiento secuencial con memoria; adecuado para dependencias de largo alcance.
- **Por qué se incluye:** la familia recurrente aparece en el 92 % de los estudios y la LSTM es la más citada.

### CNN 1D (red convolucional temporal)

Desliza filtros de ancho $k$ sobre el eje del tiempo {cite}`lecun1998`:

$$z_{t,c'} = \sigma\Big(b_{c'} + \sum_{c=1}^{C_{\text{in}}} \sum_{j=0}^{k-1} w_{j,c,c'}\, x_{t+j,c}\Big)$$

Se apilan tres capas convolucionales con agrupamiento entre ellas, de modo que el campo receptivo supera las 24 h del
ciclo diario, y se termina con un promedio global sobre el tiempo y una capa lineal.

- **Parámetros:** $k \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ por capa convolucional. Con la
  configuración elegida ($4 \to 32 \to 64 \to 64$, $k = 7$) y la salida lineal: **44 129**.
- **Sesgo inductivo:** patrones locales invariantes a la posición; con agrupamiento o dilatación amplía el campo
  receptivo.
- **Por qué se incluye:** segunda familia más frecuente (60 %) y la ganadora en varios estudios recientes
  {cite}`li2023,mugware2024,makubyane2024`.

### Comparación de complejidad

| Modelo | Parámetros (configuración elegida) | Costo por muestra | Paralelizable en el tiempo |
|---|---|---|---|
| MLP | 102 785 | $O(Ld\,h_1)$ | Sí |
| LSTM | 13 345 | $O(L\,h(d+h))$, secuencial | No |
| CNN 1D | 44 129 | $O(L\,k\,C_{\text{in}} C_{\text{out}})$ | Sí |

Las configuraciones salen del ajuste de hiperparámetros, con el mismo presupuesto (20 configuraciones) para las tres;
el detalle está en el capítulo de implementación.

## Análisis crítico y vacío

1. **No hay una arquitectura ganadora.** Entre los modelos independientes, la CNN gana en
   {cite}`li2023,mugware2024,makubyane2024`, la GRU o la RNN en {cite}`yaghoubirad2023,zaman2021,mohandes2021` y la
   red densa en {cite}`sari2020`. {cite}`saxena2021` lo dice de forma explícita: con seis redes y dos sitios costa
   afuera, la mejor arquitectura cambia de un sitio a otro. El resultado depende del sitio, del horizonte y de la
   preparación de los datos.
2. **Casi todos proponen un híbrido** (descomposición más red) y lo comparan contra modelos base sin ajustar con el
   mismo esfuerzo, lo que favorece al modelo propuesto.
3. **Casi nunca se reporta la persistencia.** Ninguno de los 20 resúmenes menciona el pronóstico ingenuo como
   referencia, a pesar de que en horizontes cortos es difícil de superar.
4. **Sin inferencia estadística.** Ningún resumen menciona varias semillas ni pruebas de diferencia de precisión como
   Diebold–Mariano; algunos reportan $R^2$ cercanos a 1 {cite}`hittawe2024`, lo que sugiere evaluación a un paso o
   fuga de información.
5. **Validación poco descrita.** Los resúmenes casi nunca describen la partición; los que lo hacen usan un único
   corte, sin periodo de separación entre entrenamiento y prueba.
6. **Poca cobertura tropical.** No se encontraron estudios con estaciones del Caribe colombiano.

**Vacío que aborda este trabajo:** una comparación **justa** de MLP, LSTM y CNN 1D (mismo presupuesto de ajuste,
varias semillas), con **persistencia como baseline**, **partición cronológica con embargo** y **pruebas
estadísticas** de la diferencia, sobre una red de estaciones del Caribe colombiano.

Los Transformers no se incluyen como modelo base porque aparecen solo en el 20 % de los estudios y requieren más
datos; quedan para el Entregable 2. La revisión de {cite}`majidigharehnaz2024` ofrece un panorama más amplio de las
aplicaciones de Deep Learning al pronóstico del viento.

## Limitaciones de la revisión

- La elegibilidad se decidió leyendo los resúmenes, no el texto completo; 31 registros sin resumen quedaron fuera.
- Las exclusiones de elegibilidad se registraron con la decisión y un motivo general, no con un motivo específico por
  artículo.
- Las búsquedas no usan exactamente los mismos campos en todas las fuentes (en Semantic Scholar, título y resumen; en
  las demás, el título); el tamizaje por título iguala el criterio después de unirlas.
