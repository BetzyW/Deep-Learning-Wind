# Estado del arte y modelos base

El objetivo de esta revisión es indagar **qué arquitecturas de Deep Learning se usan para pronosticar la
velocidad del viento** y **qué tan bien se comparan entre sí**, para elegir, con evidencia, los tres modelos base del
benchmark. Se sigue la guía PRISMA 2020 {cite}`page2021`.

## Protocolo de búsqueda

| Elemento | Decisión |
|---|---|
| Pregunta | ¿Qué arquitecturas de Deep Learning se comparan para pronosticar la velocidad del viento a partir de series observadas, y cuál rinde mejor? |
| Fuentes | **Scopus**, **Web of Science Core Collection** e **IEEE Xplore** (acceso de la Universidad del Norte y cuenta IEEE), **Semantic Scholar** (API de búsqueda masiva) y **arXiv** (API oficial) para preprints |
| Fecha de la búsqueda | 3 de noviembre de 2026 (Semantic Scholar y arXiv) y 4 de noviembre de 2026 (Scopus, Web of Science e IEEE Xplore) |
| Periodo | 2018–2026 |
| Idioma | Inglés |
| Reproducibilidad | En `literatura/` están `s2_fetch.py`, `arxiv_fetch.py`, `tamizaje.py` (unión, duplicados y tamizaje), `decisiones_elegibilidad.csv` (decisión manual por registro) y `registros_tamizados.csv` (los 399 registros evaluados). Las exportaciones de Scopus (CSV), Web of Science (texto delimitado, registro completo) e IEEE Xplore (CSV) no se publican porque sus términos no permiten redistribuir los resúmenes; se regeneran con las consultas de abajo y se guardan como `scopus.csv`, `wos.txt` e `ieee.csv` |

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
IEEE Xplore:     ("Document Title":"wind speed") AND ("Document Title":forecast* OR "Document Title":predict*)
                 AND ("All Metadata":"deep learning" OR ... OR "All Metadata":transformer), años 2018–2026
arXiv:           ti:"wind speed" AND (ti:forecast* OR ti:predict*) AND (abs:"deep learning" OR abs:LSTM ...)
```

En Scopus, Web of Science, IEEE Xplore y arXiv la condición sobre el viento se aplicó al título, que es el mismo filtro del
primer tamizaje; en Semantic Scholar la búsqueda abarcó título y resumen y el filtro de título se aplicó después.

## Criterios de inclusión y exclusión

| Etapa | Criterio | Cómo se aplicó |
|---|---|---|
| Tamizaje 1 (título) | El título trata de **pronosticar la velocidad del viento** (contiene *wind speed* y *forecast/predict*); se excluye el viento solar | Automático (`tamizaje.py`) |
| Tamizaje 2 (resumen) | Hay resumen disponible y en él aparecen **al menos dos familias** de redes: MLP, recurrentes (LSTM/GRU/RNN), convolucionales (CNN/TCN) o Transformer | Automático, con expresiones regulares |
| Elegibilidad | El estudio **compara cuantitativamente al menos dos familias como modelos independientes** (no solo como piezas de un híbrido) sobre datos de viento atmosférico | Lectura manual del resumen; decisión registrada en `decisiones_elegibilidad.csv` |
| Síntesis | Los **20 estudios elegibles más citados** (máximo de citas entre las fuentes); el conteo de frecuencias usa los 66 elegibles | Orden por número de citas |

## Diagrama PRISMA

```{mermaid}
flowchart TD
    A["Identificados: 5 441<br/>Scopus: 1 186 · Web of Science: 734 · IEEE Xplore: 371<br/>Semantic Scholar: 3 111 · arXiv: 39"] --> B["Duplicados eliminados: 1 864<br/>(mismo DOI o mismo título)"]
    B --> C["Registros únicos: 3 577"]
    C --> D["Excluidos por título o periodo: 2 206<br/>(no tratan de pronóstico de velocidad del viento)"]
    C --> E["Tamizados por resumen: 1 371"]
    E --> F["Sin resumen disponible: 31<br/>Menos de dos familias de redes: 941"]
    E --> G["Evaluados para elegibilidad: 399"]
    G --> H["Excluidos: 333<br/>(solo comparan híbridos, una sola familia<br/>como modelo independiente, viento no atmosférico<br/>o revisiones)"]
    G --> I["Elegibles: 66"]
    I --> J["Incluidos en la síntesis: 20<br/>(los más citados)"]
```

## Frecuencia de las familias en los estudios elegibles

| Familia | Estudios (de 66) | Proporción |
|---|---|---|
| Recurrentes (LSTM, GRU, RNN) | 60 | 91 % |
| Convolucionales (CNN 1D, TCN) | 40 | 61 % |
| Perceptrón multicapa (MLP, ANN, BPNN) | 31 | 47 % |
| Transformer y variantes | 13 | 20 % |

El conteo es automático sobre título y resumen, así que una familia cuenta aunque solo aparezca como comparador. Los
Transformers son minoría y aparecen sobre todo desde 2022.

## Proceso de selección

El tamizaje automático (título y resumen) lo ejecuta `tamizaje.py`. La elegibilidad la decidió **un único revisor**
leyendo cada resumen, con apoyo de Claude (ver «Uso de inteligencia artificial» en la portada); cada decisión quedó
registrada en `decisiones_elegibilidad.csv`. Al no haber un segundo revisor independiente no hubo desacuerdos que
resolver, lo que se declara como limitación: los casos dudosos (por ejemplo, estudios que comparan modelos
individuales solo dentro de un ensamble) se resolvieron con el criterio escrito de elegibilidad.

## Tabla de síntesis

Todos los estudios abordan el pronóstico de la velocidad del viento. Las métricas y valores se toman del resumen de
cada artículo; «no en el resumen» indica que el resumen no da cifras.

| Estudio | Datos y horizonte | Modelos de Deep Learning de referencia | Mejor modelo | Métricas y valores reportados |
|---|---|---|---|---|
| {cite}`bentsen2022` | Estaciones de la plataforma continental noruega; 10 min a 4 h | LSTM, MLP (espacio-temporales) | GNN con FFTransformer y Autoformer | MAE y MSE; no en el resumen |
| {cite}`joseph2023` | Estación objetivo y estaciones vecinas; casi tiempo real | LSTM, RNN, MLP | BiLSTM híbrido | RRMSE 9.6–23.8 %, MAPE 8.8–21.5 %; 77–85 % de errores ≤ 0.5 m/s |
| {cite}`zhu2018` | Arreglo de aerogeneradores; corto plazo | MLP | CNN + MLP (PDCNN) | Tres índices de error propios; no en el resumen |
| {cite}`duan2022` | Tres estaciones de un parque eólico, cuatro meses; corto plazo | CNN, RNN y otras tres redes individuales | Combinación con regresión lineal | No en el resumen |
| {cite}`liu2020` | Registros de un parque eólico; 10 min | MLP, LSTM, SAE | SDAE-LSTM | No en el resumen |
| {cite}`yaghoubirad2023` | Zabol (Irán), multivariado; 6 meses a 5 años | LSTM, GRU, CNN, CNN-LSTM | GRU | No en el resumen |
| {cite}`fukuoka2018` | Una estación; corto plazo | Red densa, LSTM, CNN 1D | LSTM y CNN 1D | Error y retardo; no en el resumen |
| {cite}`jin2025` | Una serie, cuatro rangos de tiempo; ultracorto, multipaso | GRU, ViT, CNN-BiLSTM | Informer con doble descomposición | RMSE −33 % frente a GRU y −25 % frente a ViT |
| {cite}`shivam2020` | Seis series reales; 50 pasos | LSTM, GRU | CNN dilatada residual con atención | Varias métricas; no en el resumen |
| {cite}`saxena2021` | LiDAR y mástil costa afuera (India), 80–180 m; un paso | CNN, LSTM, BiLSTM, CNN-LSTM, MLP, ConvLSTM | Depende del sitio | No en el resumen |
| {cite}`li2023` | Una serie; corto plazo | ANN, LSTM, CNN | CNN | MSE, RMSE, MAE, MAPE, $R^2$; no en el resumen |
| {cite}`sarp2022` | Datos reales a gran escala; multipaso | MLP, RNN | (estudio de costo de entrenamiento) | No en el resumen |
| {cite}`ibrahim2020` | Parque eólico; corto plazo | ANN, CNN, LSTM, ConvLSTM | SVM (más preciso); ConvLSTM por costo | No en el resumen |
| {cite}`qin2019` | Tres sitios de distinto terreno; 4 h directo | BPNN, LSTM | EEMD-LSTM | No en el resumen |
| {cite}`biswas2021` | Bahía de Bengala y mar Arábigo, diario 2006–2018; diario | Red densa, LSTM, BiLSTM | BiLSTM | MSE y RMSE; no en el resumen |
| {cite}`qiao2022` | Varios sitios en valles; 1 paso y multipaso | CNN, LSTM, CNN-LSTM | Aprendizaje multisitio | RMSE −19 %, −13 % y −8 % frente a CNN, LSTM y CNN-LSTM |
| {cite}`jaseena2020` | Una serie; corto plazo | ANN, RNN, LSTM | SAE-LSTM | MAE, RMSE, $R^2$; no en el resumen |
| {cite}`hittawe2024` | Diez puntos del mar Rojo; corto plazo | LSTM, BiLSTM, GRU, BiGRU, Transformer | Transformer apilado | $R^2$ = 0.9996 |
| {cite}`ullah2025` | Cuatro series de Alemania y China; corto plazo | GRU, Transformer (individuales) | Híbrido GRU-Transformer | $R^2$ de 0.956 a 0.996 |
| {cite}`namboodiri2025` | Una serie con transferencia a otros sitios; corto plazo | LSTM, BiLSTM, CNN | CNN-BiLSTM en ensamble | RMSE 0.155–0.194 m/s, $R^2$ 0.988–0.992 |

## Calidad de los estudios incluidos

La valoración se hace con lo que informa el resumen, así que «no se describe» no significa que el estudio no lo haya
hecho, sino que no se puede verificar sin el texto completo.

| Estudio | Partición descrita | Riesgo aparente de fuga | Repeticiones (semillas) | Pruebas estadísticas |
|---|---|---|---|---|
| {cite}`bentsen2022` | No se describe | No se aprecia | No se describe | Dice «significativamente», sin nombrar la prueba |
| {cite}`joseph2023` | No se describe | Selección de variables en tres etapas: fuga si usó todo el periodo | No se describe | No |
| {cite}`zhu2018` | No se describe | No se aprecia | No se describe | No |
| {cite}`duan2022` | No se describe | Descomposición EMD: fuga si se aplicó a toda la serie | No se describe | No |
| {cite}`liu2020` | No se describe | Selección por información mutua: fuga si usó todo el periodo | No se describe | No |
| {cite}`yaghoubirad2023` | No se describe | No se aprecia | No se describe | No |
| {cite}`fukuoka2018` | No se describe | No se aprecia | No se describe | No |
| {cite}`jin2025` | No se describe | Doble descomposición VMD + CEEMDAN | No se describe | No |
| {cite}`shivam2020` | Validación con ventana deslizante | Bajo | No se describe | No |
| {cite}`saxena2021` | No se describe | Descomposición EEMD | No se describe | No |
| {cite}`li2023` | No se describe | No se aprecia | No se describe | No |
| {cite}`sarp2022` | No se describe | No se aprecia | No se describe | No |
| {cite}`ibrahim2020` | No se describe | No se aprecia | No se describe | No |
| {cite}`qin2019` | No se describe | Descomposición EEMD | No se describe | No |
| {cite}`biswas2021` | Validación independiente (año 2018) | Bajo | Varias configuraciones, no semillas | No |
| {cite}`qiao2022` | Por estaciones del año | No se aprecia | No se describe | No |
| {cite}`jaseena2020` | No se describe | No se aprecia | No se describe | No |
| {cite}`hittawe2024` | No se describe | Alto: $R^2 \approx 1$ | No se describe | No |
| {cite}`ullah2025` | No se describe | Doble descomposición; $R^2$ hasta 0.996 | No se describe | No |
| {cite}`namboodiri2025` | No se describe | Descomposición con ondículas; $R^2 \approx 0.99$ | No se describe | No |

**Balance:** solo 3 de 20 describen la partición; **ninguno** menciona varias semillas ni una prueba estadística
explícita; 6 aplican una descomposición de la señal (EMD, EEMD, VMD u ondículas), que produce fuga si se calcula
sobre la serie completa antes de partirla, y 3 reportan $R^2 \geq 0.99$, poco creíble para viento a horizontes de
horas sin información futura.

## Fichas de los tres modelos base

Con $d = 4$ variables de entrada y una ventana de $L = 168$ horas (decisiones del EDA), la entrada de cada muestra es
$\mathbf X \in \mathbb R^{L \times d}$ y la salida es la velocidad $\hat v_{t+24}$. Los tres se entrenan igual: pérdida
MSE, optimizador Adam, lotes de 1024 ventanas y parada temprana con la validación de 2024.

**Configuración en la literatura.** Los resúmenes casi nunca informan profundidad, ancho, regularización ni
configuración de entrenamiento de los modelos base; por eso, como pide el enunciado, la arquitectura no se fijó por
criterio propio sino con una búsqueda con el mismo presupuesto para los tres modelos (capítulo de implementación).

```{mermaid}
flowchart LR
    X["Ventana 168 h × 4 variables"] --> M1["MLP: aplanar → densa 128 → densa 128 → salida"]
    X --> M2["LSTM: 2 capas de 32 recorren las 168 h → último estado → salida"]
    X --> M3["CNN 1D: 3 convoluciones (k = 7) con agrupamiento → promedio global → salida"]
    M1 --> Y["Velocidad en t + 24 h"]
    M2 --> Y
    M3 --> Y
```

### MLP (perceptrón multicapa)

Aplana la ventana en un vector de $Ld = 672$ entradas y aplica capas densas {cite}`rumelhart1986`:

$$\mathbf h_1 = \sigma(\mathbf W_1 \operatorname{vec}(\mathbf X) + \mathbf b_1), \quad
\mathbf h_2 = \sigma(\mathbf W_2 \mathbf h_1 + \mathbf b_2), \quad \hat y = \mathbf w_3^\top \mathbf h_2 + b_3$$

- **Parámetros:** $(Ld+1)h_1 + (h_1+1)h_2 + (h_2+1)$. Con la configuración elegida en el ajuste ($h_1 = h_2 = 128$):
  **102 785**.
- **Sesgo inductivo:** ninguno temporal; trata cada hora de la ventana como una entrada distinta.
- **Por qué se incluye:** es un comparador frecuente (47 % de los estudios) y el punto de partida para medir si la
  estructura temporal aporta algo. En {cite}`sari2020` llegó a superar a CNN y LSTM solas.
- **En la literatura:** comparador en {cite}`joseph2023,zhu2018,liu2020,saxena2021,biswas2021`, con datos de
  estaciones, parques eólicos y LiDAR costa afuera; casi siempre queda por debajo del modelo propuesto, pero los
  resúmenes no dan su error.
- **Fortalezas:** el más simple y rápido de entrenar (todo en paralelo); fácil de reproducir.
- **Limitaciones:** el número de parámetros crece con la longitud de la ventana; no comparte pesos entre horas, así
  que puede sobreajustar con ventanas largas.

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
- **Por qué se incluye:** la familia recurrente aparece en el 91 % de los estudios y la LSTM es la más citada.
- **En la literatura:** la familia recurrente (LSTM o GRU) es referencia en
  {cite}`liu2020,yaghoubirad2023,jin2025,shivam2020,qiao2022,hittawe2024`; por
  ejemplo, el modelo multisitio de {cite}`qiao2022` reduce el RMSE de una LSTM en 13 %.
- **Fortalezas:** pocos parámetros, independientes de la longitud de la ventana; modela dependencias largas.
- **Limitaciones:** entrenamiento secuencial y lento; sensible a la tasa de aprendizaje; puede olvidar información
  muy lejana pese a las compuertas.

### CNN 1D (red convolucional temporal)

Desliza filtros de ancho $k$ sobre el eje del tiempo {cite}`lecun1998`:

$$z_{t,c'} = \sigma\Big(b_{c'} + \sum_{c=1}^{C_{\text{in}}} \sum_{j=0}^{k-1} w_{j,c,c'}\, x_{t+j,c}\Big)$$

Se apilan tres capas convolucionales con agrupamiento entre ellas, de modo que el campo receptivo supera las 24 h del
ciclo diario, y se termina con un promedio global sobre el tiempo y una capa lineal.

- **Parámetros:** $k \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ por capa convolucional. Con la
  configuración elegida ($4 \to 32 \to 64 \to 64$, $k = 7$) y la salida lineal: **44 129**.
- **Sesgo inductivo:** patrones locales invariantes a la posición; con agrupamiento o dilatación amplía el campo
  receptivo.
- **Por qué se incluye:** segunda familia más frecuente (61 %) y la ganadora en varios estudios recientes
  {cite}`li2023,mugware2024,makubyane2024`.
- **En la literatura:** referencia en {cite}`fukuoka2018,saxena2021,qiao2022,namboodiri2025`; el modelo multisitio
  de {cite}`qiao2022` reduce el RMSE de una CNN en 19 %.
- **Fortalezas:** paralelizable; comparte pesos a lo largo del tiempo, así que generaliza bien con pocos parámetros
  por capa.
- **Limitaciones:** el campo receptivo es limitado si no se apilan capas o se usa dilatación; no tiene memoria
  explícita del orden más allá de ese campo.

### Comparación de complejidad

| Modelo | Parámetros (configuración elegida) | Costo por muestra | Paralelizable en el tiempo |
|---|---|---|---|
| MLP | 102 785 | $O(Ld\,h_1)$ | Sí |
| LSTM | 13 345 | $O(L\,h(d+h))$, secuencial | No |
| CNN 1D | 44 129 | $O(L\,k\,C_{\text{in}} C_{\text{out}})$ | Sí |

Las configuraciones salen del ajuste de hiperparámetros, con el mismo presupuesto (20 configuraciones) para las tres;
el detalle está en el capítulo de implementación.

Los tres modelos difieren en su **sesgo inductivo** (ninguno temporal, memoria secuencial, patrones locales), así que
no son redundantes: compararlos dice qué tipo de estructura temporal aprovecha mejor el problema.

### Equivalencia con la SVR lineal

El enunciado pide justificar que estos tres modelos cumplen, en Deep Learning, el papel de la **SVR lineal** en
regresión clásica: un piso de desempeño que cualquier modelo avanzado debe superar.

- **Capacidad de representación.** La SVR lineal sobre la ventana aplanada es una función lineal de las 672
  entradas. El MLP es su extensión directa: sin capas ocultas sería exactamente un modelo lineal; con dos capas ReLU
  añade la mínima no linealidad. La LSTM y la CNN 1D agregan solo el sesgo temporal más básico de su familia.
- **Supuestos.** Como la SVR lineal, los tres suponen que la ventana reciente contiene la información necesaria y que
  la relación es estable en el tiempo; ninguno usa información espacial ni exógena futura.
- **Propósito comparativo.** Igual que la SVR lineal en un artículo de regresión, estos modelos no pretenden ser los
  mejores, sino fijar el error que se obtiene con las arquitecturas más simples de cada familia. Si un modelo
  avanzado no los supera por más de $\delta$, su complejidad no se justifica.

Como referencia de contexto (no como modelo base, porque no es Deep Learning), el capítulo de benchmark incluye una
regresión lineal regularizada (ridge) sobre la misma ventana aplanada, que es el equivalente con pérdida cuadrática
de la SVR lineal.

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
7. **Brecha entre modelos avanzados y modelos base.** Los resúmenes que la cuantifican reportan mejoras del
   **8 % al 33 % en RMSE** de los modelos propuestos sobre redes base como CNN, LSTM o GRU
   {cite}`qiao2022,jin2025`. Dado el riesgo de fuga y la ausencia de repeticiones, esas cifras deben leerse como
   cotas optimistas; sirven de contexto para el margen que deja el baseline de este trabajo.

**Vacío que aborda este trabajo:** una comparación **justa** de MLP, LSTM y CNN 1D (mismo presupuesto de ajuste,
varias semillas), con **persistencia como baseline**, **partición cronológica con embargo** y **pruebas
estadísticas** de la diferencia, sobre una red de estaciones del Caribe colombiano.

Los Transformers no se incluyen como modelo base porque aparecen solo en el 20 % de los estudios y requieren más
datos; quedan para el Entregable 2. La revisión de {cite}`majidigharehnaz2024` ofrece un panorama más amplio de las
aplicaciones de Deep Learning al pronóstico del viento.

## Limitaciones de la revisión

- La elegibilidad, la tabla de síntesis y la valoración de calidad se basan en los resúmenes, no en el texto
  completo; 31 registros sin resumen quedaron fuera.
- La selección la hizo un único revisor, sin un segundo revisor independiente.
- Las exclusiones de elegibilidad se registraron con la decisión y un motivo general, no con un motivo específico por
  artículo.
- Las búsquedas no usan exactamente los mismos campos en todas las fuentes (en Semantic Scholar, título y resumen; en
  las demás, el título); el tamizaje por título iguala el criterio después de unirlas.
