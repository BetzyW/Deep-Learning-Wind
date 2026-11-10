# Revisión sistemática, Top 10 y paper guía (Entregable 2)

La revisión del E1 eligió tres modelos base. Esta amplía la búsqueda a las arquitecturas recientes de Deep Learning
para elegir, con criterios fijados antes de calcular, los 10 modelos del benchmark, el paper guía y el hueco de
investigación. Se sigue la guía PRISMA 2020 {cite}`page2021`. Los registros, las consultas exactas y los scripts están
en `literatura/e2/`; las exportaciones de Scopus y Web of Science no se publican porque sus términos no permiten
redistribuir los resúmenes.

## Protocolo

| Elemento | Decisión |
|---|---|
| Bases | Scopus y Web of Science; Semantic Scholar y arXiv como complemento |
| Periodo | 2018–2026; se citan los trabajos fundacionales anteriores (Transformer 2017, GRU 2014) |
| Cadenas sobre el problema | C1 atención · C2 convolucionales y MLP modernos · C3 espacio de estados · C4 preentrenados · C5 grafos · C6 probabilísticos; siempre con *wind speed* y *forecast/predict* en el título |
| Cadenas transversales | Mecanismos de otras aplicaciones que podrían transferirse (base para el modelo original del E3) |
| Registro | `registro_busquedas.csv`: fecha, base, consulta exacta y número de resultados |

## Flujo PRISMA

```{mermaid}
flowchart TD
  A["Identificados: 1 998<br/>Scopus 470 · WoS 322 · Semantic Scholar 1 176 · arXiv 30"] --> B["Duplicados eliminados: 656"]
  B --> C["Registros únicos: 1 342"]
  C --> D["Excluidos por título o periodo: 813"]
  C --> E["Tamizados por resumen: 529"]
  E --> F["Sin resumen o sin arquitectura del diccionario: 65"]
  E --> G["Evaluados para elegibilidad: 464"]
  G --> H["Revisiones o no primarios: 18"]
  G --> I["Sin métrica cuantitativa: 241"]
  G --> J["Elegibles: 205"]
```

## Tabla de puntajes y Top 10

Cinco criterios del enunciado, cada uno en [0, 1] y con el mismo peso (0.2), fijados antes de calcular: citas por año
del artículo original, recencia, desempeño en viento (estudios elegibles que lo nombran en el título), implementación
disponible y representatividad de la familia. MLP, LSTM y CNN 1D (los modelos base del E1) entran aparte.

| # | Modelo | Familia | Año | Citas/año | Estudios de viento | Puntaje |
|---|---|---|---|---|---|---|
| 1 | Transformer {cite}`vaswani2017` | Atención | 2017 | 20 164 | 36 | 0.860 |
| 2 | KAN {cite}`liu2024kan` | MLP profundo | 2024 | 785 | 3 | 0.769 |
| 3 | TCN {cite}`bai2018` | Convolucional | 2018 | 860 | 7 | 0.690 |
| 4 | STGCN {cite}`yu2018` | Grafos | 2017 | 572 | 13 | 0.688 |
| 5 | Chronos {cite}`ansari2024` | Preentrenado | 2024 | 422 | 0 | 0.671 |
| 6 | ~~Mamba~~ {cite}`gu2023` | Espacio de estados | 2023 | 2 526 | 1 | 0.649 |
| 7 | GRU {cite}`cho2014` | Recurrente | 2014 | 2 183 | 8 | 0.648 |
| 8 | Informer {cite}`zhou2021` | Atención | 2020 | 1 368 | 7 | 0.646 |
| 9 | DLinear {cite}`zeng2023` | MLP profundo | 2022 | 1 028 | 0 | 0.561 |
| 10 | Moirai {cite}`woo2024` | Preentrenado | 2024 | 297 | 0 | 0.560 |
| 11 | **PatchTST** {cite}`nie2023` (sustituye a Mamba, enmienda 4) | Atención | 2022 | 1 060 | 1 | 0.550 |

**Por qué pesos iguales:** no hay evidencia para favorecer un criterio, y fijarlos antes evita ajustar el Top 10 a
gusto. **Por qué los preentrenados tienen 0 estudios en viento:** Chronos y Moirai son de 2024 y casi nadie los ha
probado en velocidad del viento; ese vacío es parte del hueco que el benchmark llena. **Limitación del criterio 3:**
contar estudios no mide el error, y los autores suelen favorecer su propio modelo.

**Mamba** quedó fuera porque su biblioteca oficial solo funciona en Linux y su versión en PyTorch puro era inviable en
el presupuesto; entró el siguiente del ranking, PatchTST. El Top 10 final tiene 6 familias (el mínimo es 4) y la de
espacio de estados queda sin representante, lo que se declara como limitación.

## Hoja de extracción

Campos fijos leídos del resumen de los 205 estudios elegibles («no» significa que el resumen no lo menciona):

| Qué reporta el estudio | Estudios | % |
|---|---|---|
| Describe la partición de datos | 23 | 11 % |
| Varias semillas o repeticiones | 2 | 1 % |
| Prueba estadística (Diebold–Mariano, Wilcoxon, etc.) | 4 | 2 % |
| Código público | 1 | 0.5 % |
| Datos públicos | 10 | 5 % |
| Descompone la señal (EMD, VMD, ondículas: riesgo de fuga) | 84 | 41 % |
| Se compara con la persistencia | 5 | 2 % |
| Varias estaciones | 50 | 24 % |
| Horizonte de 24 h o más | 14 | 7 % |
| Incertidumbre (intervalos, cuantiles) | 41 | 20 % |

Ningún estudio compara contra la persistencia **y** usa pruebas estadísticas; ninguno usa varias semillas **y**
pruebas; ninguno evalúa modelos preentrenados, y solo 3 combinan varias estaciones con horizonte de 24 h o más (sin
persistencia ni pruebas). Entre los estudios que respaldan el Top 10, ninguno se compara con la persistencia y una
parte alta descompone la señal sin describir la partición, un riesgo conocido de fuga de información.

## Hueco de investigación

> Ningún estudio incluido compara modelos preentrenados (Chronos, Moirai) y arquitecturas recientes contra la
> persistencia y una regresión lineal fuerte, en un panel de varias estaciones a 24 h, con varias semillas e
> inferencia estadística.

El hueco es verificable: se refuta si aparece en `hoja_extraccion.csv` un estudio con las columnas
`compara_persistencia`, `pruebas_estadisticas`, `semillas_o_repeticiones` y `multiestacion` verdaderas. Hoy no hay
ninguno.

## Paper guía: DLinear y su replicación

El paper guía es DLinear {cite}`zeng2023`, elegido por tres razones: se replica con datos públicos y en minutos;
conecta con el hallazgo del E1 (una regresión lineal igualó a las redes base), y su limitación, la de un modelo lineal
frente a regímenes no lineales como los vientos fuertes, se puede medir y es el punto de partida del modelo original
del E3.

**Replicación con sus condiciones originales.** Se verificaron contra el artículo (tabla 2 y apéndice B.2) y el código
oficial de los autores: ETTh1 multivariado (7 variables), ventana L = 336, horizonte T = 96, partición 12/4/4 meses,
estandarización con el entrenamiento, DLinear compartido entre canales con media móvil de 25 pasos, Adam con tasa
0.005, lote 32, 10 épocas con paciencia 3 y tasa a la mitad por época; MSE en la escala estandarizada.

| | MSE | MAE |
|---|---|---|
| Reportado por el artículo (tabla 2) | 0.375 | 0.399 |
| Nuestra réplica (5 semillas) | 0.379 (IC 95 % [0.370, 0.389]) | 0.402 |

El criterio de éxito, fijado en el preregistro, era que el MSE reportado cayera dentro del IC 95 % de 5 réplicas o
que la diferencia fuera menor que 0.01. **Se cumple por ambos lados**: 0.375 está dentro del intervalo y la diferencia
es 0.004 (`src/replicacion_dlinear.py`, `resultados/e2/replicacion_dlinear.json`).
