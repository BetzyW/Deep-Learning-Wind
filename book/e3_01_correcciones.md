# Trazabilidad de correcciones

El bloque A del enunciado pide resolver todas las observaciones de la retroalimentación del E2. **El profesor no envió
retroalimentación del E2** antes de esta entrega, así que no hay observaciones del docente que trazar. En su lugar
documentamos con el mismo formato las correcciones que hicimos por **autorrevisión**: errores que encontramos nosotros
y cambios que exige el E3. Si llega la retroalimentación, se agrega a esta tabla.

## Tabla de trazabilidad

| Observación (autorrevisión) | Acción realizada | Ubicación en el libro | Evidencia | Impacto en los resultados |
|---|---|---|---|---|
| El E2 sacó sus conclusiones con 6 semillas (1000–1005); el estándar del E3 exige al menos 10 | Se usan las 10 semillas (1000–1009) que ya estaban corridas para todos los benchmarks | Este capítulo; capítulo de validación | `resultados/e3/ranking_antes_despues.csv` | Los 4 primeros puestos no cambian; cambios de RMSE de como mucho `0.0015` m/s (tabla de abajo) |
| DLinear daba su salida en las unidades de la serie, mientras que el artículo original trabaja sobre la serie escalada y desescala la predicción; en ETTh1 su RMSE era 5 veces el de la persistencia | DLinear desescala su salida con la media y la desviación del entrenamiento (sin parámetros nuevos); se repitieron sus 60 corridas en los conjuntos adicionales | Capítulo de evaluación ampliada; `experiments/conjuntos_adicionales.md` | Corridas descartadas en `resultados/e3/adicionales/*/descartado_DLinear_salida_cruda/` | ETTh1: `10.415` → `1.997`; SDWPF: `388.3` → `379.5` kW; NOAA: `2.476` → `2.479` m/s. En el panel del IDEAM no cambia: allí el sesgo de la capa lineal sí aprendía el nivel |
| STGCN se quedaba sin memoria de GPU al predecir en lotes grandes (134 turbinas en SDWPF) | `predecir` reintenta con lotes de la mitad; el reintento va fuera del `except` para liberar la memoria | Capítulo de evaluación ampliada | `src/modelos_e2.py`, función `predecir` | Ninguno: la predicción es la misma, solo cambia la memoria usada |
| El libro usaba Jupyter Book 2 (`myst.yml`), y la sección 11.2 del enunciado pide el formato clásico | Libro movido a `book/` con `_config.yml`, `_toc.yml` y `references.bib` | Todo el libro | `jupyter-book build book/` sin advertencias | Ninguno sobre los resultados |
| La auditoría de fuga y el determinismo estaban como pruebas, pero sin informe | Informe de auditoría y de determinismo en este capítulo | Este capítulo | `tests/test_fuga.py` (9 pruebas) | Ninguno: las pruebas pasan |
| Faltaba congelar los benchmarks antes de comparar con el modelo original | Huella SHA-256 de cada archivo de resultados de los benchmarks | Este capítulo | `experiments/benchmarks_congelados.csv` | Ninguno |

## Ranking antes y después

```{include} tablas/e3_ranking.md
```

*RMSE medio por corrida en los 4 orígenes de prueba de 2025 (24 corridas con 6 semillas y 40 con 10). Los baselines
son deterministas y no cambian.*

Los cuatro primeros puestos (Transformer, TCN, Ridge y PatchTST) son los mismos con 6 y con 10 semillas. Entre el 5.º y
el 11.º hay intercambios (Informer sube del 7.º al 5.º y el MLP baja del 5.º al 7.º), pero vienen de cambios del RMSE de
como mucho `0.0015` m/s, del orden de la variación entre semillas y muy por debajo de δ = `0.02`. **Las conclusiones
del E2 no cambian**: todos los modelos entrenados quedan en una franja estrecha y ninguno se separa del resto por más de
δ. Este ranking con 10 semillas es el que usamos como referencia en el E3.

## Auditoría de fuga de información

La batería de verificaciones de la sección 10 del E2 está automatizada en `tests/test_fuga.py` y se corre en cada push
(integración continua). Las 9 pruebas pasan:

| Prueba | Qué garantiza |
|---|---|
| `test_ventanas_validas` | Cada ventana tiene sus 168 h de entrada y su objetivo a 24 h dentro del panel |
| `test_embargo_entre_entrenamiento_y_evaluacion` | Entre el último objetivo del entrenamiento y la primera entrada evaluada hay al menos 24 h |
| `test_evaluacion_dentro_de_su_periodo` | Las ventanas evaluadas en cada origen caen dentro de su trimestre |
| `test_pliegues_de_evaluacion_disjuntos` | Ninguna ventana se evalúa en dos orígenes |
| `test_relleno_causal` | El relleno de huecos solo usa el pasado: alterar los datos desde una fecha no cambia nada anterior |
| `test_escalador_y_rezagos_no_miran_el_futuro` | El escalador y los rezagos de cada origen no cambian si se altera todo lo posterior al origen |
| `test_escalador_usa_solo_entradas_de_entrenamiento` | El escalado se ajusta solo con las filas de entrada del entrenamiento |
| `test_objetivos_coinciden_con_los_datos_originales` | El objetivo de cada ventana es la velocidad real 24 h después |
| `test_panel_real` | Repite las comprobaciones del embargo y de los huecos con el panel real del IDEAM, si está disponible |

**Específico del modelo original.** Las anclas de PADRE-TCN salen de la misma ventana de 168 h que ven todos los
modelos (índices L − 1 − 24 j), así que no introducen información nueva ni futura: el ancla 1 usa la velocidad actual,
igual que la persistencia. Los conjuntos adicionales usan el mismo protocolo (escalado con el entrenamiento, partición
cronológica con embargo) y sus objetivos se compararon con los guardados en Colab, con contenido idéntico.

## Determinismo y semillas

`fijar_semilla` (`src/modelos_e2.py`) fija las semillas de Python, NumPy y PyTorch, con `cudnn.deterministic = True` y
`cudnn.benchmark = False`. La evidencia de que funciona: la segunda ronda de diseño reprodujo exactamente los 21 RMSE de
la primera, y el reentrenamiento de PADRE-TCN con dos semillas, días después, dio predicciones idénticas. Entre GPU
distintas (PC y Colab) puede haber diferencias numéricas pequeñas; por eso cada modelo se corrió completo en un mismo
tipo de equipo dentro de cada conjunto.

## Congelamiento de los benchmarks

Antes de comparar con el modelo original, congelamos los resultados de los 13 modelos del benchmark y de los baselines:
`src/trazabilidad_e3.py` calcula el SHA-256 de **572 archivos** (métricas por corrida, configuraciones elegidas y
predicciones por ventana) y los guarda en `experiments/benchmarks_congelados.csv`, con una huella global en
`experiments/benchmarks_congelados.sha256`:

`4728b786ee12cf9ffb6198c268b19dc8cd4bf9ce8def4fa1f1cff5f722e8cfee`

Las huellas se calcularon el 27 de noviembre, después de las comparaciones; lo que garantiza que los benchmarks no cambiaron durante el E3 es git: el último cambio a sus métricas, configuraciones y baselines es del 18 de noviembre (commit `b50ee98`, cierre del E2), y las predicciones por ventana tienen fecha del 12 de noviembre. Desde ese punto, los benchmarks no se reajustan para favorecer la comparación; solo se admiten correcciones de errores
documentadas en este capítulo. Recalcular las huellas y compararlas verifica que ningún resultado cambió.
