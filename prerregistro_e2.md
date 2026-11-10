# Preregistro del Entregable 2

Fijado el 7 de noviembre de 2026, **antes de cualquier entrenamiento del E2** y sin haber evaluado ningún modelo en
2025. Publicado con la etiqueta `prerregistro-e2` en GitHub; los cambios posteriores se registran como enmiendas fechadas.

## 1. Tarea, preguntas e hipótesis

| Campo | Declaración |
|---|---|
| Tipo de tarea | Serie de tiempo (tipo 3), ruta F2a |
| Horizonte y estrategia | Un horizonte, h = 24 h, estrategia directa (heredado del E1) |
| Entrada | L = 168 h de velocidad, dirección (seno y coseno) y temperatura; sin covariables futuras salvo el calendario |
| Unidad de independencia | La estación |

| Pregunta | Hipótesis |
|---|---|
| RQ1. ¿Algún modelo del Top 10 supera al mejor baseline clásico? | H1: al menos uno baja el RMSE en más de δ frente al mejor baseline clásico elegido en validación (hoy Ridge) |
| RQ2. ¿Los modelos preentrenados (Chronos, Moirai) aportan frente a los entrenados desde cero? | H2: el mejor preentrenado con ajuste fino no difiere en más de δ del mejor entrenado desde cero (equivalencia, TOST) |
| RQ3. ¿La limitación del paper guía es propia de DLinear o de toda la familia? | H3: la subestimación en vientos fuertes (≥ 6 m/s) aparece en DLinear y en al menos dos de los otros modelos del Top 10 |

## 2. Métricas y márgenes

| Campo | Declaración |
|---|---|
| Métrica primaria | RMSE en m/s (heredada del E1) |
| Secundarias | MAE, MASE respecto a la persistencia, R²; CRPS, pinball y cobertura para los 3 mejores. No sustentan conclusiones confirmatorias |
| Margen de relevancia δ | 0.02 m/s (heredado del E1) |
| Margen de sobreajuste γ | Razón RMSE validación / RMSE entrenamiento ≤ 1.15; por encima se reporta como sobreajuste |
| Resumen en validación | Se juntan las ventanas de los 4 orígenes (no promedio simple: los pliegues son desiguales) |

## 3. Datos y partición (`configs/e2.json`)

| Campo | Declaración |
|---|---|
| Panel | `panel_multivariado_causal.csv`, SHA-256 `dd0fdfad…c12fb` (E1); estación 1501500054 excluida |
| Origen móvil | Validación: 1-ene, 1-abr, 1-jul, 1-oct de 2024. Prueba: los mismos días de 2025 |
| Entrenamiento | Ventana expansiva desde 2020; solo objetivos anteriores al origen; embargo de 24 h |
| Parada temprana | Últimas 2 160 h del entrenamiento de cada origen, con el mismo embargo |
| Reentrenamiento | Desde cero en cada origen, también en prueba (confirmado por el equipo el 7-nov-2026) |
| Preprocesamiento | Relleno hacia adelante ≤ 3 h; escalado con las filas de entrenamiento de cada origen; verificado por `tests/test_fuga.py` |
| Huella de la prueba | SHA-256 `d5ab640ad37c0cf93bb2505cfd8b676a5b46d6b091d9a20cc8899910b9d35878` (estación, hora objetivo y valor de las 215 286 ventanas de 2025) |
| Huella de la configuración | SHA-256 de `configs/e2.json`: `9a82c6ea804bd5140d0ded2eeed033120c184f740e575d8f022585b4d45f4835` |

## 4. Modelos

| Grupo | Modelos |
|---|---|
| Top 10 (`literatura/e2/puntajes_top10.csv`) | Transformer, KAN, TCN, STGCN, Chronos, Mamba, GRU, Informer, DLinear, Moirai |
| Base del E1 | MLP, LSTM, CNN 1D |
| Baseline clásico fuerte | Ridge y gradient boosting sobre rezagos (`src/baselines.py`), con el mismo presupuesto de ajuste |
| Baselines triviales | Persistencia (= naive estacional diaria), naive semanal, media de la misma hora en 7 días; ETS por estación |
| Paper guía | DLinear (Zeng et al., AAAI 2023, DOI 10.1609/aaai.v37i9.26317) |

**Arquitectura definida en el paper** (se usa su configuración publicada; solo se ajustan los demás
hiperparámetros): Informer, DLinear, STGCN, Chronos y Moirai (pesos oficiales; solo el cabezal de 3 capas densas
entra en la búsqueda, como dice la sección 14.2). **Sin arquitectura definida para este problema** (entran a la
búsqueda de la sección 14): Transformer, KAN, TCN, Mamba, GRU, y los tres modelos base.

## 5. Ajuste y búsqueda de arquitectura

| Campo | Declaración |
|---|---|
| Hiperparámetros no arquitectónicos | Búsqueda aleatoria, T = 20 configuraciones por modelo, semilla 7, con la arquitectura de referencia (64, 64, 64) y dropout (0, 1, 0); solo validación |
| Espacio de arquitectura | 3 bloques, anchos W = {16, 32, 64, 128, 256}, dropout p = 0.2 con patrón b ∈ {0,1}³: 1 000 arquitecturas por modelo |
| Fase 1 (cribado) | Las 1 000, una semilla (2000), 25 % de las épocas máximas (12) con paciencia 3, en el último origen de validación (2024-10, 43 estaciones). Es un cribado, no una afirmación de optimalidad |
| Fase 2 (confirmación) | k = 20 finalistas más la referencia; 5 semillas (2001–2005), presupuesto completo, los 4 orígenes de validación |
| Regla de selección | Menor RMSE de validación medio en la fase 2; empate (diferencia < δ/2) → menos parámetros |
| Entrenamiento | Adam, MSE, lote 1 024, máximo 50 épocas, paciencia 5 |
| Reentrenamiento final | Con entrenamiento + parada de cada origen y la configuración elegida, igual para todos |
| Costo estimado C_total | ≈ 40 h de GPU para las fases 1 y 2 (8 modelos con búsqueda); se medirá en el piloto y se reportará |
| Registro | Todas las configuraciones evaluadas, no solo la ganadora |

## 6. Ejecuciones e inferencia

| Campo | Declaración |
|---|---|
| Número de ejecuciones R | 6 semillas (1000–1005). El piloto del E1 dio σ_d ≤ 0.004 m/s; se repetirá un piloto con 5 semillas en DLinear, TCN y Transformer, y si la fórmula da más de 6, se usa ese valor |
| Familia F1 (confirmatoria) | Cada modelo del Top 10 contra el mejor baseline clásico: 10 contrastes, corrección de Holm |
| Familia F2 (confirmatoria) | Cada modelo del Top 10 contra la persistencia: 10 contrastes, Holm |
| Familia F3 (confirmatoria) | Limitación del paper guía (RQ3): sesgo en vientos ≥ 6 m/s en DLinear y en 3 modelos del Top 10, Holm |
| Prueba principal | Wilcoxon de rangos con signo sobre las diferencias de RMSE por estación (la estación es la unidad) |
| Pruebas de apoyo | Diebold–Mariano con corrección HLN, Model Confidence Set, Friedman con Nemenyi entre todos los modelos, Clark–West y Giacomini–White, TOST con ±δ para RQ2 |
| Si se violan supuestos | Se usan las versiones no paramétricas y bootstrap por bloques de estación |
| Nivel | α = 0.05 |

## 7. Exclusiones y análisis exploratorios

- **Exclusión de ejecuciones:** pérdida NaN o RMSE de validación mayor que el doble de la persistencia. Se
  reemplaza con la siguiente semilla y se reporta.
- **Exploratorios (no se usan para decidir):** errores por estación y por mes, relación con la predictibilidad
  intrínseca (Ω), sensibilidad de los pesos de la tabla de puntajes, comparación con y sin preentrenamiento,
  interpretabilidad.

## 8. Paper guía

- **Replicación:** DLinear en ETTh1 con la ventana y el horizonte de una tabla del paper (por verificar contra el
  texto). Criterio de éxito fijado antes: el MSE reportado cae dentro del IC 95 % de 5 réplicas o la diferencia es
  menor que 0.01.
- **Limitación declarada:** subestimación de los vientos fuertes; se mide como el error medio (predicho − observado)
  en ventanas con observación ≥ 6 m/s, con IC 95 % por bootstrap de estaciones.

## Enmienda 1 — 7 de noviembre de 2026 (antes de la fase 1; ningún resultado de búsqueda visto)

Motivo: costo. Con todos los datos, una época de la TCN tarda ≈ 18 s, y la fase 1 completa de 8 modelos superaría
las 100 h de GPU.

1. El ajuste previo de hiperparámetros y la fase 1 usan el **10 % de las ventanas de ajuste** del último origen de
   validación, muestreadas una vez con semilla 7 e idénticas para todas las configuraciones y arquitecturas. La
   evaluación sigue siendo sobre todas las ventanas de validación de ese origen.
2. Espacio del ajuste previo (T = 20, semilla 7): tasa de aprendizaje log-uniforme en [1e-4, 3e-3], weight decay en
   {0, 1e-5, 1e-4}, tamaño de lote en {512, 1024, 2048}; máximo 20 épocas, paciencia 3.
3. Una arquitectura que se queda sin memoria de GPU se registra como excluida y se reporta.
4. Se habilita TF32 en las multiplicaciones de matrices (igual para todos los modelos).
5. **Mamba:** la biblioteca oficial solo tiene núcleos CUDA para Linux y la implementación en PyTorch puro tarda
   ≈ 51 s por lote con w = 256. Su búsqueda queda en espera de una decisión documentada (ejecutarla con los núcleos
   oficiales en WSL o sustituirla por el siguiente modelo del ranking) que se registrará como enmienda antes de
   entrenarla. El resto de modelos sigue el protocolo.

## Enmienda 2 — 8 de noviembre de 2026 (durante la fase 1 de la GRU; ningún finalista elegido)

Motivo: costo. Las recurrentes escalan mal con el ancho: en la GRU, una arquitectura con última capa de 256 tarda
≈ 375–670 s (frente a ≈ 25 s con 16). A las 7:15 iban 117 de 1 000 arquitecturas, todas con primera capa de 16 (las
más baratas), y la fase 1 completa de la GRU y la LSTM no cabía antes de la entrega (15-nov).

1. **GRU y LSTM:** la fase 1 evalúa una muestra aleatoria simple de **200 de las 1 000 arquitecturas**, fijada con la
   semilla 11 (`busqueda_arq.py`, `MUESTRA`) antes de mirar resultados y guardada en
   `resultados/e2/busqueda/{modelo}_muestra_fase1.json`. La muestra cubre todos los anchos de forma pareja.
2. Las arquitecturas de la GRU ya calculadas que no están en la muestra (88 de 117) se conservan y se reportan, pero
   **no entran en la elección de finalistas**, para no sesgar la búsqueda hacia las redes pequeñas que se evaluaron
   primero.
3. La fase 2 no cambia: k = 20 finalistas de la muestra más la referencia (64, 64, 64) / (0, 1, 0).
4. Los demás modelos (MLP, CNN, TCN, KAN, Transformer) siguen con las 1 000 arquitecturas.
5. Limitación que se reportará: con 200 de 1 000, la probabilidad de que al menos una arquitectura del 5 % mejor del
   espacio entre en la muestra es 1 − 0.95²⁰⁰ ≈ 0.99997; la búsqueda sigue siendo un cribado, no una garantía de óptimo.

## Enmienda 3 — 9 de noviembre de 2026 (fase 1 de TCN y Transformer en curso; ningún finalista elegido, prueba sin abrir)

Motivo: costo, medido. La estimación previa (C_total ≈ 40 h) resultó muy por debajo del costo real. Con los tiempos
medidos (piloto con presupuesto completo y fase 1 de MLP, CNN, GRU, LSTM y parte de TCN y Transformer), un
entrenamiento completo tarda ≈ 10 veces uno de la fase 1, y la fase 2 tal como estaba declarada (21 arquitecturas ×
5 semillas × 4 orígenes = 420 entrenamientos completos por modelo) costaría ≈ 200–350 h de GPU en el equipo del
proyecto (RTX 4060 Ti, 8 GB), que además solo puede usarse de noche. El cómputo adicional disponible (Colab Pro+, un
presupuesto fijo de unidades) debe alcanzar también para el Entregable 3, que exige al menos 10 semillas por modelo y
tres conjuntos de datos adicionales. La sección 14.4 del enunciado indica que, si el cómputo no alcanza, la desviación
se declara, se justifica y se reporta su efecto; esta enmienda lo hace antes de elegir finalistas y sin haber abierto
la prueba.

1. **Cómputo adicional:** Google Colab Pro+ (GPU NVIDIA A100 de 40 GB) además del PC. El presupuesto de cada modelo se
   define en número de arquitecturas, épocas y semillas, no en tiempo, así que no depende del equipo; el equipo de
   cada resultado queda registrado (`resultados/e2/busqueda/hardware.jsonl`), los tiempos de entrenamiento se reportan
   con su GPU y la latencia de inferencia se mide para todos los modelos en el mismo equipo. Misma versión de torch
   (2.14.1; CUDA 12.6 en el PC y 13.0 en Colab), numpy (2.4.6) y pandas (3.0.6).
2. **Fase 1 de TCN, KAN y Transformer:** la misma regla de la enmienda 2, una muestra aleatoria simple de **200 de las
   1 000 arquitecturas** con semilla 11 (`MUESTRA` en `busqueda_arq.py`). Es la misma muestra para los cinco modelos
   (GRU, LSTM, TCN, KAN y Transformer), lo que hace comparables sus búsquedas. Las arquitecturas ya calculadas fuera de
   la muestra (TCN: 101 de 134; Transformer: las primeras del orden de enumeración, todas con primera capa de 16) se
   conservan y se reportan, pero no entran en la elección de finalistas. La búsqueda aleatoria sobre un espacio fijo
   es un método reconocido de ajuste (Bergstra y Bengio, 2012, *Journal of Machine Learning Research*, 13, 281–305).
3. **Fase 2, para todos los modelos con búsqueda:** k = **10** finalistas (mínimo de la sección 14.4) más la referencia
   (64, 64, 64) / (0, 1, 0); 5 semillas (2001–2005); presupuesto completo (todas las ventanas de ajuste, máximo 50
   épocas, paciencia 5); **un origen**: el último de validación (2024-10-01), el mismo de la fase 1. Pasa de 420 a 55
   entrenamientos por modelo.
4. **Análisis de sensibilidad del muestreo:** MLP y CNN tienen la fase 1 completa. A sus 1 000 resultados se les
   aplica la misma muestra de 200 y se reporta cuántos de sus 10 finalistas y si su arquitectura ganadora habrían
   cambiado. Con eso se cuantifica, con datos del propio problema, el efecto del muestreo sobre las conclusiones.
5. **Sin cambios:** el espacio de arquitecturas, el ajuste previo de hiperparámetros, la regla de selección, las
   corridas finales (4 orígenes de prueba de 2025, reentrenando en cada uno) y las pruebas estadísticas. Las corridas
   finales usan las semillas 1000–1005 declaradas (R = 6); si se agregan semillas para el Entregable 3, se reportan
   aparte y las conclusiones del Entregable 2 se basan en las 6 declaradas.
6. **Costo declarado (sección 14.4):** C_total revisado ≈ 60–80 h de GPU entre el PC y Colab; se reportará el consumo
   real por equipo.
7. **Limitaciones que se reportarán:** (a) cinco modelos evalúan 200 arquitecturas y dos (MLP y CNN) 1 000; el punto 4
   mide el efecto. (b) Con un solo origen en la fase 2, la elección de la arquitectura no promedia sobre estaciones del
   año; el desempeño final sí se mide con origen móvil en los 4 orígenes de 2025. (c) Dos tipos de GPU: las
   diferencias numéricas entre equipos son del orden del ruido de semilla y se documentan. (d) La fase 1 sigue siendo
   un cribado, no una afirmación de optimalidad.
8. **Mamba** sigue en espera, según el punto 5 de la enmienda 1; su decisión se registrará en otra enmienda antes de
   entrenarla.

## Enmienda 4 — 9 de noviembre de 2026 (Mamba nunca entrenada; ningún resultado de Mamba ni de PatchTST visto)

Motivo: viabilidad y costo, como anticipaba el punto 5 de la enmienda 1. La biblioteca oficial de Mamba solo tiene
núcleos CUDA para Linux; la implementación en PyTorch puro tarda ≈ 51 s por lote con w = 256, y montar WSL con los
núcleos oficiales consumiría tiempo y cómputo que el presupuesto fijo de Colab (enmienda 3) no permite.

1. **Mamba se sustituye por PatchTST** (Nie et al., 2023, *ICLR*, arXiv:2211.14730), el 11.º modelo del ranking de la
   revisión sistemática (puntaje 0.550). En la revisión había quedado fuera porque la familia de atención ya tenía dos
   representantes; con la salida de Mamba, el Top 10 pasa de 7 a 6 familias (el mínimo del enunciado es 4) y la
   familia de atención queda con tres modelos (Transformer, Informer y PatchTST). Se reporta como limitación: la
   familia de espacio de estados queda sin representante.
2. **PatchTST tiene arquitectura publicada** (sección 14.2): no entra en la búsqueda de arquitectura y recibe el mismo
   ajuste previo de hiperparámetros que los demás (T = 20, semilla 7). Configuración supervisada del artículo para
   conjuntos grandes: parches P = 16 con paso S = 8, 3 capas de codificador, H = 16 cabezas, D = 128, feed-forward
   F = 256, dropout 0.2, normalización por instancia (RevIN) e independencia de canal. Adaptaciones declaradas:
   ventana de entrada L = 168 (la común a todos los modelos, en lugar de 336), solo la serie de velocidad (como
   DLinear) y un ajuste afín de 2 parámetros a la salida para pasar de la escala estandarizada a m/s. Implementación
   propia desde la formulación (`src/modelos_e2.py`).
3. DLinear, PatchTST e Informer (arquitecturas publicadas) se ajustan con el mismo procedimiento previo y van
   directo a las corridas finales.

## Enmienda 5 — 9 de noviembre de 2026 (Informer nunca entrenado)

Configuración de Informer (Zhou et al., 2021, *AAAI*), arquitectura publicada sin búsqueda (sección 14.2), con la
configuración por defecto del artículo: D = 512, 8 cabezas, feed-forward 2048, 2 capas de codificador con destilación
(Conv1d k = 3 + BatchNorm + ELU + MaxPool), 1 capa de decodificador, atención ProbSparse con factor c = 5 y dropout
0.05; decodificador generativo con las últimas 48 h y 24 posiciones en cero. Adaptaciones declaradas: ventana L = 168
(común a todos los modelos), salida en la última posición del decodificador (pronóstico directo a τ + 24) y sin
incrustación de calendario, porque el panel no trae esas variables. Implementación propia desde la formulación
(`src/modelos_e2.py`). Recibe el mismo ajuste previo de hiperparámetros que los demás (T = 20, semilla 7).
