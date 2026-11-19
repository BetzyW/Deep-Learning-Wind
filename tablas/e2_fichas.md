### Transformer

| Campo | Contenido |
|---|---|
| Identificación | Transformer, familia Atención; {cite}`vaswani2017`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `100 801` parámetros entrenables; `40.34` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `2.80e-04`, wd = `1e-05`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w64-64-64_d010` (anchos y patrón de dropout); `11` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `9` (de `6` a `13`) |
| Cómputo | Corridas finales en A100-SXM4-40GB, L4; `2.1` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.15` h; latencia en CPU `1.11` ms por ventana (lote 1) y `0.7194` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/Transformer.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.614`–`0.815` |
| Incidencias | Ninguna corrida con NaN ni excluida. Corridas finales repartidas entre una A100 y una L4 de Colab. La elegida es la referencia (64, 64, 64) / (0, 1, 0). |

### KAN

| Campo | Contenido |
|---|---|
| Identificación | KAN, familia MLP profundo; {cite}`liu2024kan`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `138 545` parámetros entrenables; `0.28` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `2.80e-04`, wd = `1e-05`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w16-32-128_d010` (anchos y patrón de dropout); `5` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `7` (de `6` a `10`) |
| Cómputo | Corridas finales en A100-SXM4-40GB, L4; `0.5` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.02` h; latencia en CPU `0.80` ms por ventana (lote 1) y `0.0909` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/KAN.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.619`–`0.822` |
| Incidencias | Ninguna corrida con NaN ni excluida. Corridas finales repartidas entre una A100 y una L4 de Colab. |

### TCN

| Campo | Contenido |
|---|---|
| Identificación | TCN, familia Convolucional; {cite}`bai2018`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `98 433` parámetros entrenables; `32.94` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `1.50e-03`, wd = `0`, lote = `1024`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w64-64-16_d110` (anchos y patrón de dropout); `11` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `10` (de `7` a `20`) |
| Cómputo | Corridas finales en GeForce RTX 4060 Ti; `2.0` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.20` h; latencia en CPU `1.10` ms por ventana (lote 1) y `0.1981` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/TCN.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.615`–`0.815` |
| Incidencias | Ninguna corrida con NaN ni excluida. Toda en el PC (RTX 4060 Ti). Modelo de la ablación (enmienda 7). |

### STGCN

| Campo | Contenido |
|---|---|
| Identificación | STGCN, familia Grafos; {cite}`yu2018`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `1 358 881` parámetros entrenables; `167.44` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `1.40e-03`, wd = `0.0001`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | Publicada por los autores; sin búsqueda (sección 14) |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `8` (de `6` a `13`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `8.3` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `2.02` h; latencia en CPU `4.37` ms por ventana (lote 1) y `2.7761` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/STGCN.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.619`–`0.839` |
| Incidencias | Ninguna corrida con NaN ni excluida. Subgrafo de la estación y sus 8 vecinas más cercanas, porque el panel no es una red sincronizada (enmienda 6). |

### Chronos

| Campo | Contenido |
|---|---|
| Identificación | Chronos, familia Preentrenado; {cite}`ansari2024`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | `amazon/chronos-t5-small` (Hugging Face, Apache 2.0), congelado |
| Arquitectura | `21 553` parámetros entrenables; `0.04` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `2.95e-03`, wd = `0.0001`, lote = `2048`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w32-16-256_d100` (anchos y patrón de dropout); `3` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `8` (de `6` a `22`) |
| Cómputo | Corridas finales en L4; `0.0` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.01` h; latencia en CPU `0.07` ms por ventana (lote 1) y `0.0004` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/Chronos.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.709`–`1.134` |
| Incidencias | Ninguna corrida con NaN ni excluida. La representación congelada se calculó en Colab (`embeber.py`); solo se entrena el cabezal. |

### GRU

| Campo | Contenido |
|---|---|
| Identificación | GRU, familia Recurrente; {cite}`cho2014`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `63 425` parámetros entrenables; `20.90` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `8.03e-04`, wd = `0`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w64-64-64_d010` (anchos y patrón de dropout); `11` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `11` (de `9` a `13`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `1.0` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `1.15` h; latencia en CPU `8.82` ms por ventana (lote 1) y `0.2639` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/GRU.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.623`–`0.818` |
| Incidencias | Ninguna corrida con NaN ni excluida. En el PC tardaba de 20 a 77 min por corrida; las fases y la validación con origen móvil se hicieron en Colab. |

### Informer

| Campo | Contenido |
|---|---|
| Identificación | Informer, familia Atención; {cite}`zhou2021`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `11 312 641` parámetros entrenables; `2532.18` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `2.38e-04`, wd = `0.0001`, lote = `1024`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | Publicada por los autores; sin búsqueda (sección 14) |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `7` (de `6` a `10`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `10.2` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `3.39` h; latencia en CPU `16.75` ms por ventana (lote 1) y `12.1891` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/Informer.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.620`–`0.817` |
| Incidencias | Ninguna corrida con NaN ni excluida. Algunas configuraciones del ajuste previo fallaron por memoria y se descartaron sin resultados; los lotes se procesan en trozos de 256 (enmienda 5). |

### DLinear

| Campo | Contenido |
|---|---|
| Identificación | DLinear, familia MLP profundo (paper guía); {cite}`zeng2023`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `338` parámetros entrenables; `0.00` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `1.40e-03`, wd = `0.0001`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | Publicada por los autores; sin búsqueda (sección 14) |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `9` (de `7` a `21`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `0.2` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.01` h; latencia en CPU `0.10` ms por ventana (lote 1) y `0.0018` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/DLinear.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.621`–`0.813` |
| Incidencias | Ninguna corrida con NaN ni excluida. Paper guía; replicación parcial en el capítulo de resultados. |

### Moirai

| Campo | Contenido |
|---|---|
| Identificación | Moirai, familia Preentrenado; {cite}`woo2024`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | `Salesforce/moirai-1.0-R-small` (Hugging Face, CC BY-NC 4.0), congelado |
| Arquitectura | `20 705` parámetros entrenables; `0.04` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `5.56e-04`, wd = `1e-05`, lote = `1024`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w32-128-32_d110` (anchos y patrón de dropout); `7` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `6` (de `6` a `12`) |
| Cómputo | Corridas finales en L4; `0.1` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.01` h; latencia en CPU `0.07` ms por ventana (lote 1) y `0.0004` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/Moirai.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.879`–`1.269` |
| Incidencias | Ninguna corrida con NaN ni excluida. Parche de 32 h, porque `"auto"` falla con 168 h de contexto. Representación calculada en Colab. |

### PatchTST

| Campo | Contenido |
|---|---|
| Identificación | PatchTST, familia Atención; {cite}`nie2023`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `404 995` parámetros entrenables; `17.28` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `8.38e-04`, wd = `0.0001`, lote = `2048`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | Publicada por los autores; sin búsqueda (sección 14) |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `13` (de `6` a `31`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `1.0` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.10` h; latencia en CPU `1.14` ms por ventana (lote 1) y `0.2044` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/PatchTST.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.617`–`0.816` |
| Incidencias | Ninguna corrida con NaN ni excluida. Sustituye a Mamba (enmienda 4). El primer ajuste previo tenía un error de escala y se descartó entero antes de elegir hiperparámetros (`PatchTST_hp_descartado_error_escala.csv`). |

### MLP

| Campo | Contenido |
|---|---|
| Identificación | MLP, familia Modelo base del E1; {cite}`rumelhart1986`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `28 753` parámetros entrenables; `0.06` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `3.51e-04`, wd = `1e-05`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w16-64-256_d001` (anchos y patrón de dropout); `10` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `9` (de `7` a `16`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `0.2` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.01` h; latencia en CPU `0.05` ms por ventana (lote 1) y `0.0006` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/MLP.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.619`–`0.816` |
| Incidencias | Ninguna corrida con NaN ni excluida. — |

### CNN

| Campo | Contenido |
|---|---|
| Identificación | CNN, familia Modelo base del E1; {cite}`lecun1998`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `59 393` parámetros entrenables; `7.83` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `3.51e-04`, wd = `1e-05`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w64-64-64_d010` (anchos y patrón de dropout); `11` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `18` (de `12` a `24`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `0.8` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.04` h; latencia en CPU `0.22` ms por ventana (lote 1) y `0.0650` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/CNN.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.625`–`0.821` |
| Incidencias | Ninguna corrida con NaN ni excluida. — |

### LSTM

| Campo | Contenido |
|---|---|
| Identificación | LSTM, familia Modelo base del E1; {cite}`hochreiter1997`; implementación propia en `src/modelos_e2.py` (commit `cdb80f6`) |
| Preentrenamiento | ninguno |
| Arquitectura | `82 945` parámetros entrenables; `27.35` MFLOPs por ventana de 168 h |
| Hiperparámetros finales | lr = `1.40e-03`, wd = `0.0001`, lote = `512`; espacio: lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}; T = `20` intentos con el 10 % de las ventanas |
| Arquitectura seleccionada | `w32-16-128_d011` (anchos y patrón de dropout); `11` finalistas indistinguibles a menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados |
| Entrenamiento | Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas 2 160 h del entrenamiento; épocas efectivas: mediana `15` (de `8` a `20`) |
| Cómputo | Corridas finales en A100-SXM4-40GB; `1.6` h para las 40 corridas (10 semillas × 4 orígenes); ajuste previo `0.11` h; latencia en CPU `1.33` ms por ventana (lote 1) y `0.1424` ms (lote 1 024); memoria pico de GPU no medida |
| Resultados por ejecución | `resultados/e2/finales/LSTM.csv` (RMSE y MAE de prueba, RMSE de parada y de entrenamiento por semilla y origen); RMSE de prueba por semilla `0.619`–`0.818` |
| Incidencias | Ninguna corrida con NaN ni excluida. — |
