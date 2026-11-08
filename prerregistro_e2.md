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
