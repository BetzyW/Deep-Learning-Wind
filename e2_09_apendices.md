# Apéndices

Este capítulo reúne los entregables formales que pide el enunciado y que no caben en el cuerpo del informe: la ficha
de ruta metodológica, la trazabilidad del EDA, el contrato de datos, las fichas de modelo, el registro de
experimentos, el entorno de reproducción y la lista de verificación del Apéndice B. Todas las tablas salen
directamente de los archivos de resultados (`src/tablas_libro.py`, `src/fichas_modelo.py`); ningún número se copió a
mano.

## Ficha de ruta metodológica

**Tipo de problema:** F2a, series de tiempo (sección 7.4 del enunciado). Pronosticamos la velocidad media horaria del
viento 24 h adelante en un panel de 62 estaciones. El pronóstico puntual es el principal y el probabilístico (los
intervalos) es secundario. La métrica primaria declarada es el RMSE por estación; el MASE frente a la persistencia y
el MAE se reportan como secundarias.

### Pasos de la ruta F2a

| Paso | Qué pide | Qué hicimos | Dónde |
|---|---|---|---|
| S1 | Horizonte, frecuencia, estrategia y covariables futuras | h = 24 h, frecuencia horaria, estrategia directa, sin covariables futuras (solo el pasado de velocidad, dirección y temperatura) | `configs/e2.json`, preregistro §1 y §3 |
| S2 | EDA temporal | Estacionariedad (ADF y KPSS), estacionalidad diaria, autocorrelación hasta 168 h, ARCH, quiebres de nivel (Pettitt) y predictibilidad intrínseca | Capítulo 2 del E1, capítulo de datos del E2 |
| S3 | Partición con origen móvil y embargo | 4 orígenes trimestrales de validación (2024) y 4 de prueba (2025), embargo de 24 h, ventana expansiva | Capítulo de datos; contrato de datos (abajo) |
| S4 | Preprocesamiento solo con el pasado | Escalado ajustado con el entrenamiento de cada origen; relleno solo hacia adelante; rezagos con el pasado | Auditoría de fuga (9 pruebas) |
| S5 | Líneas base y 10 modelos con igual presupuesto | Persistencia, naive semanal, media de la misma hora, ETS por estación, Ridge y gradient boosting; 10 modelos del Top 10 y 3 del E1 con el mismo ajuste previo (T = 20) y la misma búsqueda de arquitectura | Capítulo de métodos; fichas de modelo |
| S6 | Evaluación por horizonte y por origen | Un solo horizonte (24 h); RMSE por origen, por estación y agregado | Capítulo de resultados |
| S7 | Residuos, calibración y estabilidad | Ljung–Box, ARCH-LM, BDS, Jarque–Bera, Mincer–Zarnowitz, PIT y cobertura, Giacomini–Rossi | Capítulo de resultados, diagnóstico de los tres mejores |
| S8 | Comparación y robustez | Diebold–Mariano (HLN), MCS, Friedman con Nemenyi; robustez ante ruido, horas faltantes y quiebres de nivel | Capítulo de resultados |

Los detalles críticos de la ruta también se cumplen. No hay barajado ni validación cruzada aleatoria, y las
covariables solo usan el pasado. La referencia ingenua se reporta con el MASE y con la familia F2 frente a la
persistencia. Se compara un modelo local (ETS por estación) con los globales, y el error se reporta por estación. La
política de reentrenamiento es la misma para todos: se reentrena en cada origen.

### Tabla A: bondad de ajuste y adecuación

| Prueba | H0 | Resultado | Decisión |
|---|---|---|---|
| Comparación contra el pronóstico ingenuo (alternativa a la permutación por bloques que admite el enunciado) | El modelo no mejora a la persistencia | Familia F2 con Holm: 12 de los 13 modelos mejoran a la persistencia; los 11 entrenados desde cero entre `0.14` y `0.16` m/s por estación y Chronos con cabezal `0.03` | Los modelos capturan relación. Moirai con cabezal no la mejora (en promedio es `0.12` m/s peor, p = `0.186`) |
| Mincer–Zarnowitz con errores HAC | a = 0 y b = 1 | Transformer b = `1.000` (p = `0.608`), TCN `1.006` (p = `0.222`); se rechaza en 8 de los 13 modelos y en Ridge, con b entre `0.99` y `1.03` en los entrenados desde cero | Sesgos pequeños; los preentrenados con cabezal tienen b ≈ `1.15` |
| Residuos de serie de tiempo: Ljung–Box, ARCH-LM y BDS | Sin autocorrelación, sin ARCH, residuos i.i.d. | Se rechazan en el 98–100 % de las estaciones para los tres mejores y Ridge | Queda estructura temporal y de volatilidad sin capturar; por eso usamos IC por bootstrap y DM con HAC |
| Normalidad de residuos (Jarque–Bera) | Residuos normales | Se rechaza en el 92 % de las estaciones | Inferencia no paramétrica (Wilcoxon) y bootstrap |
| PIT y cobertura de los intervalos | PIT uniforme; cobertura = nominal | PIT: KS con p < `0.001`. Cobertura del 80 %: `76.6`–`76.9` % (conformal) y `81.3` % en Moirai sin entrenar | Intervalos algo estrechos y peores con viento alto |

### Tabla B: ausencia de sobreajuste

**Brecha entrenamiento–validación con margen.** En cada corrida final calculamos el gap_ratio del curso,
g = |RMSE de entrenamiento − RMSE de parada| / RMSE de parada, con el umbral δg = 0.15 del curso. La prueba es un
Wilcoxon unilateral con H0: mediana(g) ≥ 0.15, sobre 24 corridas por modelo (6 semillas × 4 orígenes), con Holm
entre los 13 modelos. En la misma tabla va la **sensibilidad a la semilla**: el coeficiente de variación del RMSE de
prueba entre las 6 semillas, y Friedman entre semillas con los orígenes como bloques.

```{include} tablas/e2_brecha.md
```

La brecha queda por debajo de 0.15 con significancia en 12 de los 13 modelos. En Chronos la brecha mediana es
`0.10` y la prueba no alcanza (p = `0.076`): no concluyente. Las semillas mueven el RMSE menos del 1 % en todos salvo
Chronos (`1.7` %), y ninguna semilla es sistemáticamente mejor (Friedman, ningún p menor que 0.05 tras Holm).

**Validación frente a prueba.** H0: s_val − s_test = 0, con IC por bootstrap de estaciones, para los tres mejores. La
persistencia se agrega como control, porque no se ajusta y solo puede reflejar el cambio entre periodos.

```{include} tablas/e2_val_prueba.md
```

El error de validación del último trimestre de 2024 es menor que el de 2025, pero la persistencia muestra una
diferencia aún mayor. La brecha viene del cambio entre periodos, que ya se vio en el EDA, y no de sobreajuste a la
validación.

**Estabilidad temporal (Giacomini–Rossi).** Frente a Ridge, la ventaja de TCN y Transformer no es estable en el
tiempo: el estadístico máximo es `5.34` y `7.46`, por encima del valor crítico de `3.43`. La de KAN sí lo es
(`3.27`). Las curvas de entrenamiento y validación por época están en la sección de *forward chaining*.

### Tabla C: diferencias entre modelos

| Prueba | Resultado | Dónde |
|---|---|---|
| Wilcoxon por estación con Holm (F1 frente a Ridge, F2 frente a la persistencia) | 7 modelos le ganan a Ridge con significancia, ninguno por más de δ | Capítulo de resultados, familias |
| Diebold–Mariano con HLN y varianza HAC de 23 rezagos | Coincide en la dirección con Wilcoxon | Ídem |
| Clark–West (anidados frente a la persistencia) y Giacomini–White | Todos los entrenados desde cero superan a la persistencia | Ídem |
| Model Confidence Set (α = 0.10, bloques de 168 h) | 9 modelos indistinguibles del mejor; Ridge queda fuera por poco | Capítulo de resultados |
| Friedman con la corrección de Iman–Davenport | F = `93.2` con (14, 854) grados de libertad, p < `0.001` | `resultados/e2/estadistica/ruta_evidencias.json` |
| Post-hoc de Nemenyi | Diferencia crítica de `2.72` rangos | Capítulo de resultados |
| TOST con ±δ (RQ2) | Chronos y Moirai no son equivalentes al mejor desde cero, ni con cabezal ni sin entrenar | Capítulo de resultados, RQ2 |
| Benjamini–Hochberg como sensibilidad de Holm | En F1, BH declara 11 diferencias significativas frente a 9 con Holm: se suman CNN y DLinear, ambos **peores** que Ridge. En F2 no cambia nada. Ninguna conclusión cambia | `ruta_evidencias.json` |

**Pruebas de las tablas que no aplicamos, y por qué.** No aplicamos las de clasificación (McNemar, DeLong, Hosmer–
Lemeshow), porque no corresponden a un pronóstico continuo. Tampoco el t con la corrección de Nadeau–Bengio ni el 5×2
cv, que suponen particiones aleatorias, prohibidas en series de tiempo. El optimismo por bootstrap y la validación
anidada exigen reentrenar cientos de veces, más allá del presupuesto declarado; la comparación validación–prueba de
arriba cumple el mismo papel. Las curvas de aprendizaje por tamaño de entrenamiento no se hicieron. La validación
adversarial no se hizo, porque el cambio entre periodos ya está documentado. La prueba bayesiana con ROPE no se hizo,
porque el TOST con ±δ responde la misma pregunta de equivalencia.

## Trazabilidad hallazgo–decisión del EDA

| Hallazgo | Evidencia | Decisión de preprocesamiento | Impacto en arquitectura o validación | Riesgo y fuga |
|---|---|---|---|---|
| Presión faltante en el 18 %, concentrada en estaciones altas | Faltantes por estación y altitud: patrón MAR (E1) | Se descarta la presión: 4 entradas por hora | Todos los modelos ven las mismas 4 variables | Imputar con datos futuros; se evita |
| Anemómetro bloqueado en la estación 1501500054 | 99.8 % de horas en cero (E1) | Se excluye | — | Inflar el desempeño con ceros fáciles |
| El ciclo diario explica cerca de 2/3 de la varianza | Descomposición y ACF (E1) | Ninguna | Persistencia a 24 h obligatoria; la ablación confirma el valor del ciclo diario | Ninguno |
| ACF significativa hasta 168 h | ACF y PACF (E1) | Ventana L = 168 h | Campo receptivo suficiente en TCN, Transformer y recurrentes | Ninguno |
| Sin raíz unitaria, media cambiante | ADF y KPSS (E1) | No se diferencia; se estandariza por origen | Reentrenar en cada origen | Escalado con el futuro; probado en la auditoría |
| Efecto ARCH en todas las estaciones | ARCH-LM (E1 y E2) | Ninguna | DM con HAC; intervalos que crecen con el nivel como pista para el E3 | Ninguno |
| Estaciones heterogéneas con poca historia cada una | Predictibilidad Ω por estación | Modelo global | La estación es la unidad de independencia | Ninguno |
| Cambio 2024 → 2025 por composición de la red | Conteo de estaciones por trimestre | Partición cronológica con embargo; prueba con huella SHA-256 | Origen móvil; validación frente a prueba con la persistencia como control | Selección con la prueba; prohibida por el preregistro |
| Quiebres de nivel locales, no regionales | Prueba de Pettitt por estación (E1) | Ninguna | Prueba de robustez con quiebres de nivel | Ninguno |
| Atípicos escasos | Valores con \|z\| > 6 y su efecto en el RMSE (E1) | Se conservan | Error en vientos fuertes reportado aparte (RQ3) | Ninguno |

## Contrato de datos

El contrato se ejecuta automáticamente cada vez que se carga el panel (`src/contrato_datos.py`, llamado desde
`preparar_panel`). Si una regla falla, el programa se detiene: no se entrena con datos que violen el contrato. Las
reglas comprueban:

- Que estén las columnas obligatorias, sin faltantes en el código, la hora y la velocidad.
- Que las horas estén en punto y los códigos tengan 10 dígitos.
- Que cada variable esté dentro de su rango.
- Que la media vectorial de la dirección tenga módulo de 1 o menos, y que su seno y coseno falten a la vez.

**Esquema y diccionario de atributos.** Todas las variables son causalmente anteriores al objetivo: se observan hasta
la hora de emisión τ, y el objetivo es la velocidad en τ + 24 h.

```{include} tablas/e2_esquema.md
```

**Poblaciones y transformaciones, en orden desde el archivo original** (`resultados/e2/contrato_datos.json`). En los
pasos de validación y prueba, la columna de ventanas da las de entrenamiento y las de evaluación.

```{include} tablas/e2_poblaciones.md
```

## Fichas de modelo

Siguen la plantilla de la sección 11.3. Los hiperparámetros se eligieron con el mismo ajuste previo para todos (20
configuraciones al azar del mismo espacio y con la misma semilla), y la latencia se midió en el mismo equipo para los
13 modelos (CPU con 4 hilos; `src/costo_inferencia.py`).

```{include} tablas/e2_fichas.md
```

## Registro de experimentos

Cada ejecución queda guardada con su configuración, semilla y resultado:

| Etapa | Archivo |
|---|---|
| Ajuste previo de hiperparámetros | `resultados/e2/busqueda/{modelo}_hp.csv` y `{modelo}_hp_elegido.json` |
| Fase 1 (1 000 o 200 arquitecturas) | `resultados/e2/busqueda/{modelo}_fase1.csv` y `{modelo}_muestra_fase1.json` |
| Fase 2 (finalistas × 5 semillas) | `resultados/e2/busqueda/{modelo}_fase2.csv` |
| Validación con origen móvil | `resultados/e2/busqueda/{modelo}_origenes.csv` |
| Configuración final, fijada antes de abrir la prueba | `resultados/e2/finales/{modelo}_configuracion.json` (con fecha) |
| Corridas finales (4 orígenes × 10 semillas) | `resultados/e2/finales/{modelo}.csv` y predicciones por ventana `{modelo}/*.npz` |
| Equipo y versiones de cada etapa | `resultados/e2/busqueda/hardware.jsonl` |
| Análisis estadístico | `resultados/e2/estadistica/` y `resultados/e2/f2a/` |

## Entorno y reproducción

**Versiones.** Python 3.11.16. torch 2.14.1 con CUDA 12.6 en el PC y 13.0 en Colab. numpy 2.4.6, pandas 3.0.6, scipy
1.17.1, statsmodels 0.15.0 y scikit-learn 1.9.1 (`requirements.txt`). Chronos y Moirai usan `chronos-forecasting` y
`uni2ts` en sesiones aparte de Colab.

**Hardware.** PC del equipo con una NVIDIA RTX 4060 Ti de 8 GB, y Colab Pro+ con una NVIDIA A100 de 40 GB o una L4 de
24 GB. El equipo de cada etapa queda registrado.

**Determinismo.** Cada corrida fija su semilla en Python, numpy y torch, con cuDNN determinista y sin `benchmark`.
Las diferencias que quedan entre equipos vienen de operaciones de la GPU y son de milésimas (reproducción
independiente y validación con origen móvil).

**Un solo comando.** `python reproducir_e2.py` regenera todo el análisis, las tablas, las figuras y el libro a partir
de las predicciones guardadas. Lo probamos el 18 de noviembre: tardó unos 6 minutos en CPU y reprodujo todos los
archivos de resultados sin cambios, salvo las latencias, que son mediciones de tiempo. `python reproducir_e2.py --todo` además reentrena todo desde los datos, en el orden en
que se hizo. Ese modo incluye los dos pasos que se corren en Colab y se reanuda donde quedó. Con `--listar` se ven los
pasos sin correr nada.

## Lista de verificación de reproducibilidad (Apéndice B)

| Estado | Ítem | Evidencia |
|---|---|---|
| ✓ | Preregistro fechado y versionado antes del primer entrenamiento | `prerregistro_e2.md`, commit `983b32d` del 7-nov a las 22:44; el primer código de búsqueda es posterior (23:01) |
| ✓ | Hash del dataset y del conjunto de prueba | SHA-256 del panel, de las 215 286 ventanas de prueba y de `configs/e2.json` en el preregistro |
| ✓ | Auditoría de fuga ejecutada y aprobada | 9 de 9 pruebas (`tests/test_fuga.py`) y contrato de datos |
| ✓ | Particiones y semillas guardadas y regenerables | `configs/e2.json`, `resultados/e2/pliegues.csv`; semillas en cada CSV |
| ✓ | Un único pipeline por modelo, ningún ajuste fuera de train | `src/modelos_e2.py` (`Datos` escala con el entrenamiento de cada origen) |
| ✓ | Mismo presupuesto de ajuste; todas las configuraciones registradas | T = 20 para todos; misma búsqueda para los 9 modelos sin arquitectura publicada; registro de experimentos |
| ✓ | Al menos 6 ejecuciones por modelo y resultados por ejecución | 6 semillas declaradas (10 corridas) × 4 orígenes por modelo |
| ✓ | Análisis de potencia y justificación del número de ejecuciones | Piloto y potencia en el capítulo de resultados (diferencia mínima detectable `0.0045` m/s) |
| ✓ | Baselines trivial y clásico fuerte, y los tres modelos base del E1 con el mismo presupuesto | Tabla principal (plantilla A.2) |
| ✓ | Versiones de software, hardware y determinismo | Sección anterior y `hardware.jsonl` |
| ✓ | Pruebas de las Tablas A, B y C con p exactos, ajustados, IC y tamaños de efecto | Ficha de ruta y capítulo de resultados |
| ✓ | Familias y corrección múltiple declaradas a priori | Preregistro §6 (F1, F2 y F3 con Holm) |
| ✓ | Techo de desempeño y distancia γ | Capítulo de resultados (γ = `0.011`) |
| Parcial | Fase 1 con las 1 000 arquitecturas y k finalistas con al menos 5 ejecuciones | MLP, CNN, Chronos y Moirai evaluaron las 1 000; GRU, LSTM, TCN, KAN y Transformer una muestra de 200 (enmiendas 2 y 3), con su efecto medido. Fase 2 con 10 finalistas y 5 semillas |
| ✓ | Ganadora elegida solo con validación, con indistinguibles, sesgo del ganador y correlación entre fases | Análisis del espacio de arquitecturas, frente de Pareto, sesgo del ganador y validación con origen móvil |
| ✓ | Robustez con al menos tres niveles; interpretabilidad con fidelidad y estabilidad | Tres perturbaciones con tres niveles; gradientes integrados con prueba de borrado y estabilidad entre semillas |
| ✓ | Incertidumbre en los tres mejores con cobertura empírica | Intervalos conformales al 50, 80 y 90 % |
| ✓ | Un solo acceso a la prueba con modelo ya fijado | `{modelo}_configuracion.json` fechado antes de las corridas finales |
| ✓ | Replicación parcial del paper guía con criterio a priori y limitación | DLinear en ETTh1; limitación en RQ3 |
| ✓ | Reproducción independiente con discrepancia máxima | Discrepancia máxima de `1.3e-07` m/s en la tabla principal |
| Parcial | Código ejecutable de extremo a extremo con un solo comando | `reproducir_e2.py`; Chronos y Moirai exigen dos pasos manuales en Colab por las versiones de sus bibliotecas |
