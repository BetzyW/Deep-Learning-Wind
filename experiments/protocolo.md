# Protocolo experimental pre-registrado del Entregable 3 (sección 8.1 del enunciado)

**Estado: borrador para revisión (24-nov-2026).** Se congela con el tag de Git `e3-protocolo` antes de correr cualquier
corrida confirmatoria del modelo original en la prueba de 2025. Después del tag, cualquier cambio se declara como
desviación en el informe o reinicia el protocolo con un tag nuevo (sección 7.2).

## 1. Estándar de profundidad

Nos comprometemos con el estándar de `experiments/estandar.md` (tag `e3-estandar`, puesto antes que este):
demostración formal de una propiedad con verificación numérica, tres conjuntos de datos adicionales elegidos antes de
ver resultados, familia completa de contrastes con Holm, tres pruebas por contraste, potencia por contraste de al
menos 0.90 con potencia conjunta reportada y al menos 10 semillas.

## 2. Modelo original y fuente del mecanismo

* **Modelo:** PADRE-TCN, diseño **V6** congelado el 22-nov-2026 (`experiments/design_log.md`, commit `fc3b1a3`): codificador TCN del
  E2 (w = (64, 64, 16), b = (1, 1, 0)), K = 6 anclas diarias dentro de la misma ventana de 168 h, diferencia
  antisimétrica D(a, b) = g(a, b) − g(b, a), pesos aprendidos y sin pérdida por ancla (λ = 0). Código:
  `src/original_model.py`.
* **Intervalo de predicción:** conformal al 90 %, con el error normalizado por s = √(dispersión² + std24²), donde la
  dispersión es la desviación ponderada entre anclas y std24 la desviación del viento en las últimas 24 h.
* **Fuente:** transferido y adaptado de la regresión por diferencias entre pares (PADRE; Tynes et al., 2021), de la
  regresión con redes gemelas (Wetzel et al., 2022) y del ensamble de análogos (Delle Monache et al., 2013). La
  justificación y las cinco modificaciones están en `experiments/procedencia.md`. La modificación M5 (pérdida por ancla)
  quedó fuera del diseño final (λ = 0) y se evalúa en la ablación.

## 3. Hipótesis

**Confirmatorias.**

* **H1 (superioridad, condición R3).** PADRE-TCN tiene menor RMSE que cada uno de los 12 modelos de la familia
  (sección 6), con una diferencia mayor que δ. Se acepta solo si se cumplen las seis condiciones de la sección 8.4 del
  enunciado (sección 8 de este protocolo).
* **H2 (hallazgo alternativo, sección 14.5.9).** Con el mismo método de intervalo para todos, los intervalos al 90 % de
  PADRE-TCN tienen menor *interval score* de Winkler que los de cada modelo de la familia, y su RMSE no es peor que el de
  la TCN del E2 en más de δ.
  * **H2a.** Winkler de PADRE-TCN, con s = √(dispersión² + std24²), frente al Winkler de cada rival con s = std24, que
    es el mismo método sin la dispersión entre anclas. Son 12 contrastes con Holm.
  * **H2b.** Equivalencia del RMSE frente a la TCN por TOST, con margen ±δ.

**Hipótesis del mecanismo y predicciones secundarias (exploratorias).**

1. La dispersión entre anclas anticipa el error: ρ de Spearman entre dispersión y |error| mayor que 0 en la prueba (en
   desarrollo fue `0.48`).
2. La cobertura empírica del intervalo al 90 % queda en 0.90 ± 0.02, en total y por trimestre.
3. Sesgo con viento ≥ 6 m/s: en desarrollo no mejoró (≈ −2 m/s en todas las variantes). Se reporta sin pretender
   mejora.

Cualquier análisis no listado aquí se reporta como exploratorio y no sustenta R3.

## 4. Métricas y margen

* **Primaria:** RMSE del pronóstico a 24 h, calculado por estación y promediado (heredada del E1).
* **Margen de relevancia práctica:** δ = `0.02` m/s (E1).
* **Secundarias:** MAE, sesgo con viento ≥ 6 m/s, Winkler al 90 % (total y con viento fuerte), cobertura y ancho del
  intervalo.
* **Intervalos de todos los modelos:** split conformal entre orígenes. Para cada origen de prueba, el cuantil se calibra
  con los residuos normalizados de los otros tres orígenes, con la misma semilla.

## 5. Datos, particiones, semillas y determinismo

* **Panel principal:** el del E2 (`configs/e2.json`): 62 estaciones del IDEAM en el Caribe, ventana de 168 h, horizonte
  de 24 h, ventana de entrenamiento expansiva y embargo de 24 h.
* **Desarrollo (diseño):** orígenes de validación 2024-01, 2024-04 y 2024-07. Allí se decidió todo el diseño
  (N_d = **9**).
* **Ajuste de hiperparámetros:** origen 2024-10, el mismo que usaron los benchmarks.
* **Confirmación:** los 4 orígenes de prueba de 2025 (2025-01, 2025-04, 2025-07 y 2025-10), reentrenando desde cero en
  cada uno. PADRE-TCN nunca se ha evaluado en ellos. Los benchmarks sí se evaluaron allí en el E2 y quedan congelados
  con esas corridas.
* **Semillas:** 1000–1009, las mismas de los benchmarks, para tener pares estrictos por semilla y origen.
* **Determinismo:** `fijar_semilla` fija las semillas de Python, NumPy y PyTorch, con `cudnn.deterministic = True` y
  `cudnn.benchmark = False`. La ronda 2 de diseño reprodujo exactamente los RMSE de la ronda 1 en 21 corridas.
* **Hardware:** PC con NVIDIA GeForce RTX 4060 Ti, torch 2.14.1 y CUDA 12.6. Se registra cada corrida en
  `resultados/e2/busqueda/hardware.jsonl`.

## 6. Familia de contrastes (m = 12)

* **Top 10:** Transformer, KAN, TCN, STGCN, Chronos, GRU, Informer, DLinear, Moirai y PatchTST.
* **Baselines fuertes:** Ridge y gradient boosting.
* MLP, CNN, LSTM y las líneas base triviales se reportan, pero no entran en la familia.
* No hay exclusiones.

## 7. Pruebas, corrección y nivel

* **Unidad de independencia:** la estación para las pruebas por rangos; el tiempo para las pruebas sobre series de
  pérdida.
* **Pruebas por contraste (cada una con Holm sobre los 12):**
  1. **Wilcoxon** de rangos con signo sobre la diferencia de RMSE por estación (62 pares, promediando semillas y
     orígenes).
  2. **Diebold–Mariano con corrección HLN** sobre la serie horaria de pérdida cuadrática media entre estaciones y
     semillas, con varianza HAC de 23 rezagos.
  3. **Bootstrap estacionario pareado** (Politis y Romano, 1994) de la diferencia de pérdida media, con bloques de
     media 168 h, 10 000 réplicas e intervalo al 95 %.
* **Decisión por contraste:** el máximo de los tres p ajustados debe ser menor que α = 0.05.
* **Procedimiento global:** Model Confidence Set al 90 % (nivel 0.10).
* **Tamaño del efecto:** δ de Cliff por estación (|δ| ≥ 0.147) y d de Cohen sobre las semillas.
* **Robustez (condición 6):** t pareado sobre las 10 semillas, con y sin la corrección de Nadeau–Bengio, y bootstrap
  por bloques de semana como alternativa al estacionario.
* **H2a:** las mismas tres pruebas, con el Winkler en lugar de la pérdida cuadrática.
* **H2b:** TOST con dos pruebas t unilaterales sobre las diferencias de RMSE pareadas por semilla y origen (40 pares),
  margen ±δ y α = 0.05.
* **Supuestos:** se verifican y reportan la estacionariedad de la serie de diferencias de pérdida (ADF) y la
  autocorrelación de los residuos (Ljung–Box).
* **Familias declaradas aparte, cada una con su propio Holm:** ablación (sección 11), conjuntos adicionales (sección
  10) y H2a. Se reportan siempre el p crudo, el p ajustado y el número total de comparaciones.

## 8. Regla de decisión para R3 (sección 8.4)

PADRE-TCN es superior solo si se cumplen las seis condiciones a la vez:

1. Su RMSE medio es menor que el de los 12 rivales.
2. Las tres pruebas rechazan en los 12 contrastes, con una diferencia mayor que δ.
3. El efecto no es despreciable y el límite inferior del IC 95 % de la diferencia es mayor que 0.
4. Es mejor en al menos 2 de los 3 conjuntos adicionales y no es significativamente peor que ningún rival en ninguno.
5. Es adecuado: supera a la persistencia, la brecha entre entrenamiento y validación es menor que δ, la cobertura y los
   residuos son aceptables, y validación y prueba son consistentes.
6. La conclusión no cambia con las pruebas alternativas.

Si falla cualquiera, el tope es 4.0 y el trabajo se sostiene en H2.

## 9. Potencia y número de semillas

* **Fórmula:** R ≈ (z₁₋α′/₂ + z₁₋β)² σ_d² / δ², con α′ = 0.05/12 (Bonferroni, más conservador que Holm) y potencia
  0.90. Se obtiene (2.865 + 1.282)² = `17.2`.
* **σ_d:** se toma de la dispersión entre semillas de las corridas finales del E2, la media de los 4 orígenes por
  semilla. Supuesto: PADRE-TCN varía como la TCN (DE `0.0026`). Con el rival que más varía (Chronos, DE `0.0082`),
  σ_d = √(0.0026² + 0.0082²) = `0.0086` m/s. En el caso típico (DE ≈ `0.002`), σ_d ≈ `0.0037`.
* **Resultado:** para detectar una diferencia igual a δ se necesitan R ≈ 17.2 × (0.0086 / 0.02)² = `3.2` semillas en
  el peor caso. Con las **10 semillas**, la diferencia mínima detectable con potencia 0.90 es `0.011` m/s en el peor
  caso y `0.005` m/s en el típico, ambas menores que δ.
* **Potencia conjunta** frente a un efecto igual a δ en los 12 contrastes: mayor que 0.99. Con potencia exactamente
  0.90 por contraste sería 0.90¹² ≈ 0.28. La reportamos y la discutimos.
* **Limitación:** la potencia se calcula para el efecto δ. Si el efecto real es menor que δ, no habrá significancia, y
  la falta de significancia **no se interpreta como equivalencia** (para eso está H2b).
* No hace falta aumentar las semillas.

## 10. Conjuntos de datos adicionales

**Elección.** Los tres conjuntos son ETTh1, SDWPF y NOAA ISD-Lite Kansas. Se eligieron el 21-nov-2026, antes de
entrenar cualquier modelo en ellos, con su fuente, licencia, versión y SHA-256 (`experiments/conjuntos_adicionales.md`,
`experiments/e3_sha256.csv`, commit `5872019`). Cada uno difiere del panel en al menos dos ejes: dominio, tamaño,
número de series y variable objetivo.

**Protocolo y recortes.** Aplicamos el recorte por cómputo declarado antes de entrenar:

* 2 orígenes de prueba por conjunto;
* la configuración del E2 sin reajustar (PADRE-TCN usa la suya, la de este protocolo);
* 10 semillas;
* Chronos y Moirai solo sin entrenar;
* Informer con 3 semillas en SDWPF y NOAA (recorte declarado el 23-nov).

**Pruebas.**

* **Por conjunto:** Diebold–Mariano con HLN sobre la pérdida horaria media entre series y semillas, con Holm sobre los
  12 contrastes de cada conjunto.
* **Entre conjuntos:** Friedman con la corrección de Iman–Davenport sobre los rangos medios del RMSE por conjunto, y
  comparaciones de PADRE-TCN contra cada rival con Holm. No se promedian métricas de escalas distintas (SDWPF está en
  kW).
* La conclusión se apoya también en la consistencia por conjunto, porque con tres conjuntos la prueba global tiene
  poca potencia.

**Declaración.** Los modelos ligeros, incluido PADRE-TCN, ya se corrieron en estos conjuntos el 22 y 23 de noviembre,
con el diseño y los hiperparámetros ya congelados y antes de este tag. Esas corridas no cambiaron ninguna decisión del
modelo original. Sí motivaron una corrección de implementación de DLinear: su salida no se desnormalizaba como en el
artículo. Se declaró en `experiments/conjuntos_adicionales.md` y se repitieron sus corridas.

## 11. Ablación (familia propia, 5 contrastes con Holm)

V6 completo frente a cada variante, con 10 semillas, en los 4 orígenes de prueba y con las mismas pruebas que H1:

| Componente | Variante |
|---|---|
| Anclas | V3, K = 1 |
| Pesos | V4, uniformes |
| Antisimetría | V5, diferencia libre |
| Pérdida por ancla | V1, λ = 0.5 |
| Normalización del intervalo | Combinada frente a std24 (no reentrena; se compara en Winkler) |

## 12. Presupuesto y equidad

* **Medida del presupuesto:** número de configuraciones, y también se reporta el tiempo de cómputo.
* **Hiperparámetros:** PADRE-TCN recibió T = 20 configuraciones del mismo espacio y con la misma lista que los
  benchmarks (`src/ajuste_padre.py`, `resultados/e3/PADRE_hp.csv`): 20 épocas, paciencia 3, 10 % de las ventanas y
  origen 2024-10.
* **Arquitectura:** no tuvo búsqueda; hereda la de la TCN elegida en el E2.
* **Diseño:** N_d = 9 variantes en desarrollo, frente a ninguna en los benchmarks. Se declara y se discute esa
  asimetría.
* **Corridas finales:** 50 épocas como máximo, paciencia 5 y las mismas entradas, particiones y preprocesamiento que
  los benchmarks.
* **Se reportan:** parámetros, FLOPs, tiempo de entrenamiento e inferencia, memoria pico y horas de GPU de cada modelo.

## 13. Calendario

| Paso | Cuándo | Dónde |
|---|---|---|
| Tags `e3-estandar` y `e3-protocolo` | 24-nov | — |
| Corridas confirmatorias de PADRE-TCN (40 corridas, ≈ 2 h) | Noche del 24-nov | PC |
| Ablación (160 corridas, ≈ 8 h) | Noches del 24 y 25-nov | PC |
| Modelos pesados que faltan en los conjuntos adicionales, y Chronos y Moirai sin entrenar | Cuando haya una L4 | Colab |
| Análisis estadístico | 27–29 nov | — |
