# Bitácora de diseño del modelo original (Apéndice A.3 del enunciado del E3)

**Partición de desarrollo:** los orígenes de validación 2024-01-01, 2024-04-01 y 2024-07-01. **No** se usan para el
diseño el origen 2024-10-01 (reservado para el ajuste de hiperparámetros, igual que en los benchmarks), la prueba de
2025 ni los conjuntos adicionales.

**Base fija durante el diseño:** la arquitectura de la TCN elegida en el E2, w = (64, 64, 16) y b = (1, 1, 0), con sus
hiperparámetros (lr = 1.5e-3, wd = 0, lote 1 024); semilla 2001; 50 épocas como máximo con paciencia 5. Referencia: la
TCN elegida con la misma semilla en los mismos orígenes (`resultados/e2/busqueda/TCN_origenes.csv`).

| N.° | Variante (cambio respecto a la anterior) | RMSE de desarrollo: 2024-01 / 2024-04 / 2024-07 (media) | Decisión | Motivo y fecha |
|---|---|---|---|---|
| 0 | TCN del E2 sin componente propio (referencia) | 0.8025 / 0.8130 / 0.8537 (**0.8231**) | Referencia | Línea de partida |
| 1 | V1: modelo completo (K = 6, λ = 0.5, antisimétrica, pesos aprendidos) | 0.8140 / 0.8166 / 0.8587 (0.8298) | Se conserva como candidato | 20-nov, 23:16–23:40. Peor que la TCN en 0.007 m/s, dentro de δ |
| 2 | V2: K = 3 | 0.8050 / 0.8177 / 0.8563 (0.8263) | Pendiente de la ronda 2 | 20-nov |
| 3 | V3: K = 1 (solo el ancla de persistencia) | 0.7985 / 0.8135 / 0.8576 (0.8232) | Pendiente de la ronda 2 | 20-nov. Empata con la TCN |
| 4 | V4: pesos uniformes | 0.8166 / 0.8122 / 0.8619 (0.8302) | Pendiente de la ronda 2 | 20-nov |
| 5 | V5: sin antisimetría | 0.8179 / 0.8127 / 0.8631 (0.8312) | Pendiente de la ronda 2 | 20-nov. La peor junto con V6 |
| 6 | V6: sin pérdida por ancla (λ = 0) | 0.8125 / 0.8150 / 0.8681 (0.8319) | Pendiente de la ronda 2 | 20-nov |

**Lectura de la ronda 1 (una semilla, 21-nov).** Ninguna variante mejora el RMSE de la TCN; todas quedan dentro de
`0.009` m/s, del orden de la variación entre semillas (DE ≈ `0.004` m/s en el E2). Es lo que anticipaba el techo de
desempeño del E2. El RMSE de desarrollo, por sí solo, no permite elegir entre variantes. La hipótesis del mecanismo no
es sobre el RMSE medio sino sobre el sesgo en vientos fuertes y la calibración, y la ronda 1 no guardó las
predicciones. **Decisión:** repetir las mismas 6 variantes (ronda 2, sin cambiar su definición), guardando las
predicciones y la dispersión entre anclas, para medir en desarrollo el sesgo con viento ≥ 6 m/s y la relación entre
dispersión y error. Las repeticiones no suman variantes a N_d.

**Ronda 2 (21-nov 23:58 a 22-nov 00:41, en el PC; mismas 6 variantes + la TCN, semilla 2001).** Los RMSE salieron idénticos a
los de la ronda 1 en las 21 corridas, así que el entrenamiento es reproducible. Análisis en
`src/analisis_diseno_r2.py` (resultados en `resultados/e3/diseno_r2_analisis.csv`). Medias de los tres orígenes de
desarrollo; viento fuerte = velocidad observada ≥ 6 m/s (≈ 213 ventanas por origen).

| Variante | RMSE | Sesgo con viento fuerte (m/s) | ρ de Spearman entre dispersión y error absoluto |
|---|---|---|---|
| V0 TCN | 0.8231 | −2.19 | — |
| V1 completo | 0.8298 | −2.14 | 0.45 |
| V2 K = 3 | 0.8263 | −2.06 | 0.28 |
| V3 K = 1 | 0.8232 | −2.09 | — |
| V4 pesos uniformes | 0.8302 | −2.07 | 0.43 |
| V5 sin antisimetría | 0.8312 | −2.05 | 0.42 |
| V6 λ = 0 | 0.8319 | −2.11 | 0.48 |

* **Sesgo con viento fuerte: la hipótesis no se cumple.** Todas las variantes subestiman el viento fuerte en unos 2 m/s,
  igual que la TCN (la mejora es de 0.1 m/s como mucho). Se declara así en el libro.
* **Calibración: la dispersión entre anclas sí trae información** (ρ ≈ 0.45 con el error). Para compararla con justicia
  se usó el *interval score* de Winkler al 90 % (Gneiting y Raftery, 2007), que premia intervalos estrechos y castiga
  los que no cubren, con intervalos conformales calibrados por validación cruzada entre los tres orígenes (se calibra
  con dos y se evalúa en el tercero). El rival justo no es el intervalo de ancho fijo sino la TCN con intervalos
  normalizados por la desviación del viento en las últimas 24 h (std24), que no cuesta nada.

| Intervalo al 90 % (Winkler, menor es mejor: total / viento fuerte) | Ancho fijo | std24 | Dispersión | √(dispersión² + std24²) |
|---|---|---|---|---|
| V0 TCN | 3.926 / 21.8 | **3.460 / 15.8** (rival) | — | — |
| V1 completo | 3.948 / 21.5 | 3.481 / 15.8 | 3.778 / 14.6 | 3.391 / 15.6 |
| V2 K = 3 | 3.936 / 20.5 | 3.461 / 14.9 | 4.674 / 15.4 | 3.432 / 15.0 |
| V4 pesos uniformes | 3.940 / 20.6 | 3.474 / 15.0 | 3.922 / 13.9 | 3.404 / 14.9 |
| V5 sin antisimetría | 3.956 / 20.6 | 3.477 / 15.0 | 4.086 / 14.7 | 3.415 / 14.8 |
| V6 λ = 0 | 3.959 / 21.2 | 3.477 / 15.5 | 3.481 / 14.0 | **3.317** / 15.2 |

  Todas las coberturas quedan en 0.90 ± 0.01. Con la dispersión sola, el viento fuerte se cubre mejor (0.60 frente a
  0.46) pero a costa de intervalos mucho más anchos; la combinación √(dispersión² + std24²) es la que mejora el total.

**Decisión (22-nov): se congela el diseño.**
* **Modelo original final: V6** (K = 6, pesos aprendidos, diferencia antisimétrica, λ = 0) con intervalos conformales
  normalizados por √(dispersión² + std24²). En desarrollo: Winkler `3.317` frente a `3.460` de la TCN con std24
  (−4.1 %) y RMSE `0.8319` frente a `0.8231` (+0.009, dentro de δ = 0.02).
* **Criterio**, fijado aquí y que se mantiene en la prueba: métrica principal el Winkler al 90 %, con la condición de
  no empeorar el RMSE en más de δ (prueba TOST). El hallazgo alternativo queda reformulado solo sobre la calibración,
  porque el sesgo con viento fuerte no mejora.
* **Riesgo declarado:** la ventaja de V6 sobre V1 (0.074) sale de una sola semilla y puede ser ruido; la prueba con 10
  semillas lo dirá. La pérdida por ancla (λ) pasa a ser un brazo de la ablación (V1 frente a V6).
* **Ablación (≥ 3 componentes):** anclas (K = 6 frente a K = 1), pesos aprendidos frente a uniformes, antisimetría
  frente a diferencia libre, pérdida por ancla (λ = 0.5 frente a 0) y normalización del intervalo (combinada frente a
  std24).

N_d hasta ahora: 6 variantes de arquitectura más 3 formas de normalizar el intervalo (ancho fijo, dispersión y la
combinada; std24 es la del rival) = **9**. Antes de esta tabla hubo una sola corrección de implementación, no de diseño: el sesgo
de la última capa de g se quitó porque se cancela en g(a, b) − g(b, a) y nunca recibía gradiente (lo detectó la
prueba unitaria de flujo de gradientes).
