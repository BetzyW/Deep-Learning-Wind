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

N_d hasta ahora: 6 variantes. Antes de esta tabla hubo una sola corrección de implementación, no de diseño: el sesgo
de la última capa de g se quitó porque se cancela en g(a, b) − g(b, a) y nunca recibía gradiente (lo detectó la
prueba unitaria de flujo de gradientes).
