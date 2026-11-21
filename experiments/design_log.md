# Bitácora de diseño del modelo original (Apéndice A.3 del enunciado del E3)

**Partición de desarrollo:** los orígenes de validación 2024-01-01, 2024-04-01 y 2024-07-01. **No** se usan para el
diseño el origen 2024-10-01 (reservado para el ajuste de hiperparámetros, igual que en los benchmarks), la prueba de
2025 ni los conjuntos adicionales.

**Base fija durante el diseño:** la arquitectura de la TCN elegida en el E2, w = (64, 64, 16) y b = (1, 1, 0), con sus
hiperparámetros (lr = 1.5e-3, wd = 0, lote 1 024); semilla 2001; 50 épocas como máximo con paciencia 5. Referencia: la
TCN elegida con la misma semilla en los mismos orígenes (`resultados/e2/busqueda/TCN_origenes.csv`).

| N.° | Variante (cambio respecto a la anterior) | Métrica de desarrollo (RMSE medio de 3 orígenes) | Decisión | Motivo y fecha |
|---|---|---|---|---|
| 0 | TCN del E2 sin componente propio (referencia) | se completa con `diseno.csv` | Referencia | Línea de partida, 20-nov-2026 |
| 1 | V1: modelo completo (K = 6, λ = 0.5, antisimétrica, pesos aprendidos) | pendiente | | Lanzada el 20-nov-2026, 23:4x |
| 2 | V2: K = 3 | pendiente | | |
| 3 | V3: K = 1 (solo el ancla de persistencia) | pendiente | | |
| 4 | V4: pesos uniformes | pendiente | | |
| 5 | V5: sin antisimetría | pendiente | | |
| 6 | V6: sin pérdida por ancla (λ = 0) | pendiente | | |

N_d hasta ahora: 6 variantes. Antes de esta tabla hubo una sola corrección de implementación, no de diseño: el sesgo
de la última capa de g se quitó porque se cancela en g(a, b) − g(b, a) y nunca recibía gradiente (lo detectó la
prueba unitaria de flujo de gradientes).
