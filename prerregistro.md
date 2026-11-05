# Prerregistro del experimento

Escrito el 3 de noviembre de 2026, después del piloto y **antes** del ajuste de hiperparámetros y de evaluar el
conjunto de prueba (2025).

| Elemento | Valor |
|---|---|
| Modelos | MLP, LSTM y CNN 1D (`src/modelos.py`) |
| Baseline | Persistencia a 24 h: $\hat v_{t+24} = v_t$ |
| Métrica primaria | RMSE en 2025 (m/s) |
| Métricas secundarias | MAE, $R^2$, MASE respecto a la persistencia |
| Margen de relevancia | $\delta = 0.02$ m/s |
| Unidad de independencia | La estación |
| Piloto | 5 semillas por modelo en validación; $\sigma_d$ máxima entre pares = 0.0038 m/s |
| Número de semillas | La fórmula da $R = 1$; se usan **$R = 6$** (mínimo exigido) |
| Ajuste | Búsqueda aleatoria, 20 configuraciones por modelo, semilla 7, 30 épocas máximo, paciencia 3 |
| Corridas definitivas | Semillas 1000–1005, 50 épocas máximo, paciencia 5, parada temprana con validación 2024 |
| Pruebas | Friedman (estaciones como bloques), Wilcoxon con Holm, Diebold–Mariano con HLN, TOST con $\pm\delta$, Ljung-Box y BDS en residuos |
| Nivel de significancia | $\alpha = 0.05$ |
| Huella del conjunto de prueba | SHA-256 `ad841c4f14e808cf9002821c77149f05eb927264ea717a9a87e7cb5bb60940be` (estación, hora objetivo y valor de las 211 408 ventanas de 2025) |

*Nota del 4 de noviembre de 2026:* la huella del conjunto de prueba se agregó después de las corridas definitivas. Se calculó con el mismo preprocesamiento que generó las predicciones evaluadas, de modo que certifica el conjunto usado; el resto del prerregistro no se modificó.
