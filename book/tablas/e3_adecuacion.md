| Modelo | RMSE entrenamiento | RMSE parada (validación) | RMSE prueba | Brecha parada − entrenamiento (m/s) | gap_ratio mediano / máximo | Mincer–Zarnowitz (HAC) |
|---|---|---|---|---|---|---|
| PADRE | `0.7238` | `0.7488` | `0.7239` | `+0.0250` | `0.023` / `0.108` | a = `-0.003`, b = `1.002`, p = `0.916` |
| TCN | `0.7477` | `0.7451` | `0.7206` | `-0.0027` | `0.034` / `0.071` | a = `-0.007`, b = `1.008`, p = `0.171` |

| Prueba sobre los residuos (por estación, 61 estaciones) | PADRE | TCN |
|---|---|---|
| Ljung–Box(24): autocorrelación | `100 %` | `100 %` |
| ARCH-LM(24): heterocedasticidad | `98 %` | `98 %` |
| Jarque–Bera: normalidad | `92 %` | `92 %` |
| BDS: independencia no lineal | `100 %` | `100 %` |
| Mincer–Zarnowitz: a = 0, b = 1 | `56 %` | `44 %` |
