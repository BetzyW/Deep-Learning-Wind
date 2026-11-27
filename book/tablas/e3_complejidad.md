| Serie | L (h) | Ancho c | Parámetros | MFLOPs por ventana | µs por ventana (lote 1 024) |
|---|---|---|---|---|---|
| longitud L | 168 | 64 | `100 663` | `33.0` | `149` |
| longitud L | 240 | 64 | `100 663` | `47.1` | `208` |
| longitud L | 336 | 64 | `100 663` | `65.9` | `305` |
| longitud L | 504 | 64 | `100 663` | `98.9` | `457` |
| longitud L | 672 | 64 | `100 663` | `131.8` | `581` |
| ancho c | 168 | 16 | `7 219` | `2.2` | `27` |
| ancho c | 168 | 32 | `26 367` | `8.4` | `62` |
| ancho c | 168 | 64 | `100 663` | `33.0` | `158` |
| ancho c | 168 | 128 | `393 255` | `130.5` | `444` |

| Variable | Pendiente: parámetros | Pendiente: FLOPs | Pendiente: tiempo |
|---|---|---|---|
| L | `0.00` | `1.00` | `1.00` |
| c | `1.92` | `1.96` | `1.35` |
