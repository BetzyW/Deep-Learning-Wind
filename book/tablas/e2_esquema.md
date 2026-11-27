| Columna | Tipo | Unidad | Rango permitido | Fuente |
|---|---|---|---|---|
| `CodigoEstacion` | texto de 10 dígitos | — | código IDEAM | IDEAM |
| `hora` | fecha-hora (UTC−5) | h | 2020-01-01 00:00 a 2025-12-31 23:00, en punto | IDEAM, agregada a la hora |
| `vel` | real | m/s | [0.0, 40.0] | IDEAM: velocidad del viento (media horaria) |
| `dir_sin` | real | — | [-1.0, 1.0] | IDEAM: dirección, media vectorial horaria |
| `dir_cos` | real | — | [-1.0, 1.0] | IDEAM: dirección, media vectorial horaria |
| `temp` | real | °C | [-5.0, 50.0] | IDEAM: temperatura del aire (media horaria) |
| `lat` | real | grados | [7.0, 13.0] | IDEAM: catálogo de estaciones |
| `lon` | real | grados | [-77.0, -71.0] | IDEAM: catálogo de estaciones |
