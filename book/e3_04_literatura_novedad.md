# Revisión de literatura y novedad

Este capítulo documenta la evidencia de que PADRE-TCN es nuevo para el pronóstico de la velocidad del viento
(condición R1), siguiendo el protocolo de la sección 6.3 del enunciado: tres bases, al menos cinco cadenas, matriz de
novedad, búsqueda por citación, revisión de repositorios de código, declaración de novedad y clasificación de la
diferencia con cada trabajo cercano. La revisión sistemática del Top 10 está en el capítulo de revisión del E2.

## Protocolo de búsqueda

* **Bases.** OpenAlex (índice abierto que cubre buena parte de Scopus y Web of Science), Semantic Scholar y arXiv,
  consultadas por su API con `literatura/e3/buscar_novedad.py`. Scopus y Web of Science exigen credenciales
  institucionales; en ellas usamos la cadena transversal T5 del E2 (redes siamesas, gemelas y de diferencias en
  regresión), consultada el 5 de noviembre.
* **Nueve cadenas** que combinan (i) el mecanismo, (ii) el problema y (iii) sinónimos en inglés y en español, más
  cadenas del mecanismo en otras aplicaciones para identificar su origen: red gemela y viento (C1), regresión por
  diferencias y viento (C2), red siamesa y viento (C3), análogos con redes neuronales y viento (C4), redes gemelas en
  series de tiempo (C5), redes gemelas de regresión en cualquier aplicación (C6), PADRE (C7), anclas por análogos con
  aprendizaje profundo (C8) y una cadena en español (C9).
* **Fechas.** La búsqueda se hizo el 20 de noviembre de 2026, el mismo día en que se eligió el mecanismo, y se **repitió el 27 de
  noviembre**, antes de la entrega, como pide el enunciado. Los registros de cada fecha se conservan por separado
  (`literatura/e3/busquedas_{fecha}.csv` y `resultados_{fecha}.csv`).
* **Limitación.** Semantic Scholar limitó la tasa de consultas (error 429) en 6 de las 9 cadenas, incluso con
  reintentos espaciados; respondió en C3, C8 y C9. Las otras dos bases cubrieron las 9 cadenas.

## Matriz de novedad (27 de noviembre de 2026)

```{include} tablas/e3_matriz_novedad.md
```

*Resultados: total que reporta la base. Revisados: títulos leídos (hasta 50 por cadena; se leyó el resumen de los que
combinan el mecanismo con el viento). Relevantes: trabajos que hay que contrastar en la declaración de novedad.*

## Búsqueda por citación y repositorios

**Búsqueda por citación** (`literatura/e3/snowball_repos.py`, OpenAlex, 27-nov). Desde cinco trabajos: las tres fuentes
del mecanismo (PADRE, TNNR y el ensamble de análogos), el paper guía (DLinear) y el trabajo más cercano encontrado
(la red pseudo-gemela de Yang et al.). Revisamos sus referencias (229 en total) y los trabajos que los citan (hasta 200
por trabajo; 511 revisados). Encontramos 14 títulos que combinan el mecanismo con el viento, todos citantes del
ensamble de análogos: aplicaciones del ensamble de análogos clásico (vecinos más cercanos en el espacio de un modelo
numérico del tiempo) al pronóstico de viento y de potencia eólica, y la destilación de un ensamble de análogos en una
red profunda {cite}`candido2020`. **Ninguno entrena una red que prediga diferencias contra anclas.** Ni PADRE ni TNNR
tienen citantes que los apliquen al viento.

**Repositorios de código** (GitHub y Hugging Face Hub, 27-nov, 5 consultas). Encontramos la implementación pública de
TNNR de sus autores (`sjwetzel/PublicTwinNeuralNetworkRegression`), que **no consultamos ni usamos**: el módulo se
escribió desde las ecuaciones (capítulo de implementación), y una implementación genérica de aprendizaje por
diferencias entre pares (`JacksonBurns/neural-pairwise-regression`). Ninguna está aplicada al viento ni a series de
tiempo, y no hay modelos del mecanismo en Hugging Face Hub. Papers with Code dejó de funcionar como servicio
independiente en 2025, así que no se pudo consultar.

## Trabajos cercanos y clasificación de la diferencia

| Trabajo | Qué hace | Diferencia con PADRE-TCN | Clase de la diferencia |
|---|---|---|---|
| PADRE {cite}`tynes2021` (química) | Bosques aleatorios sobre pares de moléculas que predicen la diferencia de la propiedad; promedio desde puntos de entrenamiento | Fuente del mecanismo. Pares contra el conjunto de entrenamiento, sin red, sin series de tiempo ni anclas temporales | Mecanismo (fuente transferida) |
| TNNR {cite}`wetzel2022` (física y química) | Red gemela que predice yᵢ − yⱼ; promedio uniforme desde puntos de entrenamiento | Fuente del mecanismo. Antisimetría aproximada (entrenando con los dos órdenes), una red por par, promedio uniforme | Mecanismo (fuente transferida) |
| Ensamble de análogos {cite}`dellemonache2013` (meteorología) | Busca situaciones pasadas parecidas en un modelo numérico y usa lo que pasó después | Sin red ni diferencias aprendidas; los análogos se eligen por distancia en variables del modelo numérico | Mecanismo |
| Destilación de un ensamble de análogos {cite}`candido2020` | Entrena una red profunda para imitar la salida de un ensamble de análogos | La red aprende el pronóstico del ensamble, no diferencias contra anclas; sin antisimetría ni pesos por similitud | Mecanismo |
| Red pseudo-gemela para potencia eólica {cite}`yang2025` | Dos ramas de MLP extraen rasgos de las variables físicas y del historial de potencia, y luego se mezclan | "Gemela" se refiere a dos ramas en paralelo, no a comparar pares; no predice diferencias, no tiene anclas ni antisimetría | Mecanismo |
| DLinear, paper guía {cite}`zeng2023` | Descompone la serie en tendencia y estacionalidad y aplica capas lineales | Predice el valor absoluto; ninguna noción de pares ni de análogos | Mecanismo |

## Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita (tres bases abiertas con nueve cadenas, repetida el 20 y
el 27 de noviembre de 2026; la cadena transversal del E2 en Scopus y Web of Science; búsqueda por citación desde cinco
trabajos; y revisión de GitHub y Hugging Face Hub), **no se encontraron estudios que apliquen la regresión por
diferencias entre pares con redes neuronales al pronóstico de la velocidad del viento**, ni modelos que combinen anclas
temporales causales dentro de la ventana de entrada con una diferencia antisimétrica por construcción. Los trabajos más
cercanos son la red pseudo-gemela para potencia eólica {cite}`yang2025`, que usa "gemela" en otro sentido, y la
destilación de un ensamble de análogos {cite}`candido2020`, que no aprende diferencias. El mecanismo viene de la
química (PADRE) y de la física (TNNR), y lo adaptamos con las seis modificaciones algorítmicas del capítulo de diseño;
la diferencia con cada trabajo cercano es de **mecanismo**, la clase que sustenta R1.

**Relación con el paper guía.** DLinear y todos los modelos del Top 10 predicen el valor absoluto y subestiman los
vientos fuertes. PADRE-TCN cambia lo que se predice (una corrección sobre situaciones análogas ya observadas) y agrega
una medida de incertidumbre propia; no corrigió la subestimación, pero sí mejoró los intervalos.

## Verificación por un tercero

El enunciado pide que una persona ajena al diseño repita al menos dos cadenas y confirme o refute la declaración. Esta
verificación la hace **Yuryen Rollo**, que acompañó el proyecto pero no programó el mecanismo; se declara así, sin
presentarlo como un tercero externo. *Pendiente: su concepto, con las cadenas repetidas, la fecha y los resultados, se
anexa aquí antes de la entrega del 2 de diciembre.*
