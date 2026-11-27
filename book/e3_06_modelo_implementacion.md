# Modelo original: implementación

El capítulo anterior presentó el diseño de PADRE-TCN. Este muestra cómo lo programamos y cómo lo comprobamos: el código
comentado, la correspondencia entre cada modificación algorítmica y su línea de código, las pruebas unitarias, la
ablación con las mismas pruebas estadísticas del capítulo de validación y la interpretabilidad comparada con la del
mejor benchmark.

## Código

El módulo es `src/original_model.py`. Lo escribimos desde las ecuaciones del capítulo de diseño, sin código de
terceros: solo usa capas primitivas de PyTorch (lineales y convolucionales) y la TCN propia del E2, de la que toma los
tres bloques causales como codificador. Las fuentes del mecanismo están citadas en la cabecera del módulo.

```{literalinclude} ../src/original_model.py
:language: python
:linenos:
```

**Integración al mismo pipeline.** PADRE-TCN se entrena con la misma función `entrenar` de `src/modelos_e2.py` que
todos los modelos del benchmark: el mismo optimizador (Adam), el mismo lazo, la misma parada temprana y el mismo
`Datos`, que escala la entrada y arma las ventanas por índice. La única diferencia es que `entrenar` le pasa la media y
la desviación de la velocidad (`mu_y`, `sd_y`), igual que a PatchTST y a DLinear, porque el modelo necesita llevar los
objetivos de las anclas a m/s. Si existe `perdida_aux`, la suma a la pérdida; en el diseño final (λ = 0) vale 0.

## Adaptación algorítmica frente a la fuente

| Modificación | Qué hace la fuente | Dónde está en el código |
|---|---|---|
| M1. Anclas temporales causales | TNNR y PADRE comparan con puntos del conjunto de entrenamiento | `idx_ancla` e `idx_obj` en `PadreTCN.__init__`: índices L − 1 − 24 j de las representaciones y L − 1 − 24 (j − 1) de los objetivos, dentro de la misma ventana; `componentes` lee `y_a` de la columna de velocidad desescalada |
| M2. Codificador causal compartido | Una evaluación de la red por cada par | `self.codificador = tcn.bloques`: una sola pasada da `h` para todas las horas, y la consulta y las anclas son filas de `h` |
| M3. Antisimetría exacta | Se aproxima entrenando con los dos órdenes | `DiferenciaAntisimetrica.forward`: `g(φ(a, b)) − g(φ(b, a))`; la última capa de `g` va sin sesgo porque se cancelaría |
| M4. Fusión aprendida | Promedio uniforme | `self.P`, `self.log_T` y `self.beta`; softmax de la similitud proyectada más el sesgo por día |
| M5. Pérdida por ancla | Error de cada par | `perdida_aux`, con los pesos sin gradiente (`w.detach()`); en V6 λ = 0 |
| M6. Incertidumbre | Dispersión entre pares | `dispersion`: √Σⱼ wⱼ (ŷⱼ − ŷ)²; el intervalo conformal se arma fuera del modelo, en `src/estadistica_e3.py` |

Las banderas `antisimetrica` (`DiferenciaLibre` en lugar de `DiferenciaAntisimetrica`), `pesos` ("uniformes") y `lam`,
junto con K, permiten construir cada variante de la ablación con el mismo código.

## Pruebas unitarias

`tests/test_original_model.py` cubre los seis puntos que exige la sección 6.5 del enunciado. Se corren con
`pytest tests/`, junto con las pruebas de fuga y de partición del E2 (19 en total, todas pasan).

| Requisito del enunciado | Prueba | Qué comprueba |
|---|---|---|
| (i) Sobreajusta un lote diminuto | `test_sobreajusta_lote_pequeno` | Con 32 ventanas sintéticas cuyo objetivo depende del nivel actual y del de ayer, 300 pasos de Adam bajan el MSE por debajo de `0.05` |
| (ii) Reproducibilidad | `test_reproducible_con_misma_semilla` | Dos modelos con la misma semilla dan salidas idénticas |
| (iii) Formas y entradas inválidas | `test_formas_y_entradas_invalidas` | Formas de ŷ, ŷⱼ y w; los pesos suman 1; error claro si la ventana no tiene 168 h, si K no cabe en la ventana o si cambia el número de variables |
| (iv) Flujo de gradientes | `test_todos_los_parametros_reciben_gradiente` | Todos los parámetros reciben un gradiente finito y distinto de 0 (esta prueba detectó el sesgo inútil de la última capa de g) |
| (v) Propiedad demostrada | `test_antisimetria_exacta`, `test_pronostico_entre_anclas` y `test_gradcheck_de_la_diferencia` | Proposición 1 en float64 (error < 10⁻¹²), Proposición 2 y gradientes analíticos de D iguales a los numéricos (`gradcheck`) |
| (vi) Guardar, cargar y exportar | `test_guardar_cargar_y_exportar` | Las salidas no cambian al guardar y cargar los pesos, ni al exportar a ONNX y correr con onnxruntime (diferencia < 10⁻⁴) |

Además, la exportación del modelo entrenado (capítulo de benchmarks) repitió la prueba de ida y vuelta con las 84 998
ventanas reales de un origen: la diferencia máxima entre ONNX y PyTorch fue de `2.4e-6` m/s.

**Reproducibilidad en la práctica.** Para la interpretabilidad reentrenamos PADRE-TCN con las semillas 1001 y 1002 en
el origen 2025-10-01, tres días después de las corridas confirmatorias. Las predicciones coincidieron exactamente con
las guardadas (diferencia máxima `0`), igual que en la bitácora de diseño, donde la segunda ronda reprodujo los 21 RMSE
de la primera.

## Ablación

La ablación se fijó en el protocolo antes del tag: cada componente propio se retira por separado, con todo lo demás
fijo, los mismos hiperparámetros, las 10 semillas y los 4 orígenes de prueba. Las variantes se comparan con el modelo
completo (V6) con las mismas tres pruebas del capítulo de validación y con Holm dentro de la familia de la ablación (5
contrastes), separada de la familia principal.

```{include} tablas/e3_ablacion.md
```

*Δ = V6 − variante; para el RMSE y el Winkler, un Δ negativo favorece al modelo completo. "Pérdida por ancla" es el
único componente que se **agrega** (V6 tiene λ = 0 y la variante V1 usa λ = 0.5), porque quedó fuera del diseño
congelado.*

**Lectura.**

* **Ninguno de los cuatro componentes del pronóstico puntual cambia el RMSE de forma detectable.** Las diferencias son
  de `0.0003` a `0.0022` m/s, diez veces menos que δ, y ninguna rechaza con las tres pruebas. Es lo que anticipaba el
  techo de desempeño del E2: si todos los modelos entrenados quedan a `0.011` m/s entre sí, retirar una pieza de uno de
  ellos no puede mover mucho más. El caso más claro es K = 1 (solo el ancla de la persistencia), que da un RMSE incluso
  `0.002` m/s menor, sin significancia: para el pronóstico puntual, las anclas de días anteriores no aportan sobre la
  TCN.
* **La normalización con la dispersión entre anclas sí contribuye, y es la que explica H2.** Con el mismo modelo, pero
  el intervalo normalizado solo con std24, el Winkler sube de `2.991` a `3.177` (`0.186` m/s, 6 %), con p < 0.001 en
  las tres pruebas. Es la predicción secundaria 3 del protocolo: sin la dispersión, la ventaja en los intervalos
  desaparece y PADRE-TCN queda al nivel de los rivales. El componente que aporta es M4 + M6, la dispersión ponderada
  entre los pronósticos de las anclas.
* **Capacidad controlada.** Todas las variantes tienen entre `100 658` y `100 664` parámetros, así que ninguna
  diferencia se puede atribuir al tamaño.

**Lo que la ablación no cubre.** El enunciado sugiere además agregar cada componente por separado al modelo base y,
si se sospechan interacciones, un diseño factorial fraccionado. No los corrimos: cada variante cuesta 40 corridas y el
presupuesto se priorizó para los conjuntos adicionales. El modelo base sin ningún componente propio sí está en la
comparación: es la TCN del E2, equivalente a PADRE-TCN en RMSE (TOST, capítulo de validación) y con un Winkler
`0.172` m/s peor. Como ningún componente del pronóstico puntual tuvo efecto por separado, no hay indicios de una
interacción que cambie la conclusión, pero tampoco la descartamos formalmente. La ablación apoya que la dispersión
contribuye; el *porqué* lo sustenta la predicción secundaria 1 (la dispersión se correlaciona con el error, ρ =
`0.43`).

## Interpretabilidad

Aplicamos al modelo original la misma técnica que a los benchmarks en el E2: **gradientes integrados** con línea base
0 (la media del escalado) y 32 pasos, sobre las mismas 2 000 ventanas de prueba del origen 2025-10-01 (elegidas con la
semilla 0) y con las semillas 1000, 1001 y 1002 (`src/interpretabilidad_padre.py`). Así, la comparación con el
Transformer (el mejor benchmark por RMSE) y con la TCN (su base) es directa. La **fidelidad** se mide borrando el 10 %
de las horas con mayor atribución y comparándolo con borrar el 10 % al azar, y la **estabilidad**, con la correlación de
Spearman entre los perfiles horarios de las tres semillas.

```{include} tablas/e3_interpretabilidad.md
```

*Atribución por hora: suma de |atribución| sobre las variables, normalizada para que el perfil sume 1. Horas ancla:
τ, τ − 24, …, τ − 144. Los valores del Transformer, la TCN y KAN son los del E2 (`resultados/e2/f2a/gpu_resultados.json`).*

**Fidelidad.** Borrar las horas que señalan los gradientes integrados cambia el pronóstico de PADRE-TCN entre `0.68`
y `0.76` m/s, de 7 a 8 veces más que borrar horas al azar, igual que en los benchmarks. Las atribuciones son fieles.

**Coherencia con el dominio y con el mecanismo.** Las 8 horas más importantes de PADRE-TCN son la hora de emisión, las
de la misma hora de los 6 días anteriores (τ − 24, …, τ − 144) y τ − 4. Es el ciclo diario de la brisa, el patrón
dominante en el Caribe que ya encontramos en el E2. La diferencia está en cuánto se concentra la atribución: el `42` %
cae exactamente en las 7 horas ancla, frente al `33` % del Transformer y el `25` % de la TCN. El mecanismo hace explícito
lo que los demás modelos aprenden de forma implícita: el pronóstico se construye desde los días análogos. La velocidad
pesa el `59` %, y la dirección y la temperatura el resto, una proporción parecida a la de los benchmarks.

**¿Revela atajos?** No encontramos ninguno. El perfil horario de PADRE-TCN se parece mucho al de la TCN (Spearman
`0.91`), de la que hereda el codificador, y poco al del Transformer (`0.28`). Esa diferencia no viene de PADRE: el
Transformer pone un `2.5` % de su atribución en la hora más antigua de la ventana (τ − 167), que no tiene relación con
el ciclo diario y es probablemente un efecto de su codificación posicional. PADRE-TCN no tiene ese pico.

**Estabilidad entre semillas.** Las explicaciones de PADRE-TCN son las más estables de los cuatro modelos (Spearman
`0.93`–`0.95`, frente a `0.53`–`0.68` del Transformer y `0.86`–`0.91` de la TCN). Las anclas fijan de antemano qué
horas importan, así que cada semilla aprende un perfil parecido.

### Interpretabilidad propia del mecanismo: el peso de cada día

PADRE-TCN tiene además una explicación intrínseca que los benchmarks no tienen: los pesos wⱼ dicen cuánto pesa cada
día análogo en cada pronóstico. Los medimos en las 84 998 ventanas de prueba del origen 2025-10-01, con las tres
semillas.

```{include} tablas/e3_anclas.md
```

*Media ± desviación entre las tres semillas. El ancla j = 1 es la situación de hace un día; su objetivo es la
velocidad actual, así que su pronóstico corrige la persistencia.*

El ancla más reciente pesa más (`0.27`, frente a `0.167` si los pesos fueran uniformes) y es la de mayor peso en el
99.6 % de las ventanas. El peso decrece con la antigüedad del día, aunque no del todo en orden: el día 5 pesa algo más
que el 4, lo que sugiere que el sesgo por día β aprendió un rastro del ciclo semanal. **Con viento fuerte, el modelo se
apoya más en el día anterior** (`0.38`) y menos en los días lejanos: cuando el régimen es intenso, la situación de hace
seis días es peor análogo. Es coherente con la física del problema, pero no basta para corregir la subestimación de los
vientos fuertes. Por la Proposición 2, el pronóstico no puede salir del rango de las anclas, y en esos casos ninguna
propone un valor suficientemente alto.

Los pesos son casi uniformes en promedio: la entropía media es `1.75`, cerca del máximo ln 6 = `1.79`. Por eso la
ablación con pesos uniformes no cambia el RMSE. Lo que sí importa es la **dispersión** de los pronósticos de las anclas,
que es la base del intervalo y la que se correlaciona con el error.
