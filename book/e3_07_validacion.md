# Protocolo y validación estadística

Este capítulo responde la pregunta central del E3: ¿PADRE-TCN supera a la familia de contrastes (condición R3)? Si no
la supera, ¿se sostiene el hallazgo alternativo? Seguimos la sección 8 del enunciado: el estándar y el protocolo
pre-registrados, la equidad de la comparación, la familia de contrastes, la potencia, los supuestos y las pruebas, los
resultados de las hipótesis confirmatorias y las pruebas de ajuste y sobreajuste del modelo. Todos los números salen
de las predicciones de prueba de 2025 (4 orígenes trimestrales, 62 estaciones, 10 semillas por modelo) y las tablas se
generan con `src/tablas_e3.py` a partir de `resultados/e3/estadistica/` (calculado con `src/estadistica_e3.py`) y
`resultados/e3/adecuacion.json` (`src/adecuacion_padre.py`).

## Estándar y protocolo pre-registrados

El estándar de profundidad (`experiments/estandar.md`) y el protocolo (`experiments/protocolo.md`) se congelaron con
los tags de Git `e3-estandar` y `e3-protocolo` el **24 de noviembre de 2026 a las 00:05**, sobre el commit `2511863`.
La primera corrida confirmatoria de PADRE-TCN en la prueba de 2025 empezó a las 00:09 del mismo día. El protocolo fija
todo lo que pide la sección 8.1 del enunciado:

* **Estándar:** demostración formal con verificación numérica, tres conjuntos adicionales elegidos antes de ver
  resultados, familia completa de contrastes con Holm, tres pruebas por contraste, potencia por contraste de al menos
  0.90 y al menos 10 semillas.
* **Hipótesis:** H1 (superioridad en RMSE frente a los 12 rivales, con δ y las seis condiciones de la sección 8.4) y
  H2 (hallazgo alternativo: mejor intervalo al 90 % sin empeorar el RMSE en más de δ), con las predicciones
  secundarias del mecanismo como exploratorias (capítulo de diseño).
* **Métricas y margen:** RMSE por estación como métrica primaria; MAE, sesgo con viento fuerte, Winkler, cobertura y
  ancho como secundarias; δ = `0.02` m/s, heredado del E1.
* **Fuente del mecanismo, conjuntos adicionales, particiones, semillas (1000–1009), determinismo, familia, pruebas,
  corrección, nivel, presupuesto y potencia**, y el número de variantes de diseño, N_d = 9.

**Desviación posterior al tag.** Solo hubo una, y no afecta al panel principal: el recorte de STGCN a 3 semillas en
dos de los conjuntos adicionales (SDWPF y NOAA), decidido el 26 de noviembre por costo y antes de ver sus resultados
(`experiments/conjuntos_adicionales.md`). Se analiza en el capítulo de evaluación ampliada. Los recortes de Informer y
la corrección de DLinear en esos conjuntos son del 23 de noviembre, anteriores al tag.

## Equidad de la comparación

* **Mismo presupuesto de ajuste.** Medimos el presupuesto en número de configuraciones. Todos los modelos entrenables,
  incluido PADRE-TCN, probaron la misma lista de 20 configuraciones de (tasa de aprendizaje, decaimiento, lote),
  generada con la semilla 7, en el mismo origen de validación (2024-10) y con las mismas reglas: 10 % de las ventanas,
  20 épocas y paciencia 3. Las corridas finales usan en todos los casos un máximo de 50 épocas y paciencia 5, con la
  parada temprana sobre el mismo bloque.
* **Mismos datos.** Las mismas 62 estaciones, la misma ventana de 168 h con las mismas cuatro variables, el mismo
  escalado ajustado con el entrenamiento de cada origen, las mismas particiones con embargo de 24 h y la misma métrica.
  PADRE-TCN no usa información adicional: sus anclas salen de la misma ventana.
* **Benchmarks congelados.** Los 13 modelos del E2 se comparan con sus corridas finales del E2 (10 semillas en los 4
  orígenes), sin reentrenar ni excluir ninguno.
* **Pareado estricto.** PADRE-TCN usa las mismas semillas y los mismos orígenes, así que cada corrida tiene su par en
  cada rival.
* **Capacidad.** PADRE-TCN tiene `100 663` parámetros y `33.0` MFLOPs por ventana, prácticamente lo mismo que la TCN
  de la que parte (`98 433` y `32.9`) y que el Transformer (`100 801` y `40.3`). No hay ventaja de tamaño que separar
  del mecanismo.

```{include} tablas/e3_equidad.md
```

*Parámetros y FLOPs por ventana con el contador de PyTorch, y latencia en la CPU del PC con 4 hilos, medidos juntos en
la misma sesión (`src/costo_inferencia.py --e3`). En Chronos y Moirai se mide solo el cabezal entrenado; su modelo base
congelado se calcula una vez por ventana y se reporta aparte. Los minutos por corrida son tiempo de entrenamiento con
GPU, medidos en el equipo donde corrió cada modelo (el PC con una RTX 4060 Ti o Colab), así que sirven como orden de
magnitud y no para comparar con precisión.*

## Familia de contrastes

La familia confirmatoria tiene m = 12 contrastes: PADRE-TCN frente a cada modelo del Top 10 (Transformer, KAN, TCN,
STGCN, Chronos, GRU, Informer, DLinear, Moirai y PatchTST) y frente a los dos baselines fuertes (Ridge y gradient
boosting). La familia se fijó en el estándar, antes del tag, y no se excluyó a ningún rival. La afirmación "supera a
todos" es una prueba de unión-intersección: se acepta solo si se rechazan los 12 contrastes. MLP, CNN, LSTM y las
líneas base triviales se reportan, pero no entran en la familia. La ablación y los conjuntos adicionales son familias
aparte, cada una con su propio Holm.

## Potencia

**Antes de las corridas (protocolo, sección 9).** Con la fórmula del enunciado,
R ≈ (z₁₋α′/₂ + z₁₋β)² σ_d² / δ², con α′ = 0.05/12 (Bonferroni, más conservador que Holm) y potencia 0.90, el factor
es (2.865 + 1.282)² = `17.2`. Tomamos σ_d de la dispersión entre semillas de las corridas finales del E2, suponiendo
que PADRE-TCN varía como la TCN. En el peor caso (frente a Chronos, σ_d = `0.0086` m/s) bastaban unas 3 semillas para
detectar una diferencia igual a δ. Con 10 semillas, la diferencia mínima detectable era de `0.011` m/s en el peor caso
y de `0.005` m/s en el típico, ambas menores que δ. No hizo falta aumentar las semillas.

**Después de las corridas.** Recalculamos σ_d con las diferencias reales por semilla (media de los 4 orígenes) entre
PADRE-TCN y cada rival:

```{include} tablas/e3_potencia.md
```

La dispersión real quedó en el orden previsto. La diferencia mínima detectable con potencia 0.90 va de `0.0014` a
`0.0109` m/s, siempre por debajo de δ. Para un efecto igual a δ, la potencia de cada contraste es prácticamente 1 y la
potencia conjunta sobre los 12 contrastes es mayor que 0.99. Si cada contraste tuviera exactamente la potencia mínima
exigida (0.90), la conjunta sería 0.90¹² ≈ 0.28. **El diseño tenía potencia de sobra**, así que un resultado no
significativo no se puede atribuir a falta de semillas: significa que la diferencia real, si existe, es menor que la
que el diseño podía detectar.

## Pruebas y supuestos

El panel es de **series de tiempo** (familia F2 del cuadro 4), así que usamos tres pruebas por contraste: Wilcoxon de
rangos con signo sobre la diferencia de RMSE por estación, Diebold–Mariano con la corrección HLN sobre la serie horaria
de pérdida cuadrática {cite}`diebold1995,harvey1997` y bootstrap estacionario pareado de la diferencia de pérdida media
{cite}`politis1994`. Cada prueba se ajusta con Holm sobre los 12 contrastes {cite}`holm1979`, y la decisión de cada
contraste usa el **máximo** de los tres p ajustados: todas deben rechazar. Como procedimiento global usamos el Model
Confidence Set al 90 %. Para la robustez (condición 6) repetimos con un t pareado sobre las semillas, con y sin la
corrección de Nadeau–Bengio {cite}`nadeau2003`, y con un bootstrap por bloques fijos de una semana.

```{include} tablas/e3_pruebas.md
```

*Una fila por prueba; los p son el rango sobre los 12 contrastes de H1. El detalle por contraste está en la tabla de
resultados de H1.*

**Unidad de independencia.** Para Wilcoxon es la estación: los errores de una misma estación están correlacionados en
el tiempo, así que se resumen primero en su RMSE y después se comparan las 62 estaciones. Con 62 pares, la prueba de
rangos tiene resolución de sobra para el nivel ajustado de Holm, a diferencia de lo que pasaría con 10 semillas
(sección 8.7.2 del enunciado). Para Diebold–Mariano y el bootstrap, la unidad es la hora: se promedia la pérdida de
todas las estaciones y semillas en cada una de las 6 630 horas de prueba, y la dependencia temporal se trata con la
varianza HAC (23 rezagos, horizonte de 24 h) y con bloques de 168 h.

**Supuestos verificados.** Las series de diferencias de pérdida son estacionarias (la prueba ADF rechaza la raíz
unitaria en los 12 contrastes, con p < 0.001), condición que necesita Diebold–Mariano. También están autocorrelacionadas
(Ljung–Box con 24 rezagos rechaza en todos los casos), que es justo lo que corrigen la varianza HAC y los bloques del
bootstrap. Una prueba t ingenua sobre las horas sería inválida. Los modelos no están anidados (PADRE-TCN no es un caso
particular de ningún rival), así que Clark–West no aplica.

## Resultados de H1: superioridad en RMSE

```{include} tablas/e3_h1.md
```

*Δ es la diferencia de RMSE medio (PADRE − rival) sobre las 40 corridas; un valor negativo favorece a PADRE-TCN. El
IC 95 % es el de la diferencia de pérdida cuadrática horaria (m²/s²) con bootstrap estacionario; la δ de Cliff compara
los RMSE de las 62 estaciones (umbrales de Romano et al.: < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano).*

PADRE-TCN obtiene un RMSE medio de `0.7239` m/s. **No supera a la familia.** Las tres pruebas rechazan en solo 3 de
los 12 contrastes. En dos de ellos, Chronos y Moirai sin ajuste fino de su modelo base, PADRE-TCN es mejor por mucho
(`0.17` y `0.32` m/s, con efectos pequeño y mediano). En el tercero, el **Transformer es significativamente mejor que
PADRE-TCN**, pero solo por `0.005` m/s, cuatro veces menos que δ y con un efecto despreciable (δ de Cliff `+0.011`).
Frente a los otros 9 rivales, ninguna diferencia es significativa con las tres pruebas, y todas son menores que
`0.006` m/s. Todos los modelos entrenados quedan en una franja de `0.011` m/s, la misma que encontramos en el E2: con
estas entradas, el techo de desempeño no deja margen para que un mecanismo nuevo gane por δ.

El Model Confidence Set al 90 % confirma la lectura con un procedimiento global: PADRE-TCN queda dentro del conjunto de
los mejores modelos junto con otros 9, y solo excluye a Chronos, Moirai y STGCN.

```{include} tablas/e3_mcs.md
```

*MCS de Hansen et al. {cite}`hansen2011` sobre la serie horaria de pérdida cuadrática, con 2 000 réplicas, bloques de
168 h y estadístico del máximo. Un p del MCS mayor que 0.10 deja al modelo dentro del conjunto.*

**H1 se rechaza.** No es "ausencia de evidencia" por falta de potencia (el diseño podía detectar diferencias de
`0.003` m/s): es evidencia de que, frente a los modelos entrenados de la familia, las diferencias reales son menores
que δ. La equivalencia con la TCN se prueba formalmente en H2b.

## Resultados de H2: el hallazgo alternativo

### H2a: intervalos al 90 %

Con el mismo método para todos (split conformal entre orígenes, el residuo normalizado por una escala s), PADRE-TCN
usa s = √(dispersión² + std24²) y cada rival usa s = std24.

```{include} tablas/e3_h2a.md
```

*Winkler al 90 % en m/s; menor es mejor. Δ negativo favorece a PADRE-TCN.*

**Las tres pruebas rechazan en los 12 contrastes, todos a favor de PADRE-TCN.** Su Winkler es `2.991`, frente a `3.153`
del mejor rival (el Transformer), una mejora de `0.16` m/s (5.1 %); frente a la TCN es de `0.17` m/s (5.4 %). La
cobertura es `0.898`, muy cerca del 0.90 nominal, y el intervalo es más estrecho: `2.09` m/s de ancho medio frente a
unos `2.25` de los rivales. La δ de Cliff por estación es despreciable frente a los modelos entrenados (de `−0.09` a
`−0.10`) aunque la mejora sea consistente. La razón es que la δ de Cliff compara las distribuciones del Winkler entre
estaciones, y esas distribuciones son anchas porque las estaciones difieren mucho entre sí. Las pruebas pareadas, en
cambio, comparan cada estación consigo misma, y ahí la mejora aparece en casi todas.

### H2b: el RMSE no empeora más allá de δ

```{include} tablas/e3_h2b.md
```

*TOST (dos pruebas t unilaterales) sobre las 40 diferencias pareadas por semilla y origen, con margen ±δ
{cite}`schuirmann1987`.*

PADRE-TCN es `0.003` m/s peor que la TCN, y el IC 90 % de la diferencia, [`0.0018`, `0.0047`], queda muy dentro del
margen ±`0.02`. **Son equivalentes en RMSE.** **H2 se acepta**: con el mismo método de intervalo, PADRE-TCN da
intervalos mejores que los de todos los rivales sin perder precisión puntual.

### Predicciones secundarias

1. **La dispersión anticipa el error.** La ρ de Spearman entre la dispersión y el error absoluto es `0.43` en promedio
   (mínimo `0.39` entre corridas), mayor que 0, como se predijo. Con viento fuerte baja a `0.24`.
2. **Cobertura.** `0.898` en total y entre `0.881` y `0.913` por trimestre, dentro de 0.90 ± 0.02. Se cumple.
3. **Sin la dispersión, la ventaja debe desaparecer.** Es la predicción que conecta H2 con el mecanismo. Con la misma
   PADRE-TCN pero el intervalo normalizado solo con std24, el Winkler sube a `3.177`, prácticamente igual al de los
   rivales, y la diferencia frente a la versión con dispersión es significativa con las tres pruebas (capítulo de
   implementación, ablación). La ventaja viene de la dispersión entre anclas, no del pronóstico puntual.
4. **El sesgo con viento fuerte no mejora.** Con velocidad observada ≥ 6 m/s, PADRE-TCN subestima en `−1.21` m/s,
   igual que la TCN (`−1.22`), y su cobertura baja a `0.74`. La limitación del paper guía sigue presente.

## Ajuste y sobreajuste del modelo (condición 5)

Repetimos para PADRE-TCN los diagnósticos de las Tablas A y B del E2 (ruta F2a), con la TCN como referencia. Los
residuos son los del pronóstico medio de las 10 semillas.

```{include} tablas/e3_adecuacion.md
```

*Brecha y gap_ratio por corrida (40). Mincer–Zarnowitz: regresión y = a + b ŷ con varianza HAC de 23 rezagos y prueba
conjunta de a = 0 y b = 1. Pruebas de residuos por estación (61 estaciones con al menos 500 h): proporción que rechaza
al 5 %.*

* **Supera a la persistencia.** Su RMSE es `0.150` m/s menor, con las tres pruebas.
* **Brecha.** PADRE-TCN se ajusta más al entrenamiento que la TCN: su RMSE de entrenamiento es `0.724` frente a
  `0.748`, y su RMSE de parada es `0.025` m/s mayor que el de entrenamiento. **La regla absoluta del protocolo
  (|brecha| < δ) no se cumple.** Con el criterio relativo del E2 (gap_ratio < 0.15) sí se cumple, con una mediana de
  `0.023`, menor que la de la TCN (`0.034`). La regla del protocolo es la que cuenta para R3, así que la condición 5
  falla. Lo reportamos tal cual, junto con la lectura de que el sobreajuste es pequeño: el RMSE de prueba (`0.724`) no
  es peor que el de validación (`0.749`).
* **Calibración del pronóstico puntual.** Mincer–Zarnowitz no rechaza a = 0 y b = 1 en el agregado (p = `0.92`;
  b = `1.002`), mejor que la TCN (p = `0.17`). Por estación rechaza en el 56 % de las estaciones, frente al 44 % de la
  TCN: el sesgo condicional existe en estaciones concretas y se compensa en el agregado.
* **Residuos.** En casi todas las estaciones, los residuos de los dos modelos están autocorrelacionados, son
  heterocedásticos y no son normales, en la misma proporción. Es lo esperable en un pronóstico a 24 h con datos
  horarios: los horizontes de ventanas consecutivas se solapan 23 horas, y el viento cambia de varianza con el régimen.
  Por eso las pruebas usan HAC y bloques, y los intervalos se calibran por conformal y no con un supuesto normal.
* **Consistencia entre validación y prueba.** El RMSE de prueba de PADRE-TCN (`0.724`) es menor que el de parada
  (`0.749`), con la misma relación que la TCN (`0.721` frente a `0.745`): los trimestres de prueba de 2025 son algo más
  fáciles que el bloque de parada, para todos los modelos por igual.

## Veredicto sobre R3

```{include} tablas/e3_r3.md
```

**PADRE-TCN no cumple R3.** Fallan las condiciones 1, 2, 3 y la parte de la brecha de la 5. La condición 4 (conjuntos
adicionales) se completa en el capítulo de evaluación ampliada, pero no cambia el veredicto, porque bastaba con que
fallara una. Según el árbol de decisión de la sección 6.6 del enunciado, el tope de la nota es **4.0**, siempre que se
sostengan R1 (novedad, capítulo de literatura) y R2 (implementación propia, capítulo de implementación).

El trabajo se sostiene en el **hallazgo alternativo declarado antes del tag (H2)**, que se cumple completo. PADRE-TCN
predice igual que los mejores modelos (equivalente a la TCN dentro de δ) y además da, sin costo adicional apreciable,
intervalos de predicción al 90 % un 5 % mejores que los de cualquier modelo de la familia con el mismo método. La
mejora viene del mecanismo propio, la dispersión entre anclas, como muestra la ablación. La conclusión no cambia con
las pruebas alternativas (condición 6). La causa probable de que no haya superioridad en RMSE no es la potencia ni la
varianza entre semillas: es el techo de desempeño que ya medimos en el E2. Con las mismas cuatro variables de entrada,
todos los modelos entrenados extraen prácticamente la misma información, y la diferencia restante es menor que la que
importa en la práctica.
