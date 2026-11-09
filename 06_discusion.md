# Discusión, limitaciones y conclusiones

## Resultados principales

| Modelo | RMSE 2025, media ± de (IC 95 %) | MAE | $R^2$ | MASE | Parámetros | Entrenamiento (s) | Inferencia 2025 (s) |
|---|---|---|---|---|---|---|---|
| MLP | **0.706 ± 0.002** (0.703–0.708) | 0.498 | 0.753 | 0.854 | 102 785 | 7 | 0.02 |
| LSTM | 0.709 ± 0.002 (0.707–0.711) | 0.500 | 0.751 | 0.857 | 13 345 | 58 | 0.40 |
| CNN 1D | 0.714 ± 0.005 (0.709–0.720) | 0.506 | 0.747 | 0.867 | 44 129 | 31 | 0.44 |
| Persistencia | 0.850 | 0.583 | 0.641 | 1.000 | 0 | — | — |
| *Ridge lineal (contexto, no es Deep Learning)* | *0.698* | — | — | — | *673* | — | — |

## Respuesta a la pregunta de investigación

**1. Las tres redes superan con claridad a la persistencia.** El RMSE baja de 0.850 a 0.706–0.714 m/s, una reducción
del 16 % al 17 % (≈ 0.14 m/s). Las diferencias son significativas en todas las pruebas (Friedman con post-hoc de
Nemenyi, Wilcoxon con Holm y Diebold–Mariano con corrección HLN, todas con $p < 10^{-4}$), con tamaños de efecto muy
grandes ($|d| \approx 2.4$). La mejora es siete veces el margen de relevancia declarado ($\delta = 0.02$ m/s).

**2. Entre las tres redes no hay una diferencia de importancia práctica.**

- **MLP y LSTM son equivalentes.** Su diferencia media por estación es de 0.0004 m/s; la prueba TOST confirma que
  cae dentro de $\pm\delta$ ($p < 0.001$), y ni Wilcoxon ni Diebold–Mariano detectan diferencia.
- **La LSTM supera a la CNN** según Wilcoxon con Holm ($p = 0.046$) y Diebold–Mariano ($p = 0.002$), pero no según
  el post-hoc de Nemenyi (diferencia de rangos 0.26 frente a una diferencia crítica de 0.60). La diferencia es de solo
  0.004 m/s, **cinco veces menor que $\delta$**: significativa en algunas pruebas, pero no relevante.
- **El orden cambió entre validación y prueba.** En 2024 la LSTM fue la mejor; en 2025, el MLP. Con diferencias
  tan pequeñas, el ranking no es estable y no debe leerse como superioridad de una arquitectura.
- **La arquitectura explica solo la mitad de la variación.** Entre las 18 corridas, el 54 % de la varianza del RMSE
  de prueba se debe al modelo y el 46 % a la semilla: cambiar de arquitectura mueve el error casi lo mismo que
  cambiar la semilla.

$H_0$ (igual RMSE esperado de las tres redes) se rechaza formalmente por la diferencia LSTM–CNN, pero ninguna
diferencia entre redes alcanza el umbral de relevancia práctica.

**3. Un modelo lineal hace lo mismo o un poco mejor.** La regresión ridge sobre la misma ventana aplanada, que el
enunciado sitúa como referencia clásica de contexto (equivalente a la SVR lineal), obtiene un RMSE de 0.698 m/s en
2025 y 0.789 en validación: **mejor que las tres redes** por unos 0.008 m/s, una diferencia menor que $\delta$. Es
decir, con esta entrada (la propia historia de la estación), la no linealidad de las redes base **no aporta**. Es
un resultado honesto y útil: el problema, tal como está planteado, es casi lineal sobre la ventana, y para que el
Deep Learning se justifique hay que darle información que un modelo lineal no pueda explotar (estaciones vecinas,
interacciones con la dirección y la estación del año) o mejorar donde lo lineal falla (vientos fuertes).

## Baseline y umbral de mejora

- **Baseline:** el **MLP**, con RMSE = 0.706 m/s (IC 95 % entre semillas: 0.703–0.708), por tener el menor error, la
  menor variabilidad y el menor costo.
- **Diferencia mínima detectable** con 6 semillas y potencia 0.80: 0.007 m/s; es menor que $\delta$, así que el diseño
  tiene potencia suficiente para el margen declarado.
- **Umbral para un modelo avanzado:** RMSE **< 0.686 m/s** en 2025, es decir, una mejora de al menos **2.8 %** sobre
  el baseline. Por debajo de ese valor la mejora es a la vez estadísticamente detectable y prácticamente relevante.
- **Contexto:** la literatura revisada reporta mejoras del 8 % al 33 % de modelos avanzados sobre redes base, pero
  sin repeticiones y con riesgo de fuga (capítulo de literatura). Aquí el margen real es menor: ni siquiera la ridge
  lineal supera el umbral.
- **Cota de referencia:** los anemómetros tienen un umbral de arranque del orden de 0.5 m/s y resolución de décimas,
  lo que pone un piso al error en vientos débiles; además, la persistencia en el 5 % de horas más ventosas tiene un
  RMSE de 1.5 m/s, que da idea de la variabilidad que ningún modelo con solo historia local puede anticipar del todo.

## Trade-offs

- **El MLP es la mejor opción práctica:** el menor error, la menor variabilidad entre semillas y entrena **ocho veces
  más rápido** que la LSTM, porque procesa la ventana completa en paralelo; predecir todo 2025 le toma 0.02 s.
- **La LSTM logra casi lo mismo con ocho veces menos parámetros** (13 345 frente a 102 785), lo que la haría
  preferible si la memoria fuera la restricción, pero su entrenamiento secuencial es el más lento.
- **La CNN queda en medio** en costo y ligeramente detrás en error.
- **Más complejidad no compra precisión:** el modelo con más parámetros (MLP) y el de menos (LSTM) empatan, y el
  modelo lineal, con 673 coeficientes, los iguala.

La explicación más probable está en el EDA: el **ciclo diario** explica cerca de dos tercios de la varianza y la
persistencia a 24 h ya lo captura. Lo que los modelos añaden es corregir el nivel según la semana previa, algo casi
lineal que una red densa aprende tan bien como una recurrente. El sesgo de memoria secuencial de la LSTM no aporta
cuando la señal útil está en rezagos fijos (1, 24 y 168 h).

Esto coincide con la literatura revisada: los estudios que compararon arquitecturas independientes no encuentran un
ganador consistente {cite}`sari2020,yaghoubirad2023,li2023,saxena2021`.

## Generalización y robustez

- **Validación frente a prueba:** los tres modelos tienen menor error en 2025 (0.71) que en 2024 (0.79), lo que no
  indica sobreajuste sino la diferente composición de los años: la validación está cargada hacia noviembre y
  diciembre, meses más ventosos.
- **Curvas de aprendizaje:** el error de validación se estabiliza en pocas épocas y la parada temprana corta antes
  de que suba; la brecha con el error de entrenamiento (0.035–0.048 m/s) es moderada.
- **Por estación:** las tres redes superan a la persistencia en la mediana de las 62 estaciones (0.56 frente a
  0.72 m/s).

## Dónde fallan los modelos

**Vientos fuertes.** Las tres redes **subestiman** sistemáticamente los vientos altos: con velocidades de 6 m/s o
más el sesgo es de −1.2 a −1.3 m/s, mayor que el de la persistencia (−0.8 m/s). Al minimizar el MSE, el modelo se
inclina hacia la media, y los vientos fuertes son pocos. En el 5 % de horas más ventosas (≥ 4.8 m/s) las redes
apenas mejoran a la persistencia (RMSE 1.48 frente a 1.51 m/s).

**Errores por mes.** El error es mayor en marzo y abril (0.97–0.99 m/s), los meses de transición con más
variabilidad, y menor de septiembre a diciembre (≈ 0.60–0.66 m/s). Las redes superan a la persistencia en todos los
meses disponibles.

**Residuos con estructura.** Ljung-Box (con 24, 48 y 168 rezagos) y BDS rechazan la ausencia de dependencia en las
53 estaciones evaluadas. Una parte es esperable, porque los errores de un pronóstico a 24 h se solapan durante 23
horas; pero también indica información que los modelos no aprovechan, como la relación entre estaciones vecinas.

**Sesgo medio.** El sesgo global de los residuos es pequeño (−0.01 m/s): pese a los quiebres de nivel detectados en
el EDA, los modelos no sobrestiman en promedio.

**Atípicos.** Solo 0.08 % de las horas objetivo de prueba son atípicas; quitarlas baja el RMSE en apenas 0.004 m/s,
así que no explican las diferencias.

## Sensibilidad: la presión

Con la presión como quinta variable el RMSE de validación **empeoró** en los tres modelos (0.808–0.815 frente a
0.792–0.797 m/s). La comparación es orientativa, porque con presión hay menos estaciones y el conjunto de validación
cambia, pero no muestra ninguna ganancia que justifique perder las estaciones sin barómetro. La decisión del EDA se
sostiene.

## Contexto con el proyecto de Machine Learning

El mejor modelo del proyecto de Machine Learning {cite}`rollo2026ml` obtuvo un RMSE de 0.719 m/s en 2025. El MLP
de este trabajo obtiene 0.706 m/s, pero **las cifras no son directamente comparables**: aquí el entrenamiento llega
hasta 2023 (allá, hasta 2024), las ventanas exigen una semana completa sin huecos y la presión se excluyó, de modo
que el conjunto de horas evaluadas es distinto. Ningún estudio de la revisión usa datos del IDEAM; el proyecto de
Machine Learning es el único antecedente con este dataset. Una comparación justa, con el mismo conjunto de prueba,
queda para el siguiente entregable.

## Amenazas a la validez

**Validez interna**
- *Fuga de información:* controlada con partición cronológica, embargo de $L + 24$ horas, escalado ajustado solo con
  entrenamiento, relleno solo hacia el pasado y conjunto de prueba con huella registrada. El EDA describió el panel
  completo (incluido 2025), pero ninguna decisión de modelado se tomó mirando el error de prueba.
- *Optimización desigual:* mitigada con el mismo presupuesto de búsqueda (20 configuraciones) para los tres modelos;
  aun así, cada configuración se evaluó con una sola semilla.
- *Una sola corrida de ajuste por configuración:* la configuración elegida podría no ser la mejor en promedio, dado
  que las diferencias entre configuraciones son del mismo orden que la variación entre semillas.

**Validez externa**
- Los resultados valen para estaciones automáticas del Caribe colombiano, con ventanas completas de una semana y un
  año de prueba (2025); no se extienden automáticamente a otras regiones, a estaciones con muchos huecos (14 de 76
  quedaron fuera de la prueba) ni a horizontes distintos de 24 h.
- Las estaciones vecinas no son del todo independientes, así que las pruebas por estación pueden sobrestimar la
  evidencia.

**Validez de constructo**
- El RMSE castiga los errores grandes, coherente con la pérdida de entrenamiento, pero refleja sobre todo el
  desempeño en vientos moderados (los más frecuentes); por eso se reportan aparte los vientos fuertes, el MAE y el
  MASE.
- La persistencia a 24 h es el ingenuo adecuado (coincide con el ingenuo estacional diario), pero no es el único
  referente operativo posible: un pronóstico numérico del tiempo sería una referencia más exigente.

## Limitaciones inferenciales

- **Seis semillas.** Bastan para la variabilidad observada (diferencia mínima detectable de 0.007 m/s, menor que
  $\delta$), pero capturan solo la aleatoriedad del entrenamiento, no la de los datos.
- **Pruebas con pocos bloques.** Las pruebas se hicieron con las 62 estaciones como bloques, no con las 6 semillas:
  con 6 semillas el menor p-valor posible de Wilcoxon bilateral es 0.031 y Friedman tendría muy poca potencia
  (Sección 7 del enunciado). Usar las estaciones da potencia, a costa de suponerlas independientes.
- **Un solo año de prueba.** 2025 es un único periodo; un año más seco o con más frentes podría cambiar el orden.
- **Validación desbalanceada.** El 41 % de las horas de 2024 cae en noviembre y diciembre, lo que puede explicar
  el cambio de orden entre validación y prueba.
- **Normalidad.** Las diferencias por estación no son normales (Shapiro-Wilk $p < 0.05$ en varios pares); por eso
  la conclusión se apoya en pruebas no paramétricas (Friedman, Nemenyi, Wilcoxon) y en bootstrap, y la $d$ de Cohen y
  la TOST se leen como aproximaciones.

## Conclusiones

1. Las tres arquitecturas base reducen el error de la persistencia a 24 h en ≈ 17 %, una mejora significativa y
   relevante en todas las estaciones.
2. Las diferencias entre MLP, LSTM y CNN 1D son menores que el margen de relevancia; el **MLP es el baseline**
   (RMSE 0.706 m/s) por ser el más preciso, estable y barato.
3. Un modelo avanzado debe bajar de **0.686 m/s** (mejora ≥ 2.8 %) para considerarse una contribución.
4. Una regresión lineal iguala a las redes base: el margen de mejora no está en cambiar de arquitectura temporal,
   sino en los **vientos fuertes** y en la **información espacial** que hoy no se usa.
