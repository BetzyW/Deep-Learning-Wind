# Discusión, limitaciones y conclusiones

## Resultados principales

| Modelo | RMSE 2025 (media de 6 semillas) | MAE | $R^2$ | MASE | Parámetros | Segundos por corrida |
|---|---|---|---|---|---|---|
| MLP | **0.706** | 0.498 | 0.753 | 0.854 | 102 785 | 6 |
| LSTM | 0.709 | 0.500 | 0.751 | 0.857 | 13 345 | 58 |
| CNN 1D | 0.714 | 0.506 | 0.747 | 0.867 | 44 129 | 31 |
| Persistencia | 0.850 | 0.583 | 0.641 | 1.000 | 0 | 0 |

## Respuesta a la pregunta de investigación

**1. Las tres redes superan con claridad a la persistencia.** El RMSE baja de 0.850 a 0.706–0.714 m/s, unos
0.14 m/s (≈ 17 %) en todas las estaciones. Las diferencias son significativas en las tres pruebas (Friedman,
Wilcoxon con Holm y Diebold–Mariano con corrección HLN, todas con $p < 10^{-4}$), con tamaños de efecto muy grandes
($|d| \approx 2.4$). La mejora es siete veces el margen de relevancia declarado ($\delta = 0.02$ m/s).

**2. Entre las tres redes no hay una diferencia de importancia práctica.**

- **MLP y LSTM son equivalentes.** Su diferencia media por estación es de 0.0004 m/s; la prueba TOST confirma que
  cae dentro de $\pm\delta$ ($p < 0.001$), y ni Wilcoxon ni Diebold–Mariano detectan diferencia.
- **La LSTM supera a la CNN** de forma estadísticamente significativa (Wilcoxon con Holm $p = 0.046$;
  Diebold–Mariano $p = 0.002$), pero la diferencia es de solo 0.004 m/s, **cinco veces menor que $\delta$**.
  Significativo no quiere decir relevante: con 211 000 horas de prueba, diferencias minúsculas se vuelven detectables.
- **El orden cambió entre validación y prueba.** En 2024 la LSTM fue la mejor; en 2025, el MLP. Con diferencias
  tan pequeñas, el ranking no es estable y no debe leerse como superioridad de una arquitectura.

$H_0$ (igual RMSE esperado de las tres redes) se rechaza formalmente por la diferencia LSTM–CNN, pero ninguna
diferencia entre redes alcanza el umbral de relevancia práctica.

## Trade-offs

- **El MLP es la mejor opción práctica:** el menor error, la menor variabilidad entre semillas (desviación de 0.0008
  m/s en validación) y entrena **diez veces más rápido** que la LSTM, porque procesa la ventana completa en paralelo.
- **La LSTM logra lo mismo con ocho veces menos parámetros** (13 345 frente a 102 785), lo que la haría preferible
  si la memoria fuera la restricción, pero su entrenamiento secuencial es el más lento.
- **La CNN queda en medio** en costo y ligeramente detrás en error.

La explicación más probable está en el EDA: el **ciclo diario** explica cerca de dos tercios de la varianza y la
persistencia a 24 h ya lo captura. Lo que las redes añaden es corregir el nivel según la semana previa, algo que una
red densa sobre la ventana aplanada aprende tan bien como una recurrente. El sesgo de memoria secuencial de la LSTM
no aporta cuando la señal útil está en rezagos fijos (1, 24 y 168 h).

Esto coincide con la literatura revisada: los estudios que compararon arquitecturas independientes no encuentran un
ganador consistente {cite}`sari2020,yaghoubirad2023,li2023`.

## Dónde fallan los modelos

**Vientos fuertes.** Las tres redes **subestiman** sistemáticamente los vientos altos: con velocidades de 6 m/s o
más el sesgo es de −1.2 a −1.3 m/s, mayor que el de la persistencia (−0.8 m/s). Al minimizar el MSE, el modelo se
inclina hacia la media, y los vientos fuertes son pocos. En el 5 % de horas más ventosas (≥ 4.8 m/s) las redes
apenas mejoran a la persistencia (RMSE 1.48 frente a 1.51 m/s).

**Errores por mes.** El error es mayor en marzo y abril (0.97–0.99 m/s), los meses de transición con más
variabilidad, y menor de septiembre a diciembre (≈ 0.60–0.66 m/s). Las redes superan a la persistencia en todos los
meses disponibles.

**Residuos con estructura.** Ljung-Box y BDS rechazan la ausencia de dependencia en las 53 estaciones evaluadas. Una
parte es esperable, porque los errores de un pronóstico a 24 h se solapan durante 23 horas; pero también indica
información que los modelos no aprovechan, como la relación entre estaciones vecinas.

## Sensibilidad: la presión

Con la presión como quinta variable el RMSE de validación **empeoró** en los tres modelos (0.808–0.815 frente a
0.792–0.797 m/s). La comparación es orientativa, porque con presión hay menos estaciones y el conjunto de validación
cambia, pero no muestra ninguna ganancia que justifique perder las estaciones sin barómetro. La decisión del EDA se
sostiene.

## Contexto con el proyecto de Machine Learning

El mejor modelo del proyecto de Machine Learning {cite}`rollo2026ml` obtuvo un RMSE de 0.719 m/s en 2025. El MLP
de este trabajo obtiene 0.706 m/s, pero **las cifras no son directamente comparables**: aquí el entrenamiento llega
hasta 2023 (allá, hasta 2024), las ventanas exigen una semana completa sin huecos y la presión se excluyó, de modo
que el conjunto de horas evaluadas es distinto. Una comparación justa, con el mismo conjunto de prueba, queda para el
siguiente entregable.

## Limitaciones inferenciales

- **Seis semillas.** Bastan para la variabilidad observada (diferencia mínima detectable de 0.004 m/s), pero
  capturan solo la aleatoriedad del entrenamiento, no la de los datos.
- **Un solo año de prueba.** 2025 es un único periodo; un año más seco o con más frentes podría cambiar el orden.
- **Validación desbalanceada.** El 41 % de las horas de 2024 cae en noviembre y diciembre, lo que puede explicar
  el cambio de orden entre validación y prueba.
- **Estaciones excluidas.** Las ventanas exigen 168 horas completas; las estaciones con muchos huecos (14 de 76 en
  prueba) no se evalúan, y los resultados no se extienden a ellas.
- **Dependencia espacial.** Las estaciones cercanas no son del todo independientes, así que las pruebas por estación
  pueden sobrestimar la evidencia.
- **Ajuste con una semilla.** Cada configuración se probó con una sola semilla; con diferencias tan pequeñas entre
  configuraciones, la elegida podría no ser la mejor en promedio.
- **Normalidad.** Las diferencias por estación no son normales (Shapiro-Wilk $p < 0.05$ en varios pares); por eso
  la conclusión se apoya en pruebas no paramétricas (Friedman, Wilcoxon) y en bootstrap, y la $d$ de Cohen y la TOST
  se leen como aproximaciones.

## Conclusiones

1. Las tres arquitecturas base reducen el error de la persistencia a 24 h en ≈ 17 %, una mejora significativa y
   relevante en todas las estaciones.
2. Las diferencias entre MLP, LSTM y CNN 1D son menores que el margen de relevancia; el MLP es la opción más
   eficiente.
3. El margen de mejora pendiente está en los **vientos fuertes** y en la **información espacial**, no en cambiar de
   arquitectura temporal.

**Para el Entregable 2:** modelos que usen las estaciones vecinas (redes de grafos o espacio-temporales), pérdidas
que no castiguen la cola alta (Huber o ponderadas por nivel de viento), Transformers con ventanas más largas y una
comparación justa con el modelo del proyecto de Machine Learning sobre el mismo conjunto de prueba.
