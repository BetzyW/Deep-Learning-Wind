# Discusión (Entregable 2)

Este capítulo interpreta los resultados del capítulo anterior: responde cada pregunta de investigación, explica por
qué los modelos quedan tan juntos, discute lo que pasó con los preentrenados y declara las amenazas a la validez.

## Respuesta a las preguntas de investigación

| Pregunta | Hipótesis | Resultado | Evidencia principal |
|---|---|---|---|
| RQ1 | H1: algún modelo del Top 10 supera a Ridge por más de δ | **Se rechaza** | 7 modelos ganan con significancia tras Holm, pero la mayor diferencia es `0.0073` m/s < δ = `0.02` |
| RQ2 | H2: el mejor preentrenado es equivalente al mejor desde cero | **Se rechaza** | Chronos es peor por `0.127` m/s (IC 90 % [`0.087`, `0.166`]); el TOST no declara equivalencia. Sin entrenar, Moirai sigue peor por `0.038` m/s |
| RQ3 | H3: la subestimación en vientos ≥ 6 m/s aparece en DLinear y en al menos otros dos modelos | **Se cumple** | Sesgo negativo en todos los modelos y en todas las estaciones con viento fuerte, de `−1.15` a `−1.28` m/s |

**RQ1: ganancias reales pero pequeñas.** El resultado del Entregable 1 se mantiene a una escala mucho mayor. Allí tres
redes base empataron con una regresión lineal; aquí diez arquitecturas recientes, con búsqueda de arquitectura y el
mismo presupuesto de ajuste, le ganan a Ridge por milésimas. Las pruebas coinciden en la dirección: Wilcoxon por
estación, Diebold–Mariano sobre la pérdida horaria y Clark–West. El estudio tenía potencia para ver una diferencia
cuatro veces menor que δ (diferencia mínima detectable de `0.0045` m/s), así que no encontrar una diferencia relevante
no es falta de datos. Con estas entradas, la mejor red no le saca a un modelo lineal bien ajustado una diferencia que
importe en la práctica.

**RQ2: los preentrenados no alcanzan a los modelos entrenados.** Con nuestra adaptación, Chronos apenas mejora a la
persistencia y Moirai es peor que ella. Sin entrenar los dos mejoran, pero siguen por detrás de los modelos desde cero
y de Ridge. Lo discutimos en su propia sección más abajo.

**RQ3: la limitación del paper guía es de toda la familia.** DLinear subestima los vientos fuertes, pero también lo hacen
el Transformer, la TCN, la KAN, Ridge y los demás, con magnitudes casi iguales. Esto es lo que se espera de cualquier
modelo entrenado con error cuadrático: el pronóstico que minimiza el error cuadrático es la media condicional, y la
media condicional queda por debajo cuando el valor observado es extremo. Que hasta la persistencia subestime
(`−0.76` m/s) muestra que una parte del sesgo viene de condicionar en la observación: si elegimos las horas en que el
viento fue fuerte, cualquier pronóstico que no sabía que lo sería queda por debajo. La otra parte (la diferencia entre
`−0.76` y `−1.2`) sí es propia de los modelos, que se acercan a la media más que la persistencia.

## Por qué todos los modelos quedan juntos

Cuatro resultados independientes apuntan a lo mismo: **el límite está en la información de entrada, no en la
arquitectura.**

1. **El techo está casi alcanzado.** Un promedio de 13 modelos diversos baja el RMSE solo de `0.692` a `0.690`, y un
   oráculo que elige el mejor modelo en cada estación mirando la prueba llega a `0.688`. El mejor modelo ya cubrió el
   98–99 % del margen alcanzable (γ = `0.011`–`0.023`).
2. **Los errores son los mismos.** El 86 % de los errores grandes de la TCN también son errores grandes del
   Transformer y de la KAN, y el índice de Jaccard con los errores grandes de Ridge es de `0.77`–`0.79`. Si los
   modelos fallaran en ventanas distintas, combinarlos ayudaría; como fallan en las mismas, no.
3. **La época del año pesa más que el modelo.** En el RMSE de las corridas finales, el origen explica el `99.5` % de la
   variación y el modelo el `0.27` %.
4. **La arquitectura importa poco incluso dentro de cada modelo.** En la búsqueda, casi todos los finalistas fueron
   equivalentes al mejor con ±δ, y en GRU la dispersión entre arquitecturas fue igual a la dispersión entre semillas.

Los residuos confirman que queda estructura, pero no la que estos modelos pueden aprovechar. En casi todas las
estaciones los residuos de TCN, Transformer, KAN y Ridge están autocorrelacionados (Ljung–Box), son heterocedásticos
(ARCH-LM) y no son normales (Jarque–Bera). La regresión de Mincer–Zarnowitz {cite}`mincer1969` sobre todas las
ventanas no rechaza que TCN (p = `0.22`) y Transformer (p = `0.61`) sean insesgados; sí lo rechaza para KAN y Ridge
(p < `0.01`), aunque con pendientes muy cercanas a 1. La varianza que cambia con el régimen de viento y la
autocorrelación a 24 h no se pueden explicar con la historia de la propia estación. Lo que falta es información
nueva: la de las estaciones vecinas (que STGCN usa solo en parte), la de los modelos numéricos del tiempo o la de
variables de gran escala.

**La ventaja no es constante en el tiempo.** La prueba de fluctuación de Giacomini y Rossi {cite}`giacomini2010`
detecta inestabilidad en la ventaja de TCN y del Transformer frente a Ridge (estadístico máximo de `5.34` y `7.46`
frente a un valor crítico de `3.43`), pero no en la de KAN (`3.27`). La ventaja de las dos mejores redes se concentra
en partes del año y no se mantiene igual todo el tiempo. Esto concuerda con el rechazo de la prueba condicional de
Giacomini–White: la conclusión de RQ1 es un promedio sobre 2025, no una propiedad que valga en cada trimestre.

**Los modelos usan lo que se espera, y les falta lo mismo.** Los gradientes integrados muestran que los tres mejores
modelos se apoyan en el nivel más reciente y en la misma hora de los días anteriores, es decir, en el ciclo diario de la
brisa. La ablación lo confirma por otro camino: quitarle a la TCN la dilatación, y con ella el acceso a los días
anteriores, empeora el RMSE en `0.025` m/s. Es la única diferencia de todo el E2 que supera δ. Lo que los modelos
aprenden es, en esencia, nivel más ciclo diario, y eso también lo puede representar un modelo lineal sobre los
rezagos, lo que explica por qué Ridge queda tan cerca. Esa misma dependencia del nivel reciente es la que los hace
frágiles ante un quiebre de nivel o ante horas faltantes, y la que hace que sus intervalos, de ancho fijo, cubran de
menos justo cuando pronostican viento alto. La incertidumbre crece con el nivel del viento, y un modelo para el E3
debería modelarla en lugar de suponerla constante.

## Los preentrenados: el modelo o nuestra adaptación

Chronos y Moirai se diseñaron para pronosticar sin entrenar en el conjunto nuevo (*zero-shot*), con su propio
decodificador {cite}`ansari2024,woo2024`. Nosotros los usamos de otra forma: congelamos el modelo base, resumimos su
representación con un **promedio** sobre las horas (Chronos) o los parches (Moirai) y entrenamos un cabezal denso con
la misma búsqueda de arquitectura que los demás. Elegimos esta adaptación para que todos los modelos tuvieran el mismo
presupuesto de ajuste y la misma salida, pero tiene tres consecuencias que explican buena parte del mal resultado:

- **El promedio borra el orden temporal.** Para un pronóstico a 24 h lo más informativo es el nivel reciente; al
  promediar, la hora 168 pesa lo mismo que la hora 1. En Moirai, donde los parches son de 32 h, el último nivel
  observado se diluye todavía más, y por eso es peor que la persistencia.
- **Son univariados en nuestro uso.** No ven la dirección ni la temperatura, que los modelos desde cero sí reciben.
- **El escalado propio de cada modelo se pierde.** Chronos y Moirai normalizan cada ventana internamente; el cabezal
  recibe una representación sin la escala original y tiene que recuperarla.

Por eso la respuesta a RQ2 debe leerse como "**los preentrenados congelados con un resumen por promedio** no son
equivalentes a los modelos desde cero", no como un juicio general sobre los modelos fundacionales.

Para separar el modelo de la adaptación, corrimos además Chronos y Moirai **sin entrenar**, con su propio
decodificador (enmienda 9, análisis secundario). El resultado responde la pregunta en dos partes:

- **La adaptación sí los empeoró.** Sin entrenar, Moirai baja su RMSE medio por estación en `0.238` m/s frente a su
  versión con cabezal y Chronos en `0.056` m/s. Los dos pasan a ganarle con claridad a la persistencia, cosa que
  Moirai con cabezal no lograba. El promedio sobre las horas y la pérdida de la escala les quitaban justo la
  información del nivel reciente.
- **Aun así, no alcanzan a los modelos entrenados.** Moirai sin entrenar queda `0.038` m/s por encima del Transformer
  y `0.031` m/s por encima de Ridge, y el TOST tampoco declara equivalencia. La conclusión de H2 se sostiene con la
  adaptación declarada y también con el uso original de los modelos.

Hay un punto a su favor que no esperábamos. El intervalo del 80 % de Moirai, que sale de sus propios cuantiles, cubre
el `81.3` % de los casos sin ninguna calibración. En las mismas ventanas con viento fuerte, Moirai subestima menos
(`−0.86` m/s) que los modelos entrenados con error cuadrático (`−1.23` m/s el Transformer, la TCN y Ridge). Un modelo que pronostica una distribución y no solo la media
sufre menos el sesgo de RQ3. Es una pista para el E3. Chronos, en cambio, da intervalos demasiado estrechos (cobertura
de `54.2` %).

No cambiamos la adaptación declarada después de ver la prueba, porque eso habría sido ajustar al conjunto de prueba.
La otra comparación posible, resumir la representación con el último estado en lugar del promedio, queda pendiente.

## Costo frente a beneficio

Las fases de búsqueda costaron `44.6` horas de GPU entre el PC del equipo y Colab, más las corridas finales (13 modelos
× 4 orígenes × 10 semillas), y consumieron cerca de `285` de las 600 unidades de Colab. Ridge se entrena en segundos
en una CPU, tiene unos cientos de parámetros y queda a menos de `0.01` m/s del mejor modelo. Para un uso operativo con
estas entradas, el modelo lineal es la opción razonable; las redes solo se justifican si se les da información que un
modelo lineal no puede usar bien (relaciones espaciales no lineales, regímenes de viento, variables exógenas).

## Amenazas a la validez

**Validez interna.**

- **El cribado de la fase 1 fue ruidoso.** Con una semilla y 12 épocas, el orden de la fase 1 no anticipó el de la
  fase 2 (ρ negativo en varios modelos). El ganador de cada modelo es el mejor entre los finalistas, no necesariamente
  el mejor de las 1 000. Como casi todos los finalistas son equivalentes, el efecto sobre las conclusiones es pequeño.
- **La muestra de 200 arquitecturas** en cinco modelos (enmiendas 2 y 3). En MLP y CNN, que tienen las 1 000, el mejor
  de la muestra quedó a `0.001`–`0.004` m/s del mejor de todas, muy por debajo de δ.
- **Elegir con un solo origen.** La fase 2 eligió la arquitectura solo con el cuarto trimestre de 2024. Al repetirla
  en los otros tres trimestres, la elegida quedó a δ o menos de la mejor en 8 de 9 modelos. La excepción son los
  cabezales de Moirai, que no cambian ninguna conclusión.
- **Enmiendas.** Hubo nueve. Las seis primeras se hicieron antes de abrir la prueba; la séptima (la ablación), la
  octava (la validación de la selección) y la novena (los preentrenados sin entrenar) se declaran como análisis
  secundarios hechos después.
- **Fuga de información.** La partición con embargo, el reentrenamiento en cada origen y el escalado por origen se
  verificaron con 9 pruebas automáticas.

**Validez de constructo.** El RMSE premia acercarse a la media y castiga poco la subestimación sistemática de los
extremos; un modelo puede ganar en RMSE y ser peor justo en los vientos que más importan para la generación eólica. Por
eso reportamos aparte el sesgo en vientos fuertes (RQ3) y la taxonomía de errores.

**Validez externa.** Un solo panel (62 estaciones del IDEAM en el Caribe colombiano), con muchos datos faltantes y
estaciones no sincronizadas; un solo año de prueba (2025) con cuatro orígenes. Los resultados pueden no trasladarse a
parques eólicos con anemómetros a la altura del buje ni a otros climas.

**Validez de la conclusión estadística.**

- **Holm y no FDR.** Las familias F1 y F2 son confirmatorias, con pocos contrastes (el Top 10 y los tres modelos base) y una pregunta
  binaria por contraste: ¿este modelo le gana al clásico? Ahí lo adecuado es controlar la probabilidad de cometer al
  menos un falso positivo (FWER), y Holm lo hace con más potencia que Bonferroni {cite}`holm1979`. El control de la
  tasa de falsos descubrimientos de Benjamini y Hochberg {cite}`benjamini1995` es para cribados con muchos contrastes,
  donde se aceptan algunos falsos positivos a cambio de potencia. Además, con la diferencia mínima detectable en
  `0.0045` m/s la potencia no era un problema, y cambiar a FDR no habría cambiado ninguna conclusión.
- **Independencia.** Usamos la estación como unidad, pero las estaciones vecinas comparten clima, así que las 62 no
  son del todo independientes y los p-valores de Wilcoxon pueden ser algo optimistas. Diebold–Mariano, el MCS con
  bootstrap por bloques y la prueba de fluctuación, que trabajan sobre la serie temporal, llegan a las mismas
  conclusiones.

## Implicaciones para el Entregable 3

El Entregable 2 deja tres pistas para el modelo propio del E3:

1. **Cambiar la pérdida, no la arquitectura.** La subestimación de los vientos fuertes es común a todos los modelos
   entrenados con error cuadrático; una pérdida que penalice la subestimación de los extremos, o un pronóstico por
   cuantiles, ataca directamente la limitación de RQ3.
2. **Agregar información, no capacidad.** Con el techo casi alcanzado, solo se puede mejorar con entradas nuevas,
   como las estaciones vecinas o variables exógenas.
3. **Comparar contra Ridge.** Cualquier mejora del E3 tiene que medirse contra el clásico fuerte con el mismo
   protocolo, porque ese es el rival real.
