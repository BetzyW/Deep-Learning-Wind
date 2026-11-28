# Conclusiones y limitaciones

## Hallazgos

1. **PADRE-TCN predice tan bien como los mejores modelos, pero no mejor.** Su RMSE en la prueba de 2025 es `0.7239`
   m/s. Es equivalente a la TCN dentro del margen δ = `0.02` m/s (TOST), queda dentro del Model Confidence Set de los
   mejores modelos y solo pierde con el Transformer, que es `0.005` m/s mejor: cuatro veces menos que δ.
2. **Da los mejores intervalos de predicción de toda la familia.** Con el mismo método conformal para todos, su
   *interval score* de Winkler al 90 % es `2.991` frente a `3.153` del mejor rival, alrededor de un 5 % mejor, con las
   tres pruebas a su favor en los 12 contrastes, cobertura `0.898` e intervalos más estrechos.
3. **La mejora viene del mecanismo propio.** La ablación muestra que, sin la dispersión entre anclas, el Winkler sube
   a `3.177` y la ventaja desaparece (p < 0.001). La dispersión se correlaciona con el error (ρ = `0.43`).
4. **La propiedad demostrada se cumple exactamente.** La diferencia es antisimétrica para cualquier valor de los
   parámetros, también en coma flotante (verificado en el modelo entrenado con 20 000 ventanas), y el pronóstico queda
   siempre dentro del rango de sus anclas.
5. **Es el modelo más robusto a la elección de hiperparámetros** (su RMSE varía `0.006` m/s entre las 20
   configuraciones del ajuste) y de los más estables entre semillas, con explicaciones más estables que las del
   Transformer, el 42 % de su atribución concentrada en las horas ancla y sin atajos detectados.
6. **Fuera del viento horario no destaca.** En los conjuntos adicionales queda 4.º de 11 en ETTh1, 10.º en SDWPF y
   8.º en NOAA. Entre los cuatro conjuntos, ningún modelo es consistentemente mejor (Friedman, p = `0.73`).
7. **No corrige la limitación del paper guía.** Con viento fuerte subestima en `−1.21` m/s, igual que la TCN, y ahí la
   persistencia tiene menos error que los tres modelos neuronales analizados.

## Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Sustento |
|---|---|---|
| R1. Novedad para el problema | **Sí** | Mecanismo transferido de la química (PADRE), de la física (TNNR) y de la meteorología (análogos), con seis modificaciones algorítmicas (capítulo de diseño). La búsqueda en tres bases no encontró la regresión por diferencias con anclas aplicada al pronóstico de la velocidad del viento; el trabajo más cercano usa "gemela" en otro sentido (capítulo de literatura) |
| R2. Implementación propia | **Sí** | `src/original_model.py`, escrito desde las ecuaciones con capas primitivas de PyTorch; 8 pruebas unitarias, incluida la verificación de la propiedad (capítulo de implementación) |
| R3. Superioridad sobre la familia | **No** | Fallan las condiciones 1, 2 y 3 (no supera en RMSE a los 12 rivales), la 4 (conjuntos adicionales) y la regla de la brecha de la 5 (capítulos de validación y de evaluación ampliada) |

**Tope de la nota: 4.0.** El trabajo se sostiene en el hallazgo alternativo declarado antes del tag del protocolo
(H2), que se cumplió completo.

## Por qué no hay superioridad

La causa no es la potencia: con 10 semillas, el diseño podía detectar diferencias desde `0.0014` m/s. Tampoco es la
varianza entre semillas ni la falta de ajuste. La causa es el **techo de desempeño** que medimos en el E2: con las
mismas cuatro variables de entrada (velocidad, dirección y temperatura de la propia estación), todos los modelos
entrenados extraen prácticamente la misma información. Quedan a `0.011` m/s entre sí, sus residuos están
correlacionados en `0.99` y el 90 % de sus peores ventanas coinciden. Lo que falta para pronosticar mejor no está en
la ventana de entrada.

## Amenazas a la validez

* **Validez interna.** Los benchmarks se corrieron en el E2 y PADRE-TCN en el E3, con el mismo código, los mismos
  datos, las mismas semillas y el mismo presupuesto de ajuste. Parte de las corridas fueron en Colab y parte en el
  PC; el entrenamiento es determinista (la reproducción dio predicciones idénticas), pero las GPU distintas pueden dar
  diferencias numéricas pequeñas entre modelos.
* **Sesgo de selección del diseño.** Se evaluaron N_d = 9 variantes en una partición de desarrollo con una sola
  semilla. La ventaja elegida podría ser en parte ruido; la mitigamos confirmando en otros datos (2025) con 10
  semillas, y la ventaja en los intervalos se mantuvo.
* **Validez de constructo.** El RMSE premia acercarse a la media y castiga poco la subestimación de los extremos; por
  eso reportamos también el Winkler, el sesgo con viento fuerte y la cobertura. El intervalo conformal da una garantía
  marginal: con viento fuerte la cobertura baja a `0.74`.
* **Validez externa.** El modelo se diseñó para el viento horario del Caribe y no se trasladó bien a la potencia de
  turbinas. Los conjuntos adicionales se evaluaron con 2 orígenes, sin reajustar ningún modelo, y sin Chronos ni
  Moirai.
* **Desviaciones del protocolo.** Una sola posterior al tag (STGCN con 3 semillas en dos conjuntos adicionales) y la
  falta de Chronos y Moirai en esos conjuntos. Ninguna cambia el veredicto.

## Limitaciones y riesgos

* **Vientos fuertes.** El pronóstico es un promedio ponderado de las anclas (Proposición 2): si ningún día anterior
  propone un valor alto, el modelo no puede pronosticarlo. En la operación de un parque eólico, ese es justo el caso que
  más importa.
* **Sensibilidad al ruido de la entrada.** Con ruido fuerte se degrada algo más que los benchmarks, porque los
  objetivos de las anclas se toman tal cual de la ventana.
* **Sobreajuste leve.** Se ajusta un poco más al entrenamiento que la TCN (brecha de `0.025` m/s), aunque no empeora
  en la prueba.
* **Riesgo de uso.** Un intervalo al 90 % bien calibrado en promedio puede dar una falsa sensación de seguridad en los
  eventos extremos. Cualquier uso operativo debería acompañarse de la cobertura condicional por régimen.

## Trabajo futuro

1. **Atacar los vientos fuertes con la pérdida, no con la arquitectura:** pronóstico por cuantiles o una penalización
   asimétrica de la subestimación, que es lo que sugirió el E2 y lo que confirma este trabajo.
2. **Anclas que puedan extrapolar:** permitir que la combinación salga del rango de las anclas (por ejemplo, con un
   término de tendencia), sabiendo que se pierde la Proposición 2.
3. **Romper el techo con información nueva:** reanálisis, pronósticos numéricos del tiempo o estaciones vecinas como
   entradas, que es donde está la información que falta.
4. **Completar la evaluación:** Chronos y Moirai en los conjuntos adicionales, la curva de aprendizaje con distintos
   tamaños de muestra y la ablación factorial.

## Declaración de autoría y de uso de herramientas de inteligencia artificial

**Autoría.** Betzaida Ruiz es la autora del trabajo y responsable de las decisiones, la revisión de los resultados y
su interpretación. Yuryen Rollo, oyente del curso, colaboró con la operación del cómputo (PC y Colab), la
verificación de la búsqueda de novedad y la reproducción independiente; se declara así, sin presentarlo como un
tercero ajeno al proyecto.

**Uso de inteligencia artificial.** El trabajo se desarrolló con apoyo de Claude (Anthropic), según lo permitido en el
curso, en:

* la **programación** del modelo original, de las pruebas unitarias, de los scripts de entrenamiento y de análisis
  estadístico, y de la exportación;
* la **búsqueda bibliográfica** automatizada en bases abiertas (los artículos citados son reales y se verificaron con
  su DOI);
* la **redacción** del libro, incluidas las demostraciones.

El mecanismo se escribió desde las ecuaciones de los artículos fuente, sin copiar código de repositorios. Todos los
números del libro salen de los archivos de resultados del repositorio, generados por los scripts que están en él.
La autora revisó el código, las decisiones y los resultados, y es responsable de poder explicar y modificar cualquier
parte del trabajo.

Firma: Betzaida Ruiz · noviembre de 2026
