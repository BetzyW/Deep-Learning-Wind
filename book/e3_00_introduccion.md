# Entregable 3: introducción

## Resumen ejecutivo

Pronosticamos la velocidad del viento a 24 h en 62 estaciones del IDEAM en el Caribe colombiano. Proponemos
**PADRE-TCN**, un modelo original que predice la diferencia respecto de la misma hora de los seis días anteriores
(anclas), transferido de la regresión por diferencias de la química y de las redes gemelas, con una diferencia
**antisimétrica por construcción** (demostrada y verificada con error exactamente 0) y con la dispersión entre anclas
como medida de incertidumbre. Lo comparamos, con un protocolo pre-registrado, contra 12 rivales (el Top 10 del E2,
Ridge y gradient boosting), con 10 semillas, 4 trimestres de prueba de 2025, tres pruebas por contraste y Holm.
**No supera a la familia en RMSE** (`0.7239` m/s; el Transformer es `0.005` m/s mejor, por debajo del margen
δ = `0.02`), así que el tope de la nota es 4.0. **Sí cumple el hallazgo alternativo pre-registrado**: es equivalente a
la TCN en RMSE (TOST) y sus intervalos al 90 % son los mejores de la familia (Winkler `2.991` frente a `3.153`, unos
5 % mejores, en los 12 contrastes), una mejora que la ablación atribuye a la dispersión entre anclas. También es el
modelo menos sensible a los hiperparámetros. En tres conjuntos adicionales no destaca, y la subestimación de los
vientos fuertes del paper guía persiste (`−1.21` m/s).

## Problema y motivación

El pronóstico del viento a un día sirve para planear la operación de los parques eólicos de La Guajira y la
seguridad de puertos y aeropuertos del Caribe. En el E2 comparamos los 10 modelos más representativos de la
literatura y encontramos dos cosas: todos quedan en una franja de `0.011` m/s (un techo de desempeño con estas
entradas) y todos subestiman los vientos fuertes, como el paper guía (DLinear). El E3 busca un mecanismo nuevo que,
aunque no rompa el techo, aporte algo que los demás no tienen.

## Contribuciones

1. **Un modelo original**, PADRE-TCN: la regresión por diferencias entre pares, llevada de la química (PADRE) y de la
   física (TNNR) al pronóstico de series de tiempo con anclas temporales causales, un codificador compartido, una
   fusión aprendida y una diferencia antisimétrica exacta.
2. **Dos propiedades demostradas** (antisimetría exacta, también en coma flotante, y pronóstico acotado por las
   anclas), verificadas en el modelo entrenado.
3. **Una validación estadística completa y pre-registrada**: familia de 12 contrastes, tres pruebas, Holm, potencia,
   MCS, ablación de cinco componentes, tres conjuntos adicionales y los estudios de robustez del enunciado.
4. **Un resultado honesto**: el modelo no supera a la familia en RMSE, y la evidencia indica por qué (el techo de las
   entradas, no la falta de potencia).
5. **Intervalos de predicción mejores que los de toda la familia**, con un mecanismo interpretable: cuánto discrepan
   entre sí los días análogos.

## Objetivos

* **General:** diseñar, implementar y validar un modelo neuronal original para el pronóstico del viento a 24 h, y
  determinar con rigor estadístico si supera a los mejores modelos de la literatura.
* **Específicos:** demostrar una propiedad del mecanismo; compararlo con la familia completa con un protocolo
  pre-registrado y potencia suficiente; evaluarlo en tres conjuntos de otra naturaleza; analizar su robustez,
  calibración, eficiencia e interpretabilidad; y exportarlo para inferencia.

## Mapa del Entregable 3

| Capítulo | Contenido |
|---|---|
| 0. Introducción | Este capítulo |
| 1. Trazabilidad de correcciones | Correcciones del E2, ranking antes y después, auditoría de fuga y congelamiento de los benchmarks |
| 2. Datos | EDA, auditoría y ficha de los datos |
| 3. Modelos benchmark | Top 10, baselines, mejores configuraciones y exportación |
| 4. Literatura y novedad | Búsqueda de novedad, matriz y declaración |
| 5. Modelo original: diseño | Limitación, fuente, hipótesis, formulación, pseudocódigo, demostración, complejidad y bitácora |
| 6. Modelo original: implementación | Código, pruebas unitarias, ablación e interpretabilidad |
| 7. Protocolo y validación estadística | Protocolo, equidad, potencia, pruebas, H1, H2 y veredicto R3 |
| 8. Evaluación ampliada y robustez | Conjuntos adicionales, sensibilidad, perturbaciones, calibración, Pareto y errores |
| 9. Dashboard y despliegue | Opcional; artefactos de inferencia listos |
| 10. Conclusiones y limitaciones | Hallazgos, veredicto, amenazas a la validez, trabajo futuro y declaración de IA |
| 11. Reproducibilidad | Entorno, semillas, tabla maestra y lista de verificación |

**Dashboard:** no aplica (opcional en nuestro caso; ver el capítulo 9).
