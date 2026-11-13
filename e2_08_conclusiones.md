# Conclusiones (Entregable 2)

## Resumen estructurado

**Contexto.** El pronóstico de viento a 24 h es clave para integrar la energía eólica, y la literatura reciente
propone muchas arquitecturas de Deep Learning sin compararlas con un protocolo común.

**Objetivo.** Comparar los 10 modelos más relevantes de una revisión sistemática con los modelos base del Entregable 1
y con clásicos fuertes, con el mismo presupuesto de ajuste e inferencia formal.

**Métodos.** Usamos 62 estaciones del IDEAM en el Caribe colombiano y una partición con origen móvil trimestral, embargo
de 24 h y reentrenamiento en cada origen. La arquitectura se buscó en dos fases y cada modelo se corrió con 6 semillas.
Comparamos los modelos con Wilcoxon por estación y Holm, Diebold–Mariano, Model Confidence Set, Friedman–Nemenyi y
TOST, con margen de relevancia δ = 0.02 m/s. Todo quedó preregistrado.

**Resultados.** Siete modelos superan a la regresión ridge con significancia, pero por `0.0073` m/s como máximo
(RMSE `0.692` frente a `0.700`). Los preentrenados congelados (Chronos, Moirai) fueron claramente peores. Todos los
modelos subestiman los vientos de 6 m/s o más en unos `1.2` m/s, y el mejor modelo ya cubre el 98–99 % del margen
alcanzable.

**Conclusiones.** Con la historia de cada estación como única entrada, la arquitectura casi no importa: un modelo
lineal queda a milésimas del mejor. Mejorar exige información nueva o una pérdida que corrija la subestimación de los
extremos, no más capacidad.

## Conclusiones

1. **Ningún modelo del Top 10 supera a Ridge con relevancia práctica (H1 se rechaza).** Las ganancias son
   estadísticamente reales pero menores que la mitad de δ, y el estudio tenía potencia para detectar diferencias
   mucho menores.
2. **Los preentrenados, adaptados con el modelo base congelado y un resumen por promedio, no son equivalentes a los
   modelos desde cero (H2 se rechaza).** La conclusión se limita a esa adaptación: su uso *zero-shot* original queda
   como comparación pendiente.
3. **La subestimación de los vientos fuertes es de toda la familia, no solo de DLinear (H3 se cumple).** Es la
   consecuencia esperable de entrenar con error cuadrático y el problema más claro que dejan abierto todos los modelos.
4. **El límite está en los datos.** El techo casi alcanzado, los errores compartidos entre modelos y el peso de la
   época del año frente al del modelo apuntan a lo mismo.
5. **La búsqueda de arquitectura aportó poco** frente a su costo (`44.6` horas de GPU): casi todos los finalistas
   fueron equivalentes, y el cribado corto no anticipó el orden de la confirmación.
6. **El rigor del protocolo es lo que permite afirmar todo lo anterior.** Hicimos todo lo siguiente:
   - Preregistramos las preguntas, las métricas y el análisis, y registramos cada enmienda.
   - Probamos que no hubiera fuga de información.
   - Reentrenamos los modelos en cada origen.
   - Tratamos la estación como unidad de independencia.
   - Reprodujimos de forma independiente las 520 corridas.

   Con un protocolo más débil, las milésimas de diferencia se habrían leído como una victoria del Deep Learning.

## Trabajo futuro (Entregable 3)

- Proponemos un modelo con una pérdida asimétrica o por cuantiles que corrija la subestimación de los extremos, medido
  contra Ridge y contra el mejor modelo de este entregable con el mismo protocolo y al menos 10 semillas.
- Queremos incorporar información espacial (estaciones vecinas) o exógena, que es donde queda margen.
- Falta comparar Chronos y Moirai en su modo *zero-shot* original para separar el modelo de nuestra adaptación.
