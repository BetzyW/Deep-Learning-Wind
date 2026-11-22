# Conclusiones

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
(RMSE `0.692` frente a `0.700`). Los preentrenados (Chronos, Moirai) fueron claramente peores, con nuestro cabezal y también
sin entrenar. Todos los
modelos subestiman los vientos de 6 m/s o más en unos `1.2` m/s, y el mejor modelo ya cubre el 98–99 % del margen
alcanzable.

**Conclusiones.** Con la historia de cada estación como única entrada, la arquitectura casi no importa: un modelo
lineal queda a milésimas del mejor. Mejorar exige información nueva o una pérdida que corrija la subestimación de los
extremos, no más capacidad.

## Conclusiones

1. **Ningún modelo del Top 10 supera a Ridge con relevancia práctica (H1 se rechaza).** Las ganancias son
   estadísticamente reales pero menores que la mitad de δ, y el estudio tenía potencia para detectar diferencias
   mucho menores.
2. **Los preentrenados no son equivalentes a los modelos desde cero (H2 se rechaza).** Nuestra adaptación (modelo
   base congelado, resumen por promedio y cabezal) los empeoró, porque sin entrenar son mejores. Pero aun así Moirai
   queda `0.038` m/s por encima del mejor modelo desde cero. Lo que sí hace bien Moirai sin entrenar es dar intervalos
   calibrados y subestimar menos los vientos fuertes.
3. **La subestimación de los vientos fuertes es de toda la familia, no solo de DLinear (H3 se cumple).** Es la
   consecuencia esperable de entrenar con error cuadrático y el problema más claro que dejan abierto todos los modelos.
4. **El límite está en los datos.** El techo casi alcanzado, los errores compartidos entre modelos y el peso de la
   época del año frente al del modelo apuntan a lo mismo.
5. **Lo que sí importa es ver el ciclo diario.** Elegir una arquitectura u otra mueve el error en milésimas, pero
   quitarle a la TCN la dilatación (y con ella el acceso a los días anteriores) lo sube en `0.025` m/s, más que δ.
   Los gradientes integrados coinciden: los modelos usan el nivel reciente y la misma hora de días anteriores. Esa
   dependencia los hace frágiles ante horas faltantes y cambios de nivel, y sus intervalos cubren de menos cuando
   pronostican viento alto.
6. **La búsqueda de arquitectura aportó poco** frente a su costo (`44.6` horas de GPU): casi todos los finalistas
   fueron equivalentes, y el cribado corto no anticipó el orden de la confirmación. Elegir con un solo origen de
   validación no sesgó la selección: en los otros tres trimestres, la elegida quedó a δ o menos de la mejor en 8 de
   9 modelos.
7. **El rigor del protocolo es lo que permite afirmar todo lo anterior.** Hicimos todo lo siguiente:
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
- Moirai sin entrenar da intervalos calibrados y subestima menos los vientos fuertes que los modelos entrenados con
  error cuadrático. Un pronóstico probabilístico, por cuantiles o por distribución, es un camino directo para el E3.
