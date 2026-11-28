# Modelos benchmark y mejores configuraciones

Los 10 modelos del Top 10 (Transformer, KAN, TCN, STGCN, Chronos, GRU, Informer, DLinear, Moirai y PatchTST) y los
baselines fuertes (Ridge y gradient boosting) se eligieron con la revisión sistemática del E2. Su definición, su
fuente y su implementación están en el capítulo de modelos del E2; aquí reportamos sus mejores configuraciones, el
criterio con que se eligieron y la exportación (bloque B del enunciado).

## Presupuesto de ajuste y criterio de selección

* **Mismo presupuesto para todos, incluido PADRE-TCN.** Cada modelo probó la misma lista de 20 configuraciones de
  (tasa de aprendizaje, decaimiento, lote), generada con la semilla 7, en el origen de validación 2024-10, con el 10 %
  de las ventanas de ajuste, 20 épocas y paciencia 3. Los modelos con búsqueda de arquitectura (Transformer, KAN, TCN,
  GRU y los cabezales de Chronos y Moirai) la hicieron antes, en dos fases, con el mismo procedimiento del E2.
* **Criterio.** Se elige la configuración con **menor RMSE de validación**, nunca con la prueba. Cada configuración
  elegida se guardó con su fecha (`{modelo}_configuracion.json`) antes de abrir la prueba. En todos los modelos, la
  elegida es la mejor de las 20.
* **Reentrenamiento final, igual para todos.** Con la configuración fija, cada modelo se reentrena desde cero en cada
  uno de los 4 orígenes de prueba de 2025, con todo el entrenamiento disponible hasta el origen, un máximo de 50
  épocas y parada temprana con paciencia 5 sobre el bloque de parada; se conservan los pesos de la mejor época. Esa
  política es la misma para PADRE-TCN.

## Tabla maestra de mejores configuraciones

```{include} tablas/e3_seleccion.md
```

*RMSE de validación: el de la configuración elegida en el ajuste (10 % de las ventanas, 20 épocas), por eso es mayor
que el de prueba. RMSE de prueba: media de las 10 semillas, cada una promediada en los 4 orígenes; el IC 95 % es el de
la media con una t de 9 grados de libertad. Parámetros, FLOPs y latencia con `src/costo_inferencia.py --e3`, en la
misma CPU y la misma sesión para todos; en Chronos y Moirai se mide solo el cabezal. "La del artículo": modelos sin
búsqueda de arquitectura, que usan la del trabajo original.*

**Mejor modelo global.** El **Transformer** tiene el menor RMSE de prueba (`0.7189` m/s) y es el modelo con mayor p
(`1.00`) del Model Confidence Set al 90 %, así que pertenece al conjunto de los mejores. Ese conjunto incluye además a
otros 9 modelos, PADRE-TCN entre ellos (capítulo de validación).

**Sesgo del ganador.** La arquitectura de cada modelo se eligió con un solo origen de validación, y el mínimo de varios
valores ruidosos es optimista. En el E2 lo medimos repitiendo la mejor, la elegida y la peor arquitectura en los otros
tres orígenes de validación: la ventaja de la mejor bajó en promedio de `0.011` a `0.008` m/s (cerca de un tercio era
suerte) y se mantuvo en 13 de 16 contrastes. Por eso el RMSE de validación no se usa como estimación del desempeño: el
desempeño se mide solo en la prueba.

## Exportación y serialización

* **Modelo original.** PADRE-TCN está exportado a ONNX en `artifacts/padre_v6.onnx`, con `metadata.json` (versiones,
  semilla, hiperparámetros, estadísticas del escalado, esquema de entrada y huellas) y la prueba de ida y vuelta en
  `artifacts/ida_y_vuelta.json`: en 84 998 ventanas reales, la diferencia máxima entre onnxruntime y PyTorch es
  `2.4e-6` m/s (tolerancia `1e-4`) y el RMSE del modelo exportado es el mismo del modelo en memoria.
* **Benchmarks.** De cada modelo se guardaron los pesos de la primera semilla en el último origen de prueba
  (`resultados/e2/finales/{modelo}/2025-10-01_1000.pt`), con su configuración. **No los exportamos a ONNX**: la
  exportación de los benchmarks servía al dashboard, que en nuestro caso es opcional y no se implementó (capítulo 9).
* **Checkpoint.** Se guardan los pesos y la configuración completa; el estado del optimizador no se guardó, porque
  ningún modelo se reanuda desde un punto intermedio: cada corrida se repite desde cero con su semilla y da el mismo
  resultado (capítulo de reproducibilidad).
* **Cuantización.** No hace falta: el ONNX de PADRE-TCN pesa unos 0.4 MB.

## Costo de inferencia

Los parámetros, los FLOPs y la latencia por ventana en CPU están en la tabla de arriba y, con lote de 1 024 ventanas,
en la tabla de equidad del capítulo de validación. Reportamos la mediana de 50 repeticiones con lote 1 y de 10 con
lote 1 024; no medimos el percentil 95 ni la memoria pico en CPU. PADRE-TCN cuesta lo mismo que la TCN de la que parte,
menos que el Transformer y mucho menos que STGCN e Informer.
