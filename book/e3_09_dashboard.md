# Dashboard y despliegue

El dashboard predictivo en Dash Plotly y su despliegue en Render (bloques E y F del enunciado) **son opcionales** en
nuestro caso y **no se implementaron**. Concentramos el tiempo en el modelo original y en su validación estadística,
que son los bloques que definen el tope de la nota.

Lo que sí quedó listo, y es la base de un despliegue futuro:

* **Artefacto de inferencia.** PADRE-TCN exportado a ONNX en `artifacts/padre_v6.onnx`, con lote dinámico y dos
  salidas: el pronóstico a 24 h y la dispersión entre anclas, con la que se arma el intervalo al 90 %.
* **Ficha de metadatos** (`artifacts/metadata.json`): columnas y orden de la entrada, ventana, horizonte, media y
  desviación del escalador, arquitectura, hiperparámetros, versiones y huellas SHA-256. Con ella, el modelo se puede
  usar sin el repositorio.
* **Prueba de ida y vuelta** (`artifacts/ida_y_vuelta.json`): en 84 998 ventanas reales, onnxruntime y PyTorch
  difieren como máximo en `2.4e-6` m/s (tolerancia `1e-4`).
* **Costo de inferencia en CPU**, que es lo que importa en un servidor gratuito: `0.98` ms con una ventana y `0.175` ms
  por ventana en lote, con `100 663` parámetros (unos 0.4 MB en ONNX), así que cabría sin cuantizar en el plan gratuito
  de Render.

Una aplicación que lo usara solo tendría que escalar la ventana con la ficha, correr el `.onnx` con onnxruntime y
aplicar el cuantil conformal a la dispersión; en ningún caso reentrenaría el modelo.
