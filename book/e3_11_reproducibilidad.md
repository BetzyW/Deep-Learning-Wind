# Reproducibilidad

## Entorno

* **Contenedor.** El `Dockerfile` de la raíz arma un entorno en CPU (Python 3.11, torch 2.14.1 sin CUDA y las
  versiones exactas de `requirements.txt` y `book/requirements.txt`). Por defecto corre las pruebas unitarias y
  construye el libro: `docker build -t viento-dl .` y `docker run --rm viento-dl`. Con los datos y las predicciones
  montados como volúmenes corre también el análisis estadístico. *No pudimos construir la imagen en el equipo del
  proyecto, que no tiene Docker; la construcción se verifica en la revisión final.*
* **Integración continua.** `.github/workflows/pruebas.yml` corre `pytest` en cada push, con torch en CPU. Las pruebas
  usan paneles sintéticos; la del panel real se omite si los datos no están.
* **Libro.** Jupyter Book 1 (formato clásico): `pip install -r book/requirements.txt` y `jupyter-book build book/`.
  Se construye sin advertencias, y GitHub Actions lo publica con `--warningiserror`.

**Versiones y hardware de los entrenamientos.**

| Elemento | Valor |
|---|---|
| Sistema | Windows 11 Pro; Python 3.11.16 |
| Deep Learning | torch 2.14.1 con CUDA 12.6 (en Colab, el torch de Colab con CUDA 13.0) |
| Análisis | numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, statsmodels 0.15.0, scikit-learn 1.9.1, arch 8.0.0 |
| Exportación | onnx y onnxruntime (versiones en `artifacts/metadata.json`) |
| PC del proyecto | AMD Ryzen 7 8700F (8 núcleos), 32 GB de RAM, NVIDIA GeForce RTX 4060 Ti de 8 GB |
| Colab Pro+ | GPU A100, L4 o T4 según disponibilidad; cada corrida registra su equipo en `resultados/e2/busqueda/hardware.jsonl` |

## Semillas y determinismo

* **Semillas de las corridas finales:** 1000–1009, las mismas para todos los modelos, para tener pares estrictos por
  semilla y origen. Diseño del modelo original: semilla 2001. Ajuste de hiperparámetros: semilla 2000 para el
  entrenamiento y semilla 7 para generar las 20 configuraciones. Muestras: semilla 0 (interpretabilidad) y 11
  (estaciones de Kansas).
* **Determinismo:** `fijar_semilla` (`src/modelos_e2.py`) fija las semillas de Python, NumPy y PyTorch (CPU y CUDA),
  con `cudnn.deterministic = True` y `cudnn.benchmark = False`. Se permite TF32 en las multiplicaciones de matrices.
* **Evidencia:** la segunda ronda de diseño reprodujo exactamente los 21 RMSE de la primera, y el reentrenamiento de
  PADRE-TCN con las semillas 1001 y 1002, tres días después, dio predicciones idénticas a las confirmatorias
  (diferencia máxima `0`). Entre GPU distintas (PC y Colab) puede haber diferencias numéricas pequeñas; por eso cada
  modelo se corrió completo en un mismo tipo de equipo dentro de cada conjunto.

## Tabla maestra de experimentos

`resultados/e3/tabla_maestra.csv` (generada con `src/tabla_maestra.py`) tiene **una fila por corrida, 1 452 en total**:
el modelo original, su ablación, los 13 modelos del benchmark (Top 10 y modelos base del E1) y los baselines, en el
panel del IDEAM y en los tres conjuntos adicionales. Cada fila trae el conjunto, el modelo, el grupo, el origen, la
semilla, el número de ventanas, el RMSE, el MAE, los RMSE de parada y de entrenamiento, las épocas, los segundos de
entrenamiento y los parámetros; en el panel del IDEAM, también los FLOPs y la latencia en CPU.

| Conjunto | Corridas | Horas de entrenamiento con GPU |
|---|---|---|
| Panel del IDEAM (benchmarks, PADRE-TCN y ablación) | 744 | `36.7` |
| ETTh1 | 252 | `1.1` |
| SDWPF | 232 | `20.4` |
| NOAA Kansas | 224 | `9.0` |
| **Total de las corridas finales** | **1 452** | **`67.4`** |

*Los baselines (Ridge, gradient boosting y los triviales) se ajustan en segundos en CPU. Las horas no incluyen la
búsqueda de arquitectura ni el ajuste de hiperparámetros del E2.*

**Consumo energético estimado.** Las 67.4 horas de entrenamiento de las corridas finales, a la potencia máxima de la
GPU del PC (165 W), darían a lo sumo unos **11 kWh**; en Colab, las L4 y T4 consumen menos de la mitad (unos 70 W), así
que el consumo real es menor. Es una cota superior, sin contar la CPU ni la búsqueda del E2.

## Lista de verificación de reproducibilidad

| Elemento | Dónde |
|---|---|
| Versiones de software y hardware | Tabla de este capítulo; `resultados/e2/busqueda/hardware.jsonl` por corrida |
| Semillas | 1000–1009 en todas las corridas finales; las demás, arriba |
| Configuración de determinismo | `fijar_semilla` en `src/modelos_e2.py` |
| Rangos de hiperparámetros | Las mismas 20 configuraciones para todos: tasa de aprendizaje entre `1.0e-4` y `3.0e-3`, decaimiento en {0, `1e-5`, `1e-4`} y lote en {512, 1 024, 2 048}; `resultados/e3/PADRE_hp.csv` y `resultados/e2/busqueda/*_hp.csv` |
| Número de ejecuciones | 10 semillas × 4 orígenes en el panel (40 por modelo); 10 × 2 en los conjuntos adicionales (3 × 2 en STGCN e Informer en SDWPF y NOAA); tabla maestra |
| Medidas de dispersión | Media ± DE entre semillas, IC 95 % por bootstrap y coeficiente de variación (capítulos de validación y de evaluación ampliada) |
| Protocolo y estándar | `experiments/protocolo.md` y `estandar.md`, tags `e3-protocolo` y `e3-estandar` (24-nov-2026) |
| Datos | IDEAM (datos.gov.co, CC BY-SA 4.0) con el ETL del proyecto de ML; ETTh1, SDWPF y NOAA con fuente, licencia y SHA-256 en `experiments/conjuntos_adicionales.md` y `experiments/e3_sha256.csv`; `src/e3_descargar.py` los baja y los verifica |
| Código | `https://github.com/BetzyW/Deep-Learning-Wind` (rama `e3`) |
| Tiempo de cómputo | Tabla de arriba (67.4 h de GPU en las corridas finales) |
| Consumo energético | Cota superior de unos 11 kWh |
| Modelo exportado | `artifacts/padre_v6.onnx`, `metadata.json` e `ida_y_vuelta.json` |
| Identificador persistente (DOI) | Pendiente: requiere enlazar el repositorio con Zenodo y publicar una versión |
| Reproducción independiente | Pendiente: Yuryen Rollo corre el libro desde cero en el contenedor y regenera la tabla principal; se reportará la discrepancia máxima |

## Referencias

Todas las referencias del libro están en `book/references.bib` (BibTeX) y en la página de bibliografía.
