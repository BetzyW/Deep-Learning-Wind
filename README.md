# Pronóstico de viento a 24 h con Deep Learning

Entregable 1 · Deep Learning · Universidad del Norte. Autora: Betzaida Ruiz.

## Entregable 3 (rama `e3`)

Modelo original **PADRE-TCN** (regresión por diferencias con anclas diarias causales), validación estadística
pre-registrada frente a 12 rivales y evaluación en tres conjuntos adicionales. Resumen: no supera a la familia en RMSE
(tope 4.0), es equivalente a la TCN y da los mejores intervalos de predicción al 90 % de la familia.

| Qué | Dónde |
|---|---|
| Libro (Jupyter Book 1, capítulos 0–11 del E3) | `book/` → `pip install -r book/requirements.txt` y `jupyter-book build book/` |
| Modelo original y pruebas | `src/original_model.py`, `tests/test_original_model.py` (`pytest tests`) |
| Estándar, protocolo (tags `e3-estandar`, `e3-protocolo`), bitácora y procedencia | `experiments/` |
| Modelo exportado | `artifacts/padre_v6.onnx`, `metadata.json`, `ida_y_vuelta.json` |
| Tabla maestra de experimentos (1 452 corridas) | `resultados/e3/tabla_maestra.csv` |
| Ficha del modelo y declaración de los datos | `MODEL_CARD.md`, `DATA_STATEMENT.md` |
| Entorno reproducible | `Dockerfile`; CI con pytest en `.github/workflows/pruebas.yml` |

**Reproducir el E3** (con los datos en `datos/`): `src/e3_descargar.py` y `src/e3_conjuntos.py` (conjuntos
adicionales), `src/ajuste_padre.py`, `src/confirmatorias_padre.py V6 V3 V4 V5 V1` y `src/e3_adicionales.py` (GPU);
luego `src/estadistica_e3.py`, `src/adicionales_e3.py`, `src/adecuacion_padre.py`, `src/evaluacion_ampliada.py`,
`src/interpretabilidad_padre.py`, `src/tabla_maestra.py` y `src/tablas_e3.py`, y construir el libro.

Licencia del código: MIT (`LICENSE`); del libro: CC BY 4.0. Cómo citar: `CITATION.cff`.

## Reproducir

1. `pip install -r requirements.txt` (PyTorch con CUDA: ver pytorch.org).
2. Obtener el panel de datos (ver «Datos» más abajo) y copiarlo en `datos/`.
3. `python src/ficha_panel.py` → hash SHA-256 y ficha del panel.
4. `python src/datos.py` → ventanas, partición 2020–2023 / 2024 / 2025 y RMSE de persistencia.
5. `python src/a_notebook.py src/eda.py 02_eda.ipynb` y ejecutar el notebook → EDA.
6. `python src/piloto.py`, `python src/ajuste.py 20` y `python src/corridas.py 6` → piloto, ajuste y corridas en GPU.
7. `python src/corridas.py 3 configs/datos_con_presion.json` → sensibilidad con presión (solo validación).
8. Convertir `src/implementacion.py` y `src/benchmark.py` en `04_modelos.ipynb` y `05_benchmark.ipynb` con `src/a_notebook.py` y ejecutarlos.
9. `cd literatura && python tamizaje.py` → une Scopus, Web of Science, Semantic Scholar y arXiv, quita duplicados, aplica el tamizaje y da los conteos PRISMA.

La configuración de los datos está en `configs/datos.json`.

## Datos

Mediciones horarias del IDEAM, publicadas en el Portal Nacional de Datos Abiertos con licencia CC BY-SA 4.0.

| Variable | Conjunto en datos.gov.co |
|---|---|
| Velocidad del viento | https://www.datos.gov.co/d/sgfv-3yp8 |
| Dirección del viento | https://www.datos.gov.co/d/kiw7-v9ta |
| Temperatura ambiente del aire | https://www.datos.gov.co/d/sbwg-7ju4 |
| Presión atmosférica | https://www.datos.gov.co/d/62tk-nxj5 |
| Catálogo nacional de estaciones del IDEAM | https://www.datos.gov.co/d/hp9r-jxuu |

1. Descargar de cada conjunto los CSV de 2020 a 2025 de los departamentos del Caribe (Atlántico, Bolívar, Cesar,
   Córdoba, La Guajira, Magdalena y Sucre), en subcarpetas `VELOCIDAD DEL VIENTO`, `DIRECCION DEL VIENTO`,
   `TEMPERATURA`, `PRESION` y `CATALOGO estaciones`.
2. Clonar el repositorio del proyecto de Machine Learning, que contiene el ETL:
   `git clone https://github.com/yuryenr-ui/entregable1-viento`
3. Generar el panel con la limpieza causal (cada lectura se limpia solo con información anterior a ella):
   `python entregable1-viento/scripts/procesar_todo.py "<carpeta de descargas>" --causal`
4. Copiar el `panel_multivariado_causal.csv` resultante en `datos/` y comprobar su huella con
   `python src/ficha_panel.py` (SHA-256 `dd0fdfad…c12fb`, ver el capítulo 1).
