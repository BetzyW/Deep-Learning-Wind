# Pronóstico de viento a 24 h con Deep Learning

Entregable 1 · Deep Learning · Universidad del Norte. Autora: Betzaida Ruiz.

## Reproducir

1. `pip install -r requirements.txt` (PyTorch con CUDA: ver pytorch.org).
2. Obtener el panel de datos (ver «Datos» más abajo) y copiarlo en `datos/`.
3. `python src/ficha_panel.py` → hash SHA-256 y ficha del panel.
4. `python src/datos.py` → ventanas, partición 2020–2023 / 2024 / 2025 y RMSE de persistencia.
5. `python src/a_notebook.py src/eda.py 02_eda.ipynb` y ejecutar el notebook → EDA.
6. `python src/piloto.py`, `python src/ajuste.py 20` y `python src/corridas.py 6` → piloto, ajuste y corridas en GPU.
7. `python src/corridas.py 3 configs/datos_con_presion.json` → sensibilidad con presión (solo validación).
8. Convertir `src/implementacion.py` y `src/benchmark.py` en `04_modelos.ipynb` y `05_benchmark.ipynb` con `src/a_notebook.py` y ejecutarlos.
9. `literatura/` → búsqueda reproducible (Semantic Scholar y arXiv) y decisiones de elegibilidad.

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
