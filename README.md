# Pronóstico de viento a 24 h con Deep Learning

Entregable 1 · Deep Learning · Universidad del Norte. Autora: Betzaida Ruiz.

## Reproducir

1. `pip install -r requirements.txt` (PyTorch con CUDA: ver pytorch.org).
2. Copiar `panel_multivariado_causal.csv` en `datos/` (se genera con el ETL del proyecto de Machine Learning, que se cita en el libro).
3. `python src/ficha_panel.py` → hash SHA-256 y ficha del panel.
4. `python src/datos.py` → ventanas, partición 2020–2023 / 2024 / 2025 y RMSE de persistencia.
5. `python src/a_notebook.py src/eda.py 02_eda.ipynb` y ejecutar el notebook → EDA.
6. `literatura/` → búsqueda reproducible (Semantic Scholar y arXiv) y decisiones de elegibilidad.

La configuración de los datos está en `configs/datos.json`.
