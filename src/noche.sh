#!/usr/bin/env bash
# Cadena completa: ajuste → corridas definitivas → sensibilidad con presión → notebooks 04 y 05 ejecutados.
set -e
DL=/c/Users/USER/miniconda3/envs/dl_env/python.exe
ML=/c/Users/USER/miniconda3/envs/ml_env/python.exe
echo "== Ajuste $(date)";        $DL src/ajuste.py 20
echo "== Corridas $(date)";      $DL src/corridas.py 6
echo "== Sensibilidad $(date)";  $DL src/corridas.py 3 configs/datos_con_presion.json
echo "== Notebooks $(date)"
$ML src/a_notebook.py src/implementacion.py 04_modelos.ipynb
$ML src/a_notebook.py src/benchmark.py 05_benchmark.ipynb
for nb in 04_modelos.ipynb 05_benchmark.ipynb; do
  $ML -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $nb
done
echo "== Listo $(date)"
