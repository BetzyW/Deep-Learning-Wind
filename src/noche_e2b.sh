#!/usr/bin/env bash
# Noche del 10-nov en el PC (sin unidades de Colab): termina la muestra de la TCN, ajuste previo de DLinear y PatchTST,
# y fase 2 de los modelos baratos primero (MLP, CNN, TCN, GRU, LSTM). Se reanuda donde quedó si el PC se apaga.
DL=/c/Users/USER/miniconda3/envs/dl_env/python.exe
LOG=resultados/e2/busqueda/noche_e2b.log
export PYTHONIOENCODING=utf-8
echo "== Fase 1 TCN + ajuste previo DLinear, PatchTST $(date)" >> $LOG
$DL -u src/busqueda_arq.py TCN DLinear PatchTST >> $LOG 2>&1
for m in MLP CNN TCN GRU LSTM; do
  echo "== Fase 2 $m $(date)" >> $LOG
  $DL -u src/fase2.py --k 10 --origenes ultimo $m >> $LOG 2>&1
done
echo "== Listo $(date)" >> $LOG
