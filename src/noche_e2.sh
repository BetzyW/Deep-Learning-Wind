#!/usr/bin/env bash
# Cadena nocturna del E2 en el PC (sin gastar unidades de Colab). Espera a que termine la búsqueda de la TCN que ya
# corre, luego: ajuste previo de los modelos con arquitectura publicada (DLinear, PatchTST) y fase 2 (k = 10, último
# origen) de MLP, CNN, TCN, GRU y LSTM. Todo se reanuda donde quedó si el PC se apaga.
DL=/c/Users/USER/miniconda3/envs/dl_env/python.exe
LOG=resultados/e2/busqueda/noche_e2.log
export PYTHONIOENCODING=utf-8
while tasklist //FI "PID eq $1" 2>/dev/null | grep -q python; do sleep 60; done
echo "== Publicados (ajuste previo) $(date)" >> $LOG
$DL -u src/busqueda_arq.py DLinear PatchTST >> $LOG 2>&1
echo "== Fase 2 $(date)" >> $LOG
$DL -u src/fase2.py --k 10 --origenes ultimo MLP CNN TCN GRU LSTM >> $LOG 2>&1
echo "== Listo $(date)" >> $LOG
