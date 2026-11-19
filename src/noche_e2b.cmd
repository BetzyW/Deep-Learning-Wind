@echo off
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e2\busqueda\noche_e2b.log
echo == Fase 1 TCN + ajuste previo DLinear PatchTST %date% %time% >> %LOG%
%PY% -u src\busqueda_arq.py TCN DLinear PatchTST >> %LOG% 2>&1
for %%m in (MLP CNN TCN GRU LSTM) do (
  echo == Fase 2 %%m %date% %time% >> %LOG%
  %PY% -u src\fase2.py --k 10 --origenes ultimo %%m >> %LOG% 2>&1
)
echo == Listo %date% %time% >> %LOG%
