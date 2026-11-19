@echo off
rem Noche del PC de Yuryen: solo la fase 2 de TCN (GRU y LSTM corren en Colab). Se detiene a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e2\busqueda\noche_e2b.log
echo == Fase 2 TCN %date% %time% >> %LOG%
%PY% -u src\fase2.py --k 10 --origenes ultimo --hasta 06:00 TCN >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
