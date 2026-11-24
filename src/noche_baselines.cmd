@echo off
rem Baselines de SDWPF y NOAA Kansas (CPU, ml_env), en paralelo con la GPU.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\ml_env\python.exe
set LOG=resultados\e3\noche_baselines.log
%PY% -u -W ignore src\e3_adicionales.py sdwpf --baselines >> %LOG% 2>&1
%PY% -u -W ignore src\e3_adicionales.py noaa_kansas --baselines >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
