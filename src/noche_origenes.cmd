@echo off
rem Noche del PC de Yuryen: validacion con origen movil de la seleccion de arquitectura. Para a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e2\busqueda\noche_origenes.log
echo == Validacion origen movil %date% %time% >> %LOG%
%PY% -u src\validacion_origenes.py TCN Transformer --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
