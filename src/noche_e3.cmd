@echo off
rem Noche del PC (E3): ajuste previo de PADRE y luego modelos ligeros en los conjuntos adicionales. Para a las 06:00.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e3\noche_e3.log
echo == Noche E3 %date% %time% >> %LOG%
%PY% -u -W ignore src\ajuste_padre.py --hasta 06:00 >> %LOG% 2>&1
%PY% -u -W ignore src\e3_adicionales.py todos --ligeros --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
