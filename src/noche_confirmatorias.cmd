@echo off
rem Noche del PC (E3): corridas confirmatorias de PADRE-TCN (V6, 10 semillas x 4 origenes de prueba). Para a las 06:00.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e3\noche_confirmatorias.log
echo == Confirmatorias PADRE %date% %time% >> %LOG%
%PY% -u -W ignore src\confirmatorias_padre.py V6 --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
