@echo off
rem Noche del PC: corridas de diseno del modelo original (E3). Para a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\e3 mkdir resultados\e3
set LOG=resultados\e3\noche_diseno.log
echo == Diseno PADRE %date% %time% >> %LOG%
%PY% -u -W ignore src\diseno_padre.py %1 %2 --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
