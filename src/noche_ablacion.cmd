@echo off
rem Noche del PC de Yuryen: estudio de ablacion de TCN (enmienda 7). Se detiene a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\e2\ablacion mkdir resultados\e2\ablacion
set LOG=resultados\e2\ablacion\noche_ablacion.log
echo == Ablacion TCN %date% %time% >> %LOG%
%PY% -u src\ablacion.py --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
