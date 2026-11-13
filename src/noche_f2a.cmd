@echo off
rem Noche del PC de Yuryen: ruta F2a con GPU (validacion para intervalos, semillas 1001-1002, robustez, interpretabilidad). Se detiene a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\e2\f2a mkdir resultados\e2\f2a
set LOG=resultados\e2\f2a\noche_f2a.log
echo == F2a GPU %date% %time% >> %LOG%
%PY% -u src\f2a_gpu.py --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
