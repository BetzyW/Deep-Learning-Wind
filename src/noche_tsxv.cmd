@echo off
rem Búsqueda timeseries-cv (sección 2.21), solo de noche: se detiene a las 06:00 y se retoma la noche siguiente.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\tsxv mkdir resultados\tsxv
set LOG=resultados\tsxv\noche_tsxv.log
echo == timeseries-cv %date% %time% >> %LOG%
%PY% -u src\timeseries_cv.py --hasta 06:00 >> %LOG% 2>&1
%PY% -u src\timeseries_cv.py --resumen >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
