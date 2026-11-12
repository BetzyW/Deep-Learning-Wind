@echo off
rem Noche del PC de Yuryen: corridas finales de TCN (4 orígenes x 10 semillas). Se detiene a las 06:00 y se retoma.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\e2\finales mkdir resultados\e2\finales
set LOG=resultados\e2\finales\noche_tcn_finales.log
echo == Finales TCN %date% %time% >> %LOG%
%PY% -u src\finales.py --hasta 06:00 TCN >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
