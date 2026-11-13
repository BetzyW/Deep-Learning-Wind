@echo off
rem Noche del PC de Yuryen: reproduccion independiente (reentrena en el PC corridas hechas en Colab). Se detiene a las 06:00.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
if not exist resultados\e2\reproduccion mkdir resultados\e2\reproduccion
set LOG=resultados\e2\reproduccion\noche_reproduccion.log
echo == Reproduccion %date% %time% >> %LOG%
%PY% -u src\reproducir.py --reentrenar --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
