@echo off
rem Noche del PC (E3): LSTM, GRU y Transformer en SDWPF y NOAA (Colab queda para STGCN e Informer). Para a las 06:00.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set PY=C:\Users\USER\miniconda3\envs\dl_env\python.exe
set LOG=resultados\e3\noche_e3_pesados.log
echo == Noche E3 pesados %date% %time% >> %LOG%
%PY% -u -W ignore src\e3_adicionales.py sdwpf LSTM GRU Transformer --hasta 06:00 >> %LOG% 2>&1
%PY% -u -W ignore src\e3_adicionales.py noaa_kansas LSTM GRU Transformer --hasta 06:00 >> %LOG% 2>&1
echo == Fin %date% %time% >> %LOG%
