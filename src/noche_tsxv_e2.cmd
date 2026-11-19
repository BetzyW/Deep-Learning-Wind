@echo off
rem Noche del PC de Yuryen: primero timeseries-cv (≈ 5–6 h), luego la cadena del E2 (fase 2). Ambos se retoman.
cd /d %~dp0..
call src\noche_tsxv.cmd
cd /d %~dp0..
call src\noche_e2b.cmd
