# Vigilante de temperatura del PC de Yuryen: cada minuto lee la GPU con nvidia-smi y, si pasa del límite en dos
# lecturas seguidas, detiene los procesos de Python del entrenamiento (los resultados ya guardados no se pierden y
# las corridas se retoman al relanzar). Termina solo cuando ya no queda Python corriendo.
param([int]$Limite = 83, [string]$Log = "resultados\temperatura.log")
$seguidas = 0
while ($true) {
    $py = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
    if (-not $py) { Add-Content $Log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') sin Python corriendo: fin del vigilante"; break }
    $t = [double]((nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) -split "`n")[0]
    $seguidas = if ($t -ge $Limite) { $seguidas + 1 } else { 0 }
    Add-Content $Log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') GPU $t C"
    if ($seguidas -ge 2) {
        $py | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
        Add-Content $Log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') DETENIDO: GPU $t C >= $Limite C dos lecturas seguidas"
        break
    }
    Start-Sleep 60
}
