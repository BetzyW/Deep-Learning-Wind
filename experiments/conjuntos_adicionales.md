# Conjuntos de datos adicionales (sección 8.3 del enunciado del E3)

**Elegidos y descargados el 21-nov-2026, antes de entrenar cualquier modelo en ellos.** No se usan para decisiones de
diseño del modelo original. Los datos no se versionan en git (`datos/` está en `.gitignore`); estas son las
instrucciones exactas para obtenerlos y sus huellas.

| | NOAA ISD-Lite, Kansas | SDWPF (KDD Cup 2022) | ETTh1 |
|---|---|---|---|
| Tarea | Velocidad del viento a 24 h | Potencia activa de turbinas a 24 h | Temperatura del aceite (OT) a 24 h |
| Fuente | NOAA Integrated Surface Database, versión Lite, en AWS Open Data (`s3://noaa-isd-pds/isd-lite/data/{año}/`) | Zhou et al. (2024), *Scientific Data*, doi:10.1038/s41597-024-03427-5; figshare doi:10.6084/m9.figshare.24798654, archivos `sdwpf_245days_v1.csv` y `sdwpf_baidukddcup2022_turb_location.csv` | Zhou et al. (2021), Informer, AAAI; GitHub `zhouhaoyi/ETDataset`, archivo `ETT-small/ETTh1.csv` |
| Licencia | Dominio público (datos del gobierno de EE. UU.) | CC BY 4.0 | CC BY-ND 4.0 (solo se usa para evaluar, sin redistribuir versiones modificadas) |
| Tamaño | 34 estaciones, 2019–2024, horario | 134 turbinas, 245 días, cada 10 min (se agrega a horario) | 1 serie con 7 variables, 17 420 h (jul-2016 a jun-2018) |
| En qué difiere del panel del IDEAM | Clima continental y llano (más viento y menos ciclo de brisa), red más densa y completa, otro país | Otra variable objetivo (potencia, acotada por la curva de la turbina), resolución original de 10 min, un solo parque muy denso | Otro dominio (eléctrico), una sola serie, sin estructura espacial |
| Estructura espacial (para STGCN) | Coordenadas de cada estación | Posición relativa de cada turbina | No hay: STGCN usa un grafo de un solo nodo (se declara) |
| SHA-256 | Listado de los 204 archivos: `058624096c866563e271f6fcecea7ce00992ca7ce77dbf33e787f52291009288`; `isd-history.csv`: `1994747a…6604` | `sdwpf_245days_v1.csv`: `6cf7613a…7667` (MD5 igual al publicado: `8f0cc58c…5e76`); ubicaciones: `15c19cec…f0d9fe` | `f18de3ad…e066` |

**Regla de selección de las estaciones de Kansas (fijada antes de descargar los datos):** todas las estaciones de
`isd-history.csv` con país US, estado KS, identificadores USAF y WBAN válidos, inicio de registro antes del
1-ene-2019 y fin después del 1-ene-2025. Salen 34; se usan todas. El catálogo de NOAA en AWS termina en agosto de
2025, por eso el periodo es 2019–2024.

**Protocolo común en los tres conjuntos** (se detalla en `protocolo.md`): el mismo de los benchmarks, adaptado a la
longitud de cada conjunto. Ventana de 168 h, horizonte de 24 h, partición cronológica con origen móvil y embargo,
escalado con el entrenamiento, el mismo ajuste previo de hiperparámetros para todos los modelos y 10 semillas.
