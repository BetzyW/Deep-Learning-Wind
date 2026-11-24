# Conjuntos de datos adicionales (sección 8.3 del enunciado del E3)

**Elegidos y descargados el 21-nov-2026, antes de entrenar cualquier modelo en ellos.** No se usan para decisiones de
diseño del modelo original. Los datos no se versionan en git (`datos/` está en `.gitignore`); estas son las
instrucciones exactas para obtenerlos y sus huellas.

| | NOAA ISD-Lite, Kansas | SDWPF (KDD Cup 2022) | ETTh1 |
|---|---|---|---|
| Tarea | Velocidad del viento a 24 h | Potencia activa de turbinas a 24 h | Temperatura del aceite (OT) a 24 h |
| Fuente | NOAA Integrated Surface Database, versión Lite, en AWS Open Data (`s3://noaa-isd-pds/isd-lite/data/{año}/`) | Zhou et al. (2024), *Scientific Data*, doi:10.1038/s41597-024-03427-5; figshare doi:10.6084/m9.figshare.24798654, archivos `sdwpf_245days_v1.csv` y `sdwpf_baidukddcup2022_turb_location.csv` | Zhou et al. (2021), Informer, AAAI; GitHub `zhouhaoyi/ETDataset`, archivo `ETT-small/ETTh1.csv` |
| Licencia | Dominio público (datos del gobierno de EE. UU.) | CC BY 4.0 | CC BY-ND 4.0 (solo se usa para evaluar, sin redistribuir versiones modificadas) |
| Tamaño | 34 estaciones descargadas (2019–2024); se usan 10 de 2021 a 2024 (ver recorte) | 134 turbinas, 245 días, cada 10 min (se agrega a horario) | 1 serie con 7 variables, 17 420 h (jul-2016 a jun-2018) |
| En qué difiere del panel del IDEAM | Clima continental y llano (más viento y menos ciclo de brisa), red más densa y completa, otro país | Otra variable objetivo (potencia, acotada por la curva de la turbina), resolución original de 10 min, un solo parque muy denso | Otro dominio (eléctrico), una sola serie, sin estructura espacial |
| Estructura espacial (para STGCN) | Coordenadas de cada estación | Posición relativa de cada turbina | No hay: STGCN usa un grafo de un solo nodo (se declara) |
| SHA-256 | Listado de los 204 archivos: `058624096c866563e271f6fcecea7ce00992ca7ce77dbf33e787f52291009288`; `isd-history.csv`: `1994747a…6604` | `sdwpf_245days_v1.csv`: `6cf7613a…7667` (MD5 igual al publicado: `8f0cc58c…5e76`); ubicaciones: `15c19cec…f0d9fe` | `f18de3ad…e066` |

**Regla de selección de las estaciones de Kansas (fijada antes de descargar los datos):** todas las estaciones de
`isd-history.csv` con país US, estado KS, identificadores USAF y WBAN válidos, inicio de registro antes del
1-ene-2019 y fin después del 1-ene-2025. Salen 34 y se descargan todas; por cómputo se usan 10 (ver el recorte más
abajo). El catálogo de NOAA en AWS termina en agosto de 2025, por eso los datos llegan hasta 2024.

**Protocolo común en los tres conjuntos** (se detalla en `protocolo.md`): el mismo de los benchmarks, adaptado a la
longitud de cada conjunto. Ventana de 168 h, horizonte de 24 h, partición cronológica con origen móvil y embargo,
escalado con el entrenamiento, el mismo ajuste previo de hiperparámetros para todos los modelos y 10 semillas.

**Recorte por cómputo (21-nov-2026, antes de entrenar cualquier modelo en estos conjuntos; igual para todos los
modelos).** Las corridas finales del E2 costaron unas 28 horas de GPU en el panel del IDEAM; repetirlas en los tres
conjuntos con 4 orígenes no cabe en el presupuesto de Colab que queda (≈ 300 unidades, que también deben cubrir el
modelo original). Por eso, en los conjuntos adicionales:
* se usan **2 orígenes de prueba** en vez de 4 (los dos últimos trimestres o bloques de cada conjunto);
* en NOAA se usan **10 de las 34 estaciones**, elegidas al azar con semilla 11, y el periodo **2021–2024**;
* cada modelo usa la **configuración elegida en el E2** (hiperparámetros y arquitectura), sin reajustar en cada
  conjunto. Ningún modelo, incluido el original, recibe ajuste en estos conjuntos, así que el presupuesto es el mismo
  para todos;
* se mantienen las **10 semillas** (1000–1009) que exige el enunciado.

**Cómo obtenerlos y correrlos.** `python src/e3_descargar.py` baja los archivos de las fuentes de la tabla y compara
cada uno con su huella en `experiments/e3_sha256.csv` (207 archivos; la lista de las 34 estaciones candidatas está en
`experiments/e3_estaciones_kansas.csv`). Luego `python src/e3_conjuntos.py` arma los paneles y
`python src/e3_adicionales.py {conjunto} {modelos}` hace las corridas (`--baselines` para los clásicos). Chronos y
Moirai se evalúan aquí solo sin entrenar (cero disparos, `preentrenados.py --config`): su versión con cabezal necesita
calcular los embeddings de cada conjunto y no cabe en el presupuesto (se declara). STGCN usa k = min(8, series − 1)
vecinas; en ETTh1, con una sola serie, el grafo tiene un nodo y STGCN queda como una red convolucional temporal.

Ventanas resultantes: NOAA ≈ 300 000 (10 estaciones), SDWPF ≈ 152 000 (134 turbinas, de las que entre 21 y 132 tienen
ventanas completas en cada bloque) y ETTh1 ≈ 17 000.

**Corrección de DLinear (23-nov-2026, después de las primeras corridas; se declara).** En ETTh1, DLinear dio RMSE
`10.4`, cinco veces el de la persistencia (`2.06`), con una predicción media de `−7.0` frente a un valor real de `2.3`.
La causa es de implementación: en el E2 su salida iba directo en las unidades de la serie, mientras que el artículo
(Zeng et al., 2023) trabaja sobre la serie escalada y desescala la predicción. Con el viento (media ≈ 5 m/s y millones
de ventanas) el sesgo de la capa lineal alcanza a aprender el nivel; en ETTh1 (nivel ≈ 16 en el entrenamiento y unas
17 000 ventanas) se detiene a las 6 épocas sin lograrlo. Corregimos DLinear para que desescale su salida con la media
y la desviación del entrenamiento (fijas, sin parámetros nuevos) y repetimos sus 60 corridas en los tres conjuntos.
RMSE medio antes y después: ETTh1 `10.415` → `1.997`; SDWPF `388.3` → `379.5`; NOAA `2.476` → `2.479`. Las corridas
descartadas se conservan en `resultados/e3/adicionales/{conjunto}/descartado_DLinear_salida_cruda/`. Los resultados
del E2 no cambian: allí DLinear se corrió con la versión original.

**Sesgo de nivel en ETTh1 (se reporta tal cual).** MLP, CNN, TCN y KAN también quedan peor que la persistencia en
ETTh1, y no por un error: la serie no es estacionaria (media del objetivo `15.9` en el entrenamiento y `2.3` en el
primer origen de prueba) y estos modelos predicen un nivel desplazado (sesgo medio entre `+1.7` y `+3.8` en el primer
origen). Si se les quita ese sesgo, su RMSE baja a `2.2`–`3.0`. Los modelos que desescalan o se anclan al último
valor (PatchTST con normalización por instancia, PADRE con sus anclas y DLinear corregido) tienen sesgos menores de
`0.5`. No los corregimos porque cambiar su salida sería rediseñarlos, no corregir una desviación de su versión
publicada; lo discutimos como una limitación de los modelos sin normalización ante cambios de nivel.

**Recorte de Informer (23-nov-2026, antes de correr Informer en el segundo origen de SDWPF y en NOAA; se declara).**
En Colab, Informer tardó unos 50 minutos por corrida en SDWPF. Completar sus 10 semillas en lo que faltaba (el
segundo origen de SDWPF y los dos de NOAA, que tiene casi el doble de ventanas) costaba unas 38 horas de GPU, unas
190 de las 280 unidades disponibles, y no dejaba margen para el resto del E3. Por eso Informer se corre con **3
semillas** (1000–1002) en esos tres orígenes; en ETTh1 y en el primer origen de SDWPF conserva las 10 que ya tenía.
Los demás modelos mantienen sus 10 semillas. En las tablas, la media y la dispersión de Informer en SDWPF y NOAA se
reportan con el número de corridas de cada origen. La cola de Colab deja Informer al final para que, si el
presupuesto se agota, los modelos que falten sean solo los suyos.

**Falta de memoria de STGCN en SDWPF (23-nov-2026).** Con 134 turbinas, STGCN se quedó sin memoria de GPU al
predecir en lotes de 4 096 ventanas. `predecir` ahora reintenta en lotes de la mitad: la predicción es la misma y
solo cambia la memoria usada.
