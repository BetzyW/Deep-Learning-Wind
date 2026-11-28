# Declaración de los datos

## Panel principal: estaciones del IDEAM en el Caribe colombiano

* **Procedencia:** mediciones horarias de estaciones automáticas del IDEAM, publicadas en el Portal Nacional de Datos
  Abiertos de Colombia (datos.gov.co): velocidad del viento (`sgfv-3yp8`), dirección (`kiw7-v9ta`), temperatura
  (`sbwg-7ju4`) y catálogo de estaciones (`hp9r-jxuu`).
* **Licencia:** CC BY-SA 4.0.
* **Construcción:** el panel horario del proyecto de Machine Learning de Yuryen Rollo y Betzaida Ruiz (ETL con limpieza
  causal: cada lectura se limpia solo con información anterior). 76 estaciones de 7 departamentos, enero de 2020 a
  diciembre de 2025; 62 con ventanas válidas en el E2 y el E3.
* **Curaduría:** relleno hacia adelante de máximo 3 h; ventanas con faltantes descartadas; la estación 1501500054 se
  excluye (anemómetro bloqueado, 99.8 % de las horas en cero); la presión se descartó por faltar en el 18 % de las
  horas.
* **Partición:** origen móvil trimestral con embargo de 24 h; validación en 2024, prueba en 2025, con la huella
  SHA-256 de las ventanas de prueba fijada antes de entrenar.
* **Sesgos y limitaciones:** la red de estaciones cambia mucho entre años; el viento fuerte (≥ 6 m/s) es raro (1.9 % de
  las ventanas de prueba) y se concentra en La Guajira; más de la mitad de las estaciones tiene cambios de nivel
  locales (sensores o entorno).
* **Datos personales:** ninguno.
* **Disponibilidad:** los datos no se versionan en el repositorio; el `README.md` explica cómo obtenerlos.

## Conjuntos adicionales (solo evaluación)

| Conjunto | Fuente | Licencia | Notas |
|---|---|---|---|
| ETTh1 | Zhou et al. (2021), AAAI; repositorio `zhouhaoyi/ETDataset` | CC BY-ND 4.0 | Solo se usa para evaluar; no se redistribuyen versiones modificadas |
| SDWPF (KDD Cup 2022) | Zhou et al. (2024), *Scientific Data*, doi:10.1038/s41597-024-03427-5 | CC BY 4.0 | 134 turbinas, 245 días; agregado de 10 min a horario |
| NOAA ISD-Lite, Kansas | NOAA Integrated Surface Database, AWS Open Data | Dominio público | 10 de 34 estaciones (semilla 11), 2021–2024 |

Se eligieron el 21 de noviembre de 2026, antes de entrenar ningún modelo en ellos, con su huella SHA-256
(`experiments/e3_sha256.csv`); `src/e3_descargar.py` los baja de sus fuentes y verifica las huellas. Detalles en
`experiments/conjuntos_adicionales.md`.
