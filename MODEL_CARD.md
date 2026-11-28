# Ficha del modelo: PADRE-TCN (diseño V6)

Formato de Mitchell et al. (2019), *Model cards for model reporting*.

## Detalles del modelo

* **Autora:** Betzaida Ruiz (Deep Learning, Universidad del Norte), noviembre de 2026.
* **Tipo:** red neuronal de regresión para series de tiempo. Codificador TCN causal (3 bloques, anchos 64, 64 y 16)
  más una regresión por diferencias contra K = 6 anclas diarias, con diferencia antisimétrica por construcción y pesos
  aprendidos por similitud. Salidas: pronóstico a 24 h y dispersión entre anclas.
* **Fuente del mecanismo:** transferido y adaptado de PADRE (Tynes et al., 2021), TNNR (Wetzel et al., 2022) y el
  ensamble de análogos (Delle Monache et al., 2013); ver `experiments/procedencia.md`.
* **Parámetros:** 100 663. **Costo:** 33.0 MFLOPs por ventana; 0.98 ms por ventana en CPU (lote 1).
* **Entrenamiento:** Adam (lr = 8.5e-4, wd = 1e-5, lote 1 024), error cuadrático, máximo 50 épocas, parada temprana
  con paciencia 5. Hiperparámetros elegidos entre las mismas 20 configuraciones que los benchmarks.
* **Artefacto:** `artifacts/padre_v6.onnx` con `metadata.json`. **Código:** `src/original_model.py`. **Licencia:** MIT.

## Uso previsto

* **Previsto:** pronóstico de la velocidad media horaria del viento 24 h adelante en estaciones meteorológicas
  terrestres, con un intervalo de predicción al 90 %, a partir de las últimas 168 h de velocidad, dirección y
  temperatura de la propia estación.
* **Usuarios:** investigación y apoyo a la planeación (operación de parques eólicos, puertos).
* **Fuera de alcance:** no sustituye un pronóstico oficial; no se validó para eventos extremos (huracanes, frentes
  fríos fuertes); no se recomienda para potencia de turbinas (rindió mal en SDWPF).

## Factores

El error depende mucho de la estación (su ciclo diario y su nivel de viento), de la época del año y del régimen de
viento. El desempeño empeora con viento fuerte (≥ 6 m/s), que es raro en los datos.

## Métricas

RMSE (primaria, por estación), MAE, sesgo con viento fuerte, *interval score* de Winkler al 90 %, cobertura y ancho
del intervalo. Margen de relevancia práctica: δ = 0.02 m/s.

## Datos de evaluación y de entrenamiento

Panel horario del IDEAM, 62 estaciones del Caribe colombiano (2020–2025). Entrenamiento con todo lo anterior a cada
origen; prueba en los 4 trimestres de 2025 (215 286 ventanas), reentrenando en cada uno; 10 semillas. Evaluación
adicional en ETTh1, SDWPF y NOAA Kansas. Ver `DATA_STATEMENT.md`.

## Análisis cuantitativo (prueba de 2025)

| Métrica | PADRE-TCN | Mejor benchmark |
|---|---|---|
| RMSE (m/s) | 0.7239 ± 0.0010 | 0.7189 (Transformer) |
| Winkler al 90 % (m/s) | **2.991** | 3.153 (Transformer) |
| Cobertura del intervalo al 90 % | 0.898 | 0.899 |
| Sesgo con viento ≥ 6 m/s (m/s) | −1.21 | −1.21 (Transformer) |

No supera a la familia de 12 rivales en RMSE (diferencia con el Transformer: 0.005 m/s, menor que δ), es equivalente a
la TCN (TOST) y tiene los mejores intervalos de la familia en los 12 contrastes. Detalle en los capítulos 7 y 8 del
libro.

## Consideraciones éticas

Los datos son mediciones públicas de estaciones, sin información personal. El riesgo principal es de uso: un
intervalo bien calibrado en promedio puede dar una falsa sensación de seguridad en eventos extremos, donde la
cobertura baja a 0.74.

## Advertencias y recomendaciones

* Subestima los vientos fuertes, como todos los modelos entrenados con error cuadrático.
* El pronóstico no puede salir del rango de los pronósticos de sus anclas (Proposición 2 del libro).
* Es más sensible que los benchmarks al ruido fuerte en la entrada.
* Escalar la entrada con la media y la desviación de `metadata.json`; usar una ventana completa de 168 h.
