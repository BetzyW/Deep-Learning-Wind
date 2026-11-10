# Datos, predictibilidad y auditoría de fuga (Entregable 2)

## Data Card

| Campo | Contenido |
|---|---|
| Fuente | Estaciones automáticas del IDEAM, Portal de Datos Abiertos de Colombia {cite}`ideam2025` (licencia CC BY-SA 4.0) |
| Construcción | Panel horario del proyecto de Machine Learning {cite}`rollo2026ml`: series de velocidad, dirección y temperatura alineadas por estación y hora |
| Cobertura | 76 estaciones de 7 departamentos del Caribe colombiano (La Guajira 15, Magdalena 14, Cesar 12, Bolívar 11, Córdoba 11, Atlántico 8, Sucre 5); enero de 2020 a diciembre de 2025; 1 308 726 filas estación-hora |
| Variables de entrada | Velocidad (m/s), dirección como seno y coseno, temperatura (°C); ventana de 168 h |
| Objetivo | Velocidad media horaria 24 h después del instante de emisión |
| Limpieza | Relleno hacia adelante de máximo 3 h (solo mira el pasado); una ventana con faltantes no se usa; una estación excluida por datos anómalos (`configs/e2.json`) |
| Partición | Origen móvil trimestral: validación en 2024, prueba en 2025 (tabla siguiente) |
| Huella de la prueba | SHA-256 de las 215 286 ventanas de 2025, fijada en el preregistro antes de entrenar |
| Sesgos conocidos | La red cambia mucho entre años (el primer semestre de 2024 tiene 10–16 estaciones y el último trimestre de 2025, 56); las estaciones nuevas tienen poca historia |
| Usos no previstos | No sustituye un pronóstico oficial; no se validó para eventos extremos (huracanes) |

## Partición

| Etapa | Origen | Ventanas evaluadas | Estaciones | RMSE de la persistencia |
|---|---|---|---|---|
| Validación | 2024-01-01 | 8 459 | 10 | 0.903 |
| Validación | 2024-04-01 | 9 485 | 16 | 0.978 |
| Validación | 2024-07-01 | 22 696 | 16 | 1.085 |
| Validación | 2024-10-01 | 37 727 | 43 | 0.886 |
| Prueba | 2025 (4 orígenes) | 215 286 | 40–56 | (se calcula solo en las corridas finales) |

Como los trimestres de validación son muy desiguales, el error de validación se resume juntando las ventanas de los
cuatro pliegues y no promediando los cuatro RMSE por igual.

## Predictibilidad intrínseca

Antes de entrenar se midió qué tan predecible es cada estación con dos medidas sin modelo (62 estaciones con datos
antes de 2024):

| Medida | Qué mide | Resultado |
|---|---|---|
| Entropía de permutación, m = 5 {cite}`bandt2002` | Qué tan desordenados son los patrones de 5 horas seguidas (1 = ruido) | Media 0.926 (0.864–0.975) |
| Ídem con retardo de 24 h | Orden de un día al siguiente a la misma hora | 0.995: casi aleatorio |
| Predictibilidad espectral Ω {cite}`goerg2013` | Concentración de la potencia en pocas frecuencias (0 = ruido) | Media 0.315 (0.084–0.530) |

Los controles se comportan como se espera: el ruido blanco da entropía 0.999 y Ω = 0.002, y una senoide diaria con
ruido da 0.926 y 0.660. **Ω anticipa la dificultad de cada estación**: su correlación de Spearman con el error relativo
de la persistencia es ρ = −0.80 (p < 0.0001); las estaciones con un ciclo diario marcado son las fáciles. Esta medida
sirve para explicar por qué el error varía tanto entre estaciones y para estimar el techo de desempeño.

## Auditoría de fuga de información

| Riesgo | Control | Verificación |
|---|---|---|
| Objetivos de evaluación usados para entrenar | Embargo de 24 h entre el último objetivo de entrenamiento y el primer objetivo evaluado | `test_embargo_entre_entrenamiento_y_evaluacion` |
| Pliegues que se solapan | Cada pliegue evalúa solo su trimestre | `test_evaluacion_dentro_de_su_periodo`, `test_pliegues_de_evaluacion_disjuntos` |
| Imputación que mira el futuro | Relleno solo hacia adelante | `test_relleno_causal` |
| Escalado con datos futuros | Media y desviación de las entradas del entrenamiento de cada origen | `test_escalador_usa_solo_entradas_de_entrenamiento`, `test_escalador_y_rezagos_no_miran_el_futuro` |
| Ventanas mal construidas | Entrada completa y objetivo observado | `test_ventanas_validas`, `test_objetivos_coinciden_con_los_datos_originales`, `test_panel_real` |
| Selección con la prueba | Arquitectura e hiperparámetros elegidos solo con validación; la configuración se guarda antes de abrir la prueba | `src/finales.py` (archivo `{modelo}_configuracion.json` con fecha) |
| Descomposición de la señal (EMD, VMD) | No se usa ninguna (el 41 % de los estudios revisados sí la usa, con riesgo de fuga) | Diseño |

Las 9 pruebas pasan (`python tests/test_fuga.py`).

**Declaración.** El 9 de noviembre de 2026 se hizo una prueba de humo del script de corridas finales en una copia
temporal: DLinear con una configuración arbitraria (no la elegida), una semilla, sobre los cuatro orígenes de prueba.
Sus resultados se borraron y no se usaron para ninguna decisión, porque todas las configuraciones se fijan solo con
validación. Se declara por transparencia; las pruebas de humo posteriores usan orígenes de validación.
