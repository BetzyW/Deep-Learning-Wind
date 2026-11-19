# Introducción y preguntas (Entregable 2)

En el Entregable 1 tres redes base (MLP, LSTM y CNN 1D) redujeron en ≈ 17 % el error de la persistencia a 24 h en las
estaciones del Caribe colombiano, pero una regresión lineal las igualó: el margen de mejora no estaba en cambiar de
arquitectura temporal, sino en los vientos fuertes y en la información espacial que no se usaba. Este entregable
escala la comparación a los **10 modelos más relevantes de la literatura reciente**, elegidos con una revisión
sistemática, y los somete al mismo protocolo, con el mismo presupuesto de ajuste y con inferencia estadística formal.

## Qué cambia respecto al Entregable 1

| Tema | Entregable 1 | Entregable 2 |
|---|---|---|
| Partición | Fija: 2020–2023 / 2024 / 2025 | Origen móvil trimestral con embargo de 24 h, reentrenando en cada origen |
| Modelos | MLP, LSTM, CNN 1D | Top 10 de la revisión sistemática + los 3 del E1 + baselines clásicos y triviales |
| Arquitectura | Elegida con un ajuste corto | Búsqueda en dos fases sobre 1 000 arquitecturas (sección 14 del enunciado) |
| Inferencia | Friedman, Wilcoxon, Diebold–Mariano | Familias con Holm, DM-HLN, Giacomini–White, Clark–West, MCS, TOST |
| Paper guía | — | DLinear, replicado con sus condiciones originales |

## Preguntas e hipótesis

Fijadas en el preregistro antes de entrenar:

| Pregunta | Hipótesis |
|---|---|
| **RQ1.** ¿Algún modelo del Top 10 supera al mejor baseline clásico? | H1: al menos uno baja el RMSE en más de δ = 0.02 m/s frente al mejor clásico elegido en validación (ridge) |
| **RQ2.** ¿Los modelos preentrenados (Chronos, Moirai) aportan frente a los entrenados desde cero? | H2: el mejor preentrenado no difiere en más de δ del mejor entrenado desde cero (equivalencia, TOST) |
| **RQ3.** ¿La limitación del paper guía es propia de DLinear o de toda la familia? | H3: la subestimación en vientos fuertes (≥ 6 m/s) aparece en DLinear y en al menos dos de los otros modelos del Top 10 |

La tarea es un pronóstico directo de la velocidad media horaria a 24 h, a partir de las 168 h anteriores de velocidad,
dirección y temperatura de cada estación; la estación es la unidad de independencia. La métrica primaria es el RMSE
(heredada del E1) y el margen de relevancia δ = 0.02 m/s.
