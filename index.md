# Pronóstico de viento a 24 h con Deep Learning

**Curso:** Deep Learning · **Profesores:** Lihki Rubio y Keila Alba · **Autora:** Betzaida Ruiz · Universidad del Norte

¿Qué arquitectura base de Deep Learning pronostica mejor la velocidad del viento con 24 horas de antelación en las
estaciones del Caribe colombiano, y cuánto error queda por reducir frente al pronóstico ingenuo?

Este primer entregable define formalmente el problema, analiza los datos, revisa la literatura con el protocolo
PRISMA, implementa tres modelos base (MLP, LSTM y CNN 1D) y los compara entre sí y con la persistencia mediante
pruebas estadísticas.

## Declaraciones a priori

Se fijan antes de ver cualquier resultado de prueba y no se cambian después.

| Declaración | Valor | Justificación |
|---|---|---|
| Métrica primaria | RMSE (m/s) | Coherente con la pérdida cuadrática de entrenamiento |
| Métricas secundarias | MAE, $R^2$ y MASE respecto a la persistencia a 24 h | El MASE mide la mejora sobre el pronóstico ingenuo |
| Margen de relevancia | $\delta = 0.02$ m/s | ≈ 3 % del RMSE de referencia del proyecto de Machine Learning (0.72 m/s) {cite}`rollo2026ml` |
| Unidad de independencia | La estación | Las horas de una misma estación están autocorrelacionadas |
| Partición | Entrenamiento 2020–2023, validación 2024, prueba 2025 | Cronológica, con embargo de $L + 24$ h; prueba congelada |
| Nivel de significancia | $\alpha = 0.05$, con corrección de Holm | Varias comparaciones entre modelos |

## Uso de inteligencia artificial

Este trabajo se desarrolló con apoyo de Claude (Anthropic) para la programación, la búsqueda bibliográfica y la
redacción, según lo permitido en el curso. Las decisiones, la revisión de los resultados y la interpretación son
responsabilidad de la autora.

## Reproducir

Las instrucciones para obtener el panel de datos y ejecutar cada capítulo están en el `README.md` del repositorio.
