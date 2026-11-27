"""Reproducción del Entregable 2 con un solo comando.

    python reproducir_e2.py              análisis: regenera todas las tablas, figuras y pruebas del libro a partir de
                                         las predicciones guardadas (CPU, unos 10 minutos) y compila el libro
    python reproducir_e2.py --todo       además reentrena todo desde los datos (GPU; días de cómputo, ver el
                                         capítulo de métodos), en el orden en que se hizo
    python reproducir_e2.py --listar     solo muestra los pasos, sin correr nada

Requisitos:
* `datos/panel_multivariado_causal.csv` (instrucciones en el Data Card; el contrato de datos lo valida al cargarlo).
* Dos entornos con `requirements.txt`: uno con torch y CUDA para entrenar (variable PY_DL) y otro para el análisis
  (variable PY_ML). Si no se definen, se usa el Python que corre este archivo para todo.
* Chronos y Moirai se corren en Colab con `colab/E2_chronos.ipynb` y `colab/E2_moirai.ipynb`, porque sus bibliotecas
  piden otras versiones de torch; en el modo --todo se indica en qué momento.
* Las predicciones por ventana (`*.npz`, unos 220 MB) no se versionan en git; el modo de análisis las necesita en
  `resultados/e2/`. Se regeneran con --todo.
Cada paso se reanuda donde quedó, así que se puede cortar y volver a lanzar.
"""
import os, subprocess, sys, time

PY_DL = os.environ.get("PY_DL", sys.executable)
PY_ML = os.environ.get("PY_ML", sys.executable)
TOP = "Transformer KAN TCN STGCN GRU Informer DLinear PatchTST MLP CNN LSTM".split()

ENTRENAMIENTO = [                                     # (descripción, entorno, comando)
    ("Pruebas de fuga de información (9)", PY_ML, ["tests/test_fuga.py"]),
    ("Contrato de datos y registro de poblaciones", PY_ML, ["src/contrato_datos.py"]),
    ("Piloto de potencia y costo", PY_DL, ["src/piloto_e2.py"]),
    ("Replicación del paper guía (DLinear en ETTh1, 5 réplicas)", PY_DL, ["src/replicacion_dlinear.py"]),
    ("Líneas base en validación y prueba", PY_ML, ["src/baselines.py", "validacion"]),
    ("", PY_ML, ["src/baselines.py", "prueba"]),
    ("Ajuste previo y fase 1 de la búsqueda", PY_DL, ["src/busqueda_arq.py"]),
    ("Fase 2 de la búsqueda (10 finalistas, 5 semillas)", PY_DL,
     ["src/fase2.py", "--k", "10", "--origenes", "ultimo", "MLP", "CNN", "GRU", "LSTM", "TCN", "KAN", "Transformer"]),
    ("COLAB: representación congelada y búsqueda de Chronos y Moirai (colab/E2_chronos.ipynb, E2_moirai.ipynb)",
     None, None),
    ("Corridas finales: 4 orígenes de prueba × 10 semillas", PY_DL, ["src/finales.py", *TOP, "--semillas", "1000-1009"]),
    ("Validación de la selección con origen móvil (enmienda 8)", PY_DL,
     ["src/validacion_origenes.py", "MLP", "KAN", "CNN", "GRU", "LSTM", "TCN", "Transformer"]),
    ("Robustez e interpretabilidad de los tres mejores", PY_DL, ["src/f2a_gpu.py"]),
    ("Ablación de la TCN (enmienda 7)", PY_DL, ["src/ablacion.py"]),
    ("Reproducción independiente de la tabla principal", PY_DL, ["src/reproducir.py", "--reentrenar"]),
    ("Validación cruzada con forward chaining (sugerencia del profesor)", PY_DL, ["src/tsxv_extras.py"]),
    ("COLAB: Chronos y Moirai sin entrenar (src/preentrenados.py, enmienda 9)", None, None),
]
ANALISIS = [
    ("Sensibilidad al muestreo de 200 arquitecturas", PY_ML, ["src/sensibilidad_muestra.py"]),
    ("Análisis del espacio de arquitecturas", PY_ML, ["src/analisis_busqueda.py"]),
    ("Frente de Pareto y sesgo del ganador", PY_ML, ["src/pareto_ganador.py"]),
    ("Inferencia estadística (familias F1–F3, Friedman, MCS, TOST)", PY_ML, ["src/estadistica_e2.py"]),
    ("Techo de desempeño, potencia y componentes de varianza", PY_ML, ["src/techo_potencia.py"]),
    ("Errores y casos difíciles", PY_ML, ["src/errores_e2.py"]),
    ("Residuos, calibración, intervalos y estabilidad (ruta F2a)", PY_ML, ["src/f2a_analisis.py"]),
    ("Chronos y Moirai sin entrenar frente al benchmark", PY_ML, ["src/cero_disparos.py"]),
    ("Pruebas de las Tablas B y C que completan la ficha de ruta", PY_ML, ["src/ruta_evidencias.py"]),
    ("Costo: parámetros, FLOPs y latencia en CPU", PY_DL, ["src/costo_inferencia.py"]),
    ("Fichas de modelo (sección 11.3)", PY_ML, ["src/fichas_modelo.py"]),
    ("Tablas del libro", PY_ML, ["src/tablas_libro.py", "--escribir"]),
    ("Libro (Jupyter Book 1, formato clásico; dependencias en book/requirements.txt)",
     None, ["jupyter-book", "build", "book/"]),
]


def correr(pasos):
    for i, (desc, py, cmd) in enumerate(pasos, 1):
        if desc:
            print(f"\n[{time.strftime('%H:%M:%S')}] {i}/{len(pasos)} {desc}", flush=True)
        if cmd is None:
            if "--listar" in sys.argv:
                continue
            input("   Paso manual en Colab. Pulse Enter cuando los resultados estén en resultados/e2/ ... ")
            continue
        linea = ([py, "-W", "ignore"] if py else []) + cmd
        print("   $ " + " ".join(linea), flush=True)
        if "--listar" in sys.argv:
            continue
        r = subprocess.run(linea, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            sys.exit(f"El paso «{desc or cmd}» terminó con código {r.returncode}; se corrige y se vuelve a lanzar "
                     "(cada paso se reanuda donde quedó).")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if "--todo" in sys.argv or "--listar" in sys.argv:
        print("== Entrenamiento ==")
        correr(ENTRENAMIENTO)
    print("\n== Análisis ==")
    correr(ANALISIS)
    print("\nListo. El libro queda en book/_build/html.")
