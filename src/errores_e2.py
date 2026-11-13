"""Taxonomía de errores y errores compartidos del E2 (sección 12.3 del enunciado).

Categorías mutuamente excluyentes, definidas antes de etiquetar y aplicadas con reglas fijas (en este orden de
prioridad) a cada ventana con error grande; v_τ es la velocidad al emitir el pronóstico e y la observada 24 h después:
  1. viento_fuerte_subestimado: y ≥ 6 m/s y error < 0.
  2. calma_sobreestimada: y < 1 m/s y error > 0.
  3. rampa_ascendente: y − v_τ ≥ 2 m/s.
  4. rampa_descendente: v_τ − y ≥ 2 m/s.
  5. giro_de_direccion: la dirección cambia más de 90° entre τ y τ + 24 (con velocidades ≥ 1 m/s en ambos).
  6. otro.
"Error grande": |error| en el 10 % superior del modelo. Muestra estratificada por origen (trimestre): 40 ventanas por
trimestre y modelo (160 por modelo principal, semilla 0). Se reporta la frecuencia de cada categoría con IC 95 % de
Wilson y, como referencia, la frecuencia de la misma categoría en todas las ventanas de prueba (para ver qué
situaciones están sobrerrepresentadas entre los errores grandes).
Errores compartidos: entre los modelos principales (TCN, Transformer, KAN), Ridge y la persistencia, índice de Jaccard
de sus conjuntos de errores grandes y proporción de errores grandes de los tres modelos principales a la vez.
Salida: resultados/e2/errores/
Uso: python src/errores_e2.py      (ml_env)
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import cargar  # noqa: E402

PRINCIPALES = ["TCN", "Transformer", "KAN"]
REFERENCIAS = ["Ridge", "Persistencia"]
SALIDA = "resultados/e2/errores"
ORDEN = ["viento_fuerte_subestimado", "calma_sobreestimada", "rampa_ascendente", "rampa_descendente",
         "giro_de_direccion", "otro"]


def categorizar(d):
    c = np.full(len(d), "otro", dtype=object)
    reglas = [
        ("giro_de_direccion", (d.giro > 90) & (d.v_tau >= 1) & (d.y >= 1)),
        ("rampa_descendente", d.v_tau - d.y >= 2),
        ("rampa_ascendente", d.y - d.v_tau >= 2),
        ("calma_sobreestimada", (d.y < 1) & (d.e > 0)),
        ("viento_fuerte_subestimado", (d.y >= 6) & (d.e < 0)),
    ]
    for nombre, m in reglas:                    # de menor a mayor prioridad: la última regla que se cumple gana
        c[m.to_numpy()] = nombre
    return c


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


if __name__ == "__main__":
    from origen_movil import leer_config, preparar_panel
    p = preparar_panel(leer_config())
    os.makedirs(SALIDA, exist_ok=True)
    h = p.horizontes[0]
    datos, grandes = {}, {}
    for m in PRINCIPALES + REFERENCIAS:
        d = cargar(p, m).dropna(subset=["y"]).copy()
        d["e"] = d.pred - d.y
        d["v_tau"] = p.X[d.tau, 0]
        a0 = np.degrees(np.arctan2(p.X[d.tau, 1], p.X[d.tau, 2]))
        a1 = np.degrees(np.arctan2(p.X[d.tau + h, 1], p.X[d.tau + h, 2]))
        d["giro"] = np.abs((a1 - a0 + 180) % 360 - 180)
        d["origen"] = pd.cut(pd.to_datetime(d.tiempo), pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01",
                             "2025-10-01", "2026-01-02"]), right=False, labels=["T1", "T2", "T3", "T4"]).astype(str)
        d["categoria"] = categorizar(d)
        d["grande"] = d.e.abs() >= d.e.abs().quantile(0.9)
        datos[m] = d
        grandes[m] = set(d.tau[d.grande])
    base = datos[PRINCIPALES[0]].categoria.value_counts(normalize=True)
    filas, muestras = [], []
    rng = np.random.default_rng(0)
    for m in PRINCIPALES:
        g = datos[m][datos[m].grande]
        mu = pd.concat([x.iloc[rng.choice(len(x), min(40, len(x)), replace=False)] for _, x in g.groupby("origen")])
        muestras.append(mu.assign(modelo=m)[["modelo", "tau", "estacion", "tiempo", "origen", "y", "pred", "e",
                                             "v_tau", "giro", "categoria"]])
        n = len(mu)
        for cat in ORDEN:
            k = int((mu.categoria == cat).sum())
            lo, hi = wilson(k, n)
            filas.append(dict(modelo=m, categoria=cat, n=n, casos=k, frecuencia=k / n, ic_inf=lo, ic_sup=hi,
                              frecuencia_en_todas=float(base.get(cat, 0)),
                              frecuencia_en_todos_los_grandes=float((g.categoria == cat).mean())))
    t = pd.DataFrame(filas)
    t.to_csv(f"{SALIDA}/taxonomia.csv", index=False)
    pd.concat(muestras).to_csv(f"{SALIDA}/muestra_etiquetada.csv", index=False)
    jac = {f"{a}|{b}": len(grandes[a] & grandes[b]) / len(grandes[a] | grandes[b])
           for a, b in itertools.combinations(PRINCIPALES + REFERENCIAS, 2)}
    tres = set.intersection(*[grandes[m] for m in PRINCIPALES])
    res = dict(jaccard=jac, proporcion_grandes_compartidos_por_los_tres=len(tres) / len(grandes[PRINCIPALES[0]]),
               umbral_error_grande={m: float(datos[m].e.abs().quantile(0.9)) for m in PRINCIPALES + REFERENCIAS})
    json.dump(res, open(f"{SALIDA}/compartidos.json", "w"), indent=1, ensure_ascii=False)
    print(t.pivot_table(index="categoria", columns="modelo", values="frecuencia").round(3).to_string())
    print(t[t.modelo == "TCN"][["categoria", "frecuencia_en_todas", "frecuencia_en_todos_los_grandes"]].round(3)
          .to_string(index=False))
    print(json.dumps(res, indent=1))
