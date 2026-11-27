"""Verificación numérica de las dos proposiciones del capítulo de diseño sobre el modelo V6 **entrenado** (el .pt de la
semilla 1000 en el origen 2025-10-01) y con ventanas **reales** de prueba, además de las pruebas unitarias con pesos
aleatorios (tests/test_original_model.py).

  Proposición 1 (antisimetría): D(a, b) = −D(b, a) y D(a, a) = 0 para cualquier valor de los parámetros. Se evalúa
    con las representaciones reales de la consulta q y de las K anclas a_j de cada ventana, en float32 (como corre el
    modelo) y en float64: máx |D(q, a_j) + D(a_j, q)| y máx |D(a_j, a_j)|.
  Proposición 2 (combinación convexa): min_j ŷ_j ≤ ŷ ≤ max_j ŷ_j, pesos w_j > 0 que suman 1, y dispersión
    √Σ_j w_j (ŷ_j − ŷ)² = 0 si y solo si todas las ŷ_j coinciden. Se mide la peor violación de la cota, el mínimo de
    los pesos y |Σ_j w_j − 1|; y, para el «si y solo si», se fuerza ŷ_j iguales (todas las anclas iguales a la
    consulta) y se comprueba que la dispersión da 0.
Salida: resultados/e3/verificacion_propiedades.json
Uso: python src/verificar_propiedades.py      (dl_env; CPU)
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import Escalador, entradas, leer_config, pliegues, preparar_panel  # noqa: E402
from original_model import PadreTCN  # noqa: E402

PT = "resultados/e3/confirmatorias/V6/2025-10-01_1000.pt"
N_VENTANAS = 20_000


@torch.no_grad()
def revisar(m, X):
    h = m.codificador(X.transpose(1, 2)).transpose(1, 2)
    q, a = h[:, -1], h[:, m.idx_ancla]
    qk = q.unsqueeze(1).expand_as(a)
    y, y_j, w = m.componentes(X)
    disp = (w * (y_j - y.unsqueeze(-1)) ** 2).sum(-1).sqrt()
    return dict(
        max_abs_D_qa_mas_D_aq=float((m.D(qk, a) + m.D(a, qk)).abs().max()),
        max_abs_D_aa=float(m.D(a, a).abs().max()),
        max_abs_D_qa=float(m.D(qk, a).abs().max()),           # escala de referencia de D (m/s)
        peor_violacion_cota_convexa=float(torch.maximum(y_j.min(-1).values - y, y - y_j.max(-1).values).max()),
        peso_minimo=float(w.min()),
        max_abs_suma_pesos_menos_1=float((w.sum(-1) - 1).abs().max()),
        dispersion_minima=float(disp.min()),
    )


@torch.no_grad()
def dispersion_con_anclas_iguales(m, X):
    """Si todas las anclas tienen la misma representación que la consulta, D = 0 y cada ŷ_j = y_j (objetivo del
    ancla); con objetivos también iguales, todas las ŷ_j coinciden y la dispersión debe ser 0."""
    h = m.codificador(X.transpose(1, 2)).transpose(1, 2)
    q = h[:, -1]
    a = q.unsqueeze(1).expand(-1, m.K, -1)
    y_j = torch.full((len(X), m.K), 5.0, dtype=X.dtype) + m.D(a, a)      # mismo objetivo en todas las anclas
    w = torch.softmax(-((m.P(a) - m.P(a)) ** 2).sum(-1) / m.log_T.exp() + m.beta, -1)
    y = (w * y_j).sum(-1)
    return float((w * (y_j - y.unsqueeze(-1)) ** 2).sum(-1).sqrt().max())


if __name__ == "__main__":
    ck = torch.load(PT, map_location="cpu", weights_only=False)
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = next(f for f in pliegues(p, cfg, "prueba") if f["origen"] == ck["origen"])
    esc = Escalador().ajustar(p, f["entrenamiento"])
    ev = np.random.default_rng(0).choice(np.asarray(f["evaluacion"]), N_VENTANAS, replace=False)
    X32 = torch.tensor(entradas(esc.transformar(p.X), np.sort(ev), p.L))
    salida = dict(modelo=PT, origen=ck["origen"], ventanas=N_VENTANAS)
    for nombre, tipo in [("float32", torch.float32), ("float64", torch.float64)]:
        m = PadreTCN(p.L, p.X.shape[1], **ck["arquitectura"], mu_y=float(esc.mu[0]), sd_y=float(esc.sd[0]))
        m.load_state_dict(ck["estado"])
        m = m.to(tipo).eval()
        X = X32.to(tipo)
        salida[nombre] = revisar(m, X)
        salida[nombre]["dispersion_con_anclas_iguales"] = dispersion_con_anclas_iguales(m, X[:2000])
    json.dump(salida, open("resultados/e3/verificacion_propiedades.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(salida, indent=1))
