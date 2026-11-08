"""Pruebas de fuga de la infraestructura del Entregable 2 (origen móvil, relleno, escalado y rezagos).

Se corren con `python tests/test_fuga.py` (no hace falta pytest; si está instalado, `pytest tests` también sirve).
Usan un panel sintético pequeño con huecos; la última prueba repite las comprobaciones con el panel real si está
en `datos/`.
"""
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from origen_movil import (HORA, Escalador, filas_de_entrada, leer_config, objetivos,  # noqa: E402
                          pliegues, preparar_panel, rezagos)

CFG = dict(columnas=["vel", "dir_sin", "dir_cos", "temp"], excluir_estaciones=["X"], ffill_max_horas=3,
           L=48, horizontes=[24], inicio="2023-06-01", ventana_entrenamiento="expansiva", embargo_horas=24,
           parada_temprana_horas=720, origenes_validacion=["2024-07-01", "2024-10-01"],
           origenes_prueba=["2025-01-01", "2025-02-01"], fin="2025-03-01")
LAGS = [0, 1, 23, 24, 47]


def sintetico(semilla=0, alterar_desde=None):
    """Tres estaciones (una excluida) con huecos aleatorios; con `alterar_desde` cambia todo lo posterior."""
    rng = np.random.default_rng(semilla)
    dfs = []
    for cod in ["A", "B", "X"]:
        hora = pd.date_range("2023-06-01", "2025-03-01", freq="h", inclusive="left")
        n = len(hora)
        t = np.arange(n)
        vel = 2 + np.sin(2 * np.pi * t / 24) + rng.normal(0, .3, n)
        d = pd.DataFrame(dict(CodigoEstacion=cod, hora=hora, vel=vel, dir_sin=rng.uniform(-1, 1, n),
                              dir_cos=rng.uniform(-1, 1, n), temp=27 + rng.normal(0, 1, n)))
        hueco = rng.random(n) < 0.03
        hueco[2000:2400] = True                                    # un hueco largo
        d.loc[hueco, ["vel", "temp"]] = np.nan
        d = d[rng.random(n) > 0.01]                                # filas que ni siquiera existen en el CSV
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    if alterar_desde is not None:
        m = df.hora >= alterar_desde
        df.loc[m, ["vel", "temp"]] = rng.normal(50, 10, (m.sum(), 2))
    return df


P = preparar_panel(CFG, sintetico())


def todos_los_pliegues(p=P, cfg=CFG):
    return pliegues(p, cfg, "validacion") + pliegues(p, cfg, "prueba")


def test_ventanas_validas():
    p = P
    assert "X" not in set(p.estacion)
    e = p.X[filas_de_entrada(p, p.tau)]
    assert not np.isnan(e).any(), "una ventana válida tiene entradas faltantes"
    assert not np.isnan(objetivos(p, p.tau)).any(), "una ventana válida tiene el objetivo faltante"
    for h in p.horizontes:
        assert (p.estacion[p.tau - p.L + 1] == p.estacion[p.tau + h]).all(), "ventana que cruza dos estaciones"
        assert (p.tiempo[p.tau + h] - p.tiempo[p.tau] == h * HORA).all(), "la rejilla horaria no es regular"


def test_embargo_entre_entrenamiento_y_evaluacion():
    p = P
    for f in todos_los_pliegues():
        ult_tr = p.t(f["entrenamiento"]).max() + p.hmax * HORA       # último objetivo de entrenamiento
        pri_ev = p.t(f["evaluacion"]).min() + p.hmin * HORA          # primer objetivo evaluado
        assert pri_ev - ult_tr >= CFG["embargo_horas"] * HORA, f["origen"]
        ult_aj = p.t(f["ajuste"]).max() + p.hmax * HORA
        pri_pa = p.t(f["parada"]).min() + p.hmin * HORA
        assert pri_pa - ult_aj >= CFG["embargo_horas"] * HORA, "sin embargo entre ajuste y parada temprana"
        assert np.isin(f["ajuste"], f["entrenamiento"]).all() and np.isin(f["parada"], f["entrenamiento"]).all()
        assert not np.isin(f["ajuste"], f["parada"]).any()


def test_evaluacion_dentro_de_su_periodo():
    p = P
    primera_prueba = np.datetime64(CFG["origenes_prueba"][0], "h")
    for f in todos_los_pliegues():
        t = p.t(f["evaluacion"])
        T, Tn = np.datetime64(f["origen"], "h"), np.datetime64(f["fin"], "h")
        assert (t >= T).all() and (t + p.hmax * HORA < Tn).all()
    for f in pliegues(p, CFG, "validacion"):
        objetivo = p.t(np.concatenate([f["entrenamiento"], f["evaluacion"]])) + p.hmax * HORA
        assert (objetivo < primera_prueba).all(), "un objetivo del periodo de prueba se usa en validación"


def test_pliegues_de_evaluacion_disjuntos():
    ev = np.concatenate([f["evaluacion"] for f in todos_los_pliegues()])
    assert len(ev) == len(np.unique(ev))


def test_relleno_causal():
    """Alterar los datos desde una fecha no cambia ningún valor del panel anterior a esa fecha."""
    corte = "2024-09-15"
    q = preparar_panel(CFG, sintetico(alterar_desde=corte))
    antes = P.tiempo < np.datetime64(corte, "h")
    assert (P.tiempo == q.tiempo).all()
    np.testing.assert_array_equal(P.X[antes], q.X[antes])


def test_escalador_y_rezagos_no_miran_el_futuro():
    """El escalador y los rezagos de cada origen no cambian si se altera todo lo posterior al origen."""
    for f in todos_los_pliegues():
        q = preparar_panel(CFG, sintetico(alterar_desde=f["origen"]))
        tr = f["entrenamiento"]
        assert np.isin(tr, q.tau).all()
        a, b = Escalador().ajustar(P, tr), Escalador().ajustar(q, tr)
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.sd, b.sd)
        pd.testing.assert_frame_equal(rezagos(P, tr, LAGS), rezagos(q, tr, LAGS))
        np.testing.assert_array_equal(objetivos(P, tr), objetivos(q, tr))
        # las entradas de evaluación sí pueden ver horas previas al origen, pero ninguna posterior a τ
        ev = f["evaluacion"]
        q2 = preparar_panel(CFG, sintetico(alterar_desde=str(P.t(ev).max() + HORA)))
        pd.testing.assert_frame_equal(rezagos(P, ev, LAGS), rezagos(q2, ev, LAGS))


def test_escalador_usa_solo_entradas_de_entrenamiento():
    f = pliegues(P, CFG, "validacion")[0]
    filas = filas_de_entrada(P, f["entrenamiento"])
    assert (P.tiempo[filas] < np.datetime64(f["origen"], "h")).all()
    e = Escalador().ajustar(P, f["entrenamiento"])
    np.testing.assert_allclose(e.mu, P.X[filas].mean(0), rtol=1e-5)


def test_objetivos_coinciden_con_los_datos_originales():
    df = sintetico().set_index(["CodigoEstacion", "hora"])["vel"]
    idx = P.tau[::997]
    esperado = [df.get((e, pd.Timestamp(t) + pd.Timedelta(hours=24)), np.nan)
                for e, t in zip(P.estacion[idx], P.tiempo[idx])]
    np.testing.assert_allclose(objetivos(P, idx)[:, 0], np.array(esperado, np.float32), rtol=1e-6)


def test_panel_real():
    cfg = leer_config(os.path.join(RAIZ, "configs", "e2.json"))
    cfg["panel"] = os.path.join(RAIZ, cfg["panel"])
    if not os.path.exists(cfg["panel"]):
        print("   (panel real ausente: se omite)")
        return
    p = preparar_panel(cfg)
    fs = pliegues(p, cfg, "validacion") + pliegues(p, cfg, "prueba")
    for f in fs:
        ult_tr = p.t(f["entrenamiento"]).max() + p.hmax * HORA
        pri_ev = p.t(f["evaluacion"]).min() + p.hmin * HORA
        assert pri_ev - ult_tr >= cfg["embargo_horas"] * HORA
    assert not np.isnan(p.X[filas_de_entrada(p, p.tau)]).any()
    ev = np.concatenate([f["evaluacion"] for f in fs])
    assert len(ev) == len(np.unique(ev))


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    fallas = 0
    for n, f in pruebas:
        try:
            f()
            print(f"OK    {n}")
        except AssertionError as e:
            fallas += 1
            print(f"FALLA {n}: {e}")
    print(f"\n{len(pruebas) - fallas} de {len(pruebas)} pruebas superadas")
    sys.exit(1 if fallas else 0)
