"""Tablas en Markdown para los capítulos de resultados del E2, armadas directamente desde los archivos de resultados
(así los números del libro no se copian a mano). Imprime cada tabla; `--escribir` las guarda en
resultados/e2/tablas_libro.md.
Uso: python src/tablas_libro.py [--escribir]      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
E = "resultados/e2"
f = lambda x, n=3: f"`{x:.{n}f}`"
fp = lambda p: "`< 0.001`" if p < 0.001 else f"`{p:.3f}`"


def principal():
    from origen_movil import leer_config, preparar_panel
    from estadistica_e2 import cargar
    p = preparar_panel(leer_config())
    ic = pd.read_csv(f"{E}/f2a/rmse_ic_bootstrap.csv").set_index("modelo")
    r = json.load(open(f"{E}/estadistica/resumen.json"))
    rang = r["friedman"]["rangos_medios"]
    f1 = pd.read_csv(f"{E}/estadistica/familia_F1.csv").set_index("modelo")
    f2 = pd.read_csv(f"{E}/estadistica/familia_F2.csv").set_index("modelo")
    pers = cargar(p, "Persistencia").dropna(subset=["y"])
    mae_p = float((pers.pred - pers.y).abs().mean())
    filas = []
    for m in ic.index:
        d = cargar(p, m).dropna(subset=["y"])
        mae = float((d.pred - d.y).abs().mean())
        filas.append((ic.loc[m, "rmse"], f"| {m.replace('_', ' ')} | {f(ic.loc[m, 'rmse'])} | "
                      f"[{ic.loc[m, 'ic_inf']:.3f}, {ic.loc[m, 'ic_sup']:.3f}] | {f(mae)} | {f(mae / mae_p)} | "
                      + (f"{rang[m]:.2f}" if m in rang else "—") + " | "
                      + ("sí" if m in r["mcs"]["incluidos"] else ("no" if m in r["mcs"]["excluidos"] else "—"))
                      + " |"))
    t = ["| Modelo | RMSE (m/s) | IC 95 % (bootstrap de estaciones) | MAE (m/s) | MASE | Rango medio (Friedman) | En el MCS |",
         "|---|---|---|---|---|---|---|"] + [x for _, x in sorted(filas)]
    fam = ["| Modelo | Δ RMSE medio vs Ridge | p Wilcoxon (Holm) | DM (p) | p CW | Δ RMSE medio vs persistencia "
           "| p Wilcoxon (Holm) | DM (p) |", "|---|---|---|---|---|---|---|---|"]
    for m in f1.sort_values("wilcoxon_diferencia_media").index:
        a, b = f1.loc[m], f2.loc[m]
        fam.append(f"| {m} | {f(a.wilcoxon_diferencia_media, 4)} | {fp(a.wilcoxon_p_holm)} | "
                   f"`{a.dm_estadistico:.2f}` ({fp(a.dm_p)}) | {fp(a.cw_p)} | {f(b.wilcoxon_diferencia_media, 3)} | "
                   f"{fp(b.wilcoxon_p_holm)} | `{b.dm_estadistico:.2f}` ({fp(b.dm_p)}) |")
    return "\n".join(t), "\n".join(fam)


def f3():
    t = pd.read_csv(f"{E}/estadistica/familia_F3.csv")
    s = ["| Modelo | Sesgo agregado (m/s) | IC 95 % | Estaciones que subestiman | p (Holm) |", "|---|---|---|---|---|"]
    for r in t.itertuples():
        s.append(f"| {r.modelo} | {f(r.sesgo_agregado, 2)} | [{r.ic95_inf:.2f}, {r.ic95_sup:.2f}] | "
                 f"`{r.estaciones_subestiman:.0%}` | " + ("—" if pd.isna(r.wilcoxon_p_holm) else fp(r.wilcoxon_p_holm))
                 + " |")
    return "\n".join(s)


def busqueda():
    r = json.load(open(f"{E}/busqueda_analisis/resumen.json"))
    c = pd.read_csv(f"{E}/busqueda_analisis/costo.csv").set_index("modelo")
    s = ["| Modelo | Arquitecturas | R² efectos | Factor principal (prop. SC) | DE entre arquitecturas / entre semillas | "
         "ρ fase 1–fase 2 | Ganador | Equivalentes (TOST) | Horas GPU |", "|---|---|---|---|---|---|---|---|---|"]
    for m, v in r.items():
        if m == "costo_total_horas":
            continue
        fac = max(v["anova"].items(), key=lambda x: x[1]["prop_ss"])
        rho = v["fase2"]["spearman_fase1_fase2"]
        s.append(f"| {m} | `{v['n_arquitecturas']}` | {f(v['r2'], 2)} | {fac[0].replace('C(', '').replace(')', '')} "
                 f"(`{fac[1]['prop_ss']:.2f}`) | {f(v['razon_arquitecturas_semillas'], 1)} | "
                 + (f"{f(rho['rho'], 2)}" if rho else "—") + f" | {v['fase2']['ganador']} | "
                 f"`{len(v['fase2']['equivalentes_tost'])}` de `{v['fase2']['n_finalistas']}` | "
                 f"{f(c.loc[m, 'horas_total'], 1)} |")
    return "\n".join(s)


def errores():
    t = pd.read_csv(f"{E}/errores/taxonomia.csv")
    piv = t.pivot_table(index="categoria", columns="modelo", values="frecuencia")
    base = t.groupby("categoria").frecuencia_en_todas.first()
    s = ["| Categoría | TCN | Transformer | KAN | En todas las ventanas |", "|---|---|---|---|---|"]
    for c in piv.sort_values("TCN", ascending=False).index:
        s.append(f"| {c.replace('_', ' ')} | " + " | ".join(f"`{piv.loc[c, m]:.0%}`" for m in ["TCN", "Transformer", "KAN"])
                 + f" | `{base[c]:.1%}` |")
    return "\n".join(s)


if __name__ == "__main__":
    t1, t2 = principal()
    tablas = dict(principal=t1, familias=t2, f3=f3(), busqueda=busqueda(), errores=errores())
    print("\n\n".join(f"## {k}\n\n{v}" for k, v in tablas.items()))
    if "--escribir" in sys.argv:                 # el libro las incluye con {include} desde tablas/
        os.makedirs("tablas", exist_ok=True)
        for k, v in tablas.items():
            open(f"tablas/e2_{k}.md", "w", encoding="utf-8").write(v + "\n")
