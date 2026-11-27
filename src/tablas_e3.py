"""Tablas en Markdown para los capítulos del E3, armadas directamente desde los archivos de resultados (los números del
libro no se copian a mano). Cada tabla se escribe en tablas/e3_{nombre}.md y el capítulo la incluye con {include}.
Uso: python src/tablas_e3.py [nombre ...]      (ml_env; sin nombres, todas)
"""
import json, os, sys
import numpy as np
import pandas as pd

E3 = "resultados/e3"
f = lambda x, n=3: f"`{x:.{n}f}`"
NOMBRES_DISENO = {"V0": "TCN del E2 (referencia)", "V1": "Completo (K = 6, λ = 0.5)", "V2": "K = 3",
                  "V3": "K = 1 (solo persistencia)", "V4": "Pesos uniformes", "V5": "Sin antisimetría",
                  "V6": "λ = 0 (diseño congelado)"}


def tabla(encabezado, filas):
    return "\n".join(["| " + " | ".join(encabezado) + " |", "|" + "---|" * len(encabezado)]
                     + ["| " + " | ".join(r) + " |" for r in filas]) + "\n"


def diseno():
    """Ronda 2 de diseño en la partición de desarrollo (3 orígenes de validación de 2024, semilla 2001)."""
    d = pd.read_csv(f"{E3}/diseno_r2_analisis.csv")
    g = d.groupby("variante").agg(rmse=("rmse", "mean"), sesgo_fuerte=("sesgo_fuerte", "mean"),
                                  rho=("rho_disp_err", "mean"))
    filas = [[f"{v} · {NOMBRES_DISENO[v]}", f(r.rmse, 4), f(r.sesgo_fuerte, 2),
              "—" if np.isnan(r.rho) else f(r.rho, 2)] for v, r in g.iterrows()]
    return tabla(["Variante", "RMSE (m/s)", "Sesgo con viento ≥ 6 m/s (m/s)", r"ρ dispersión–\|error\|"], filas)


def complejidad():
    d = pd.read_csv(f"{E3}/complejidad_padre.csv")
    filas = []
    for _, r in d.iterrows():
        filas.append(["longitud L" if r.serie == "longitud" else "ancho c", str(int(r.L)), str(int(r.ancho)),
                      f"`{int(r.parametros):,}`".replace(",", " "), f(r.mflops_ventana, 1),
                      f(r.ms_ventana_lote1024 * 1e3, 0)])
    pend = []
    for serie, var in [("longitud", "L"), ("ancho", "ancho")]:
        s = d[d.serie == serie]
        pend.append([("L" if serie == "longitud" else "c"),
                     *[f(np.polyfit(np.log(s[var]), np.log(s[c]), 1)[0], 2)
                       for c in ("parametros", "mflops_ventana", "ms_ventana_lote1024")]])
    return (tabla(["Serie", "L (h)", "Ancho c", "Parámetros", "MFLOPs por ventana", "µs por ventana (lote 1 024)"],
                  filas) + "\n" + tabla(["Variable", "Pendiente: parámetros", "Pendiente: FLOPs", "Pendiente: tiempo"],
                                        pend))


def verificacion():
    v = json.load(open(f"{E3}/verificacion_propiedades.json"))
    cien = lambda x: "`0` (exacto)" if x == 0 else f"`{x:.1e}`"
    filas = [
        ["Prop. 1: máx \\|D(q, aⱼ) + D(aⱼ, q)\\|", cien(v["float32"]["max_abs_D_qa_mas_D_aq"]),
         cien(v["float64"]["max_abs_D_qa_mas_D_aq"])],
        ["Prop. 1: máx \\|D(aⱼ, aⱼ)\\|", cien(v["float32"]["max_abs_D_aa"]), cien(v["float64"]["max_abs_D_aa"])],
        ["Escala: máx \\|D(q, aⱼ)\\| (m/s)", f(v["float32"]["max_abs_D_qa"], 2), f(v["float64"]["max_abs_D_qa"], 2)],
        ["Prop. 2: peor exceso sobre [minⱼ ŷⱼ, maxⱼ ŷⱼ] (m/s; negativo = dentro)",
         f(v["float32"]["peor_violacion_cota_convexa"], 4), f(v["float64"]["peor_violacion_cota_convexa"], 4)],
        ["Prop. 2: peso mínimo wⱼ", f(v["float32"]["peso_minimo"], 4), f(v["float64"]["peso_minimo"], 4)],
        ["Prop. 2: máx \\|Σⱼ wⱼ − 1\\|", cien(v["float32"]["max_abs_suma_pesos_menos_1"]),
         cien(v["float64"]["max_abs_suma_pesos_menos_1"])],
        ["Prop. 2: dispersión con todas las ŷⱼ iguales", cien(v["float32"]["dispersion_con_anclas_iguales"]),
         cien(v["float64"]["dispersion_con_anclas_iguales"])],
        ["Prop. 2: dispersión mínima en datos reales (m/s)", f(v["float32"]["dispersion_minima"], 4),
         f(v["float64"]["dispersion_minima"], 4)],
    ]
    return tabla(["Comprobación", "float32", "float64"], filas)


TABLAS = dict(diseno=diseno, complejidad=complejidad, verificacion=verificacion)

if __name__ == "__main__":
    os.makedirs("tablas", exist_ok=True)
    for n in (sys.argv[1:] or TABLAS):
        t = TABLAS[n]()
        open(f"tablas/e3_{n}.md", "w", encoding="utf-8").write(t)
        print(f"== tablas/e3_{n}.md\n{t}")
