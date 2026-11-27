"""Tablas en Markdown para los capítulos del E3, armadas directamente desde los archivos de resultados (los números del
libro no se copian a mano). Cada tabla se escribe en book/tablas/e3_{nombre}.md y el capítulo la incluye con {include}.
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


# ---------------------------------------------------------------- capítulo 7: validación estadística

EST = f"{E3}/estadistica"
FAMILIA = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST",
           "Ridge", "Gradient_boosting"]
nom = lambda m: m.replace("_", " ")
fp = lambda p: "`< 0.001`" if p < 0.001 else f"`{p:.3f}`"
Z_POT = 2.865 + 1.282          # z(1 − 0.05/24) + z(0.90): Bonferroni sobre 12 contrastes, potencia 0.90


def _contrastes(ruta, unidad):
    t = pd.read_csv(ruta).sort_values("media_rival")
    filas = []
    for _, r in t.iterrows():
        filas.append([nom(r.rival), f(r.media_rival), f(r.diferencia, 4), f"`{r.cliff:+.3f}` ({r.efecto_cliff})",
                      f"[`{r.ic95_serie_inf:+.4f}`, `{r.ic95_serie_sup:+.4f}`]", f"`{r.dm_estadistico:+.2f}`",
                      fp(r.wilcoxon_p_holm), fp(r.dm_p_holm), fp(r.bootstrap_p_holm), fp(r.p_max_holm),
                      ("rechaza" if r.rechaza_las_tres else "no rechaza")
                      + ("; a favor de PADRE" if r.rechaza_las_tres and r.diferencia < 0 else
                         ("; a favor del rival" if r.rechaza_las_tres else ""))])
    return tabla(["Rival", f"{unidad} del rival", "Δ (PADRE − rival)", "δ de Cliff (estaciones)",
                  "IC 95 % de Δ en la serie horaria (bootstrap estacionario)", "Estadístico DM-HLN",
                  "p Wilcoxon (Holm)", "p DM-HLN (Holm)", "p bootstrap (Holm)", "p máx. (Holm)", "Conclusión"],
                 filas)


def h1():
    return _contrastes(f"{EST}/H1.csv", "RMSE")


def h2a():
    return _contrastes(f"{EST}/H2a.csv", "Winkler")


def pruebas():
    """Plantilla del Apéndice A: una fila por prueba, con rango sobre los 12 contrastes de H1."""
    t = pd.read_csv(f"{EST}/H1.csv")
    rango = lambda c: f"{fp(t[c].min())} a {fp(t[c].max())}"
    n = lambda c: f"{int((t[c] < 0.05).sum())} de 12"
    filas = [
        ["Wilcoxon de rangos con signo", "La mediana de la diferencia de RMSE por estación es 0",
         "Unidad independiente: la estación (62); diferencia simétrica alrededor de la mediana",
         "W (por contraste)", rango("wilcoxon_p"), rango("wilcoxon_p_holm"), "δ de Cliff por estación",
         f"Rechaza en {n('wilcoxon_p_holm')}"],
        ["Diebold–Mariano con corrección HLN", "Igual pérdida cuadrática esperada",
         f"Serie de diferencias estacionaria (ADF: p ≤ {fp(t.adf_p.max())}); autocorrelación tratada con HAC de 23 rezagos "
         f"(Ljung–Box(24) de la diferencia: p ≤ {fp(t.ljungbox24_p.max())})",
         f"DM-HLN de {t.dm_estadistico.min():+.2f} a {t.dm_estadistico.max():+.2f}", rango("dm_p"), rango("dm_p_holm"),
         "Δ medio de la pérdida", f"Rechaza en {n('dm_p_holm')}"],
        ["Bootstrap estacionario pareado (Politis y Romano)", "Diferencia media de pérdida igual a 0",
         "Dependencia temporal: bloques de media 168 h; 10 000 réplicas, mismas réplicas para todos los rivales",
         "Δ medio de la pérdida horaria", rango("bootstrap_p"), rango("bootstrap_p_holm"), "IC 95 % de Δ",
         f"Rechaza en {n('bootstrap_p_holm')}"],
        ["t pareado sobre semillas (robustez)", "Δ RMSE medio por semilla igual a 0",
         "10 semillas (media de 4 orígenes); normalidad aproximada", "t con 9 g. l.", rango("t_semillas_p"),
         rango("t_semillas_p_holm"), "d de Cohen", f"Rechaza en {n('t_semillas_p_holm')}"],
        ["t con corrección de Nadeau–Bengio (robustez)", "Igual que la anterior",
         "Varianza inflada por la razón prueba/entrenamiento (`0.099`)", "t corregido", rango("t_nb_p"),
         rango("t_nb_p_holm"), "d de Cohen", f"Rechaza en {n('t_nb_p_holm')}"],
        ["Bootstrap por bloques de una semana (robustez)", "Diferencia media de pérdida igual a 0",
         "Bloques fijos de 168 h", "Δ medio de la pérdida horaria", rango("bootstrap_semana_p"),
         rango("bootstrap_semana_p_holm"), "IC 95 % de Δ", f"Rechaza en {n('bootstrap_semana_p_holm')}"],
    ]
    return tabla(["Prueba", "Hipótesis nula", "Supuestos verificados", "Estadístico", "p crudo (rango)",
                  "p Holm (rango)", "Tamaño del efecto", "Conclusión (α = 0.05)"], filas)


def h2b():
    r = json.load(open(f"{EST}/resumen.json"))["H2b"]
    return tabla(["Pares (semilla, origen)", "Δ RMSE medio PADRE − TCN (m/s)", "IC 90 %", "Margen ±δ",
                  "p TOST", "Conclusión"],
                 [[str(r["n_pares"]), f(r["diferencia_media"], 4), f"[`{r['ic90'][0]:.4f}`, `{r['ic90'][1]:.4f}`]",
                   f"`±{r['delta']}`", fp(r["p"]), "equivalentes" if r["equivalentes"] else "no equivalentes"]])


def mcs_tabla():
    r = json.load(open(f"{EST}/resumen.json"))["mcs"]
    filas = [[nom(m), fp(p) if p > 0 else "`< 0.001`", "sí" if m in r["incluidos"] else "no"]
             for m, p in sorted(r["pvalores"].items(), key=lambda x: -x[1])]
    return tabla(["Modelo", "p del MCS", f"En el MCS al {int(r['nivel'] * 100)} %"], filas)


def potencia():
    t = pd.read_csv(f"{EST}/H1.csv")
    filas = []
    for _, r in t.iterrows():
        sd = abs(r.diferencia / r.d_cohen_semillas)
        filas.append([nom(r.rival), f(sd, 4), f(Z_POT * sd / np.sqrt(10), 4),
                      "`< 1`" if Z_POT ** 2 * sd ** 2 / 0.02 ** 2 < 1 else f(Z_POT ** 2 * sd ** 2 / 0.02 ** 2, 1)])
    return tabla(["Rival", "σ_d observada entre semillas (m/s)", "Diferencia mínima detectable con 10 semillas (m/s)",
                  "Semillas necesarias para δ = 0.02"], filas)


def r3():
    r = json.load(open(f"{EST}/resumen.json"))["H1"]["regla_R3"]
    a = json.load(open(f"{E3}/adecuacion.json"))["PADRE"]
    ok = lambda v: "✅ se cumple" if v is True else ("❌ no se cumple" if v is False else "⏳ " + str(v))
    filas = [
        ["1. Menor RMSE medio que los 12 rivales", ok(r["c1_menor_rmse_que_los_12"]),
         "PADRE (`0.7239`) es mejor que 2 de 12; el Transformer es el mejor (`0.7189`)"],
        ["2. Tres pruebas rechazan en los 12 contrastes, con Δ > δ", ok(r["c2_tres_pruebas_y_delta"]),
         "Las tres rechazan en 3 contrastes; solo en 2 (Chronos y Moirai) a favor de PADRE con \|Δ\| > δ"],
        ["3. Efecto no despreciable y límite inferior del IC > 0", ok(r["c3_ic_inferior_mayor_que_0"]), "—"],
        ["4. Mejor en ≥ 2 de 3 conjuntos adicionales", ok(r["c4_conjuntos_adicionales"]),
         "Se completa con el capítulo 8"],
        ["5a. Supera a la persistencia", ok(r["c5_supera_persistencia"]), "Δ = `−0.150` m/s, las tres pruebas rechazan"],
        ["5b. Brecha entrenamiento–parada menor que δ", ok(r["c5_brecha_menor_que_delta"]),
         f"Brecha media `{a['brecha_absoluta_media']:+.3f}` m/s (gap_ratio mediano `{a['gap_ratio_mediana']:.3f}`)"],
        ["6. La conclusión no cambia con las pruebas alternativas", ok(r["c6_robusta"]),
         "t pareado, t de Nadeau–Bengio y bootstrap semanal dan la misma decisión en los 12 contrastes"],
    ]
    return tabla(["Condición (sección 8.4)", "Estado", "Evidencia"], filas)


def adecuacion():
    a = json.load(open(f"{E3}/adecuacion.json"))
    fila = lambda m: [m, f(a[m]["rmse_entrenamiento_medio"], 4), f(a[m]["rmse_parada_medio"], 4),
                      f(a[m]["rmse_prueba_medio"], 4), f"`{a[m]['brecha_absoluta_media']:+.4f}`",
                      f"{f(a[m]['gap_ratio_mediana'])} / {f(a[m]['gap_ratio_max'])}",
                      f"a = `{a[m]['mincer_zarnowitz']['a']:+.3f}`, b = `{a[m]['mincer_zarnowitz']['b']:.3f}`, "
                      f"p = {fp(a[m]['mincer_zarnowitz']['p_conjunto'])}"]
    t1 = tabla(["Modelo", "RMSE entrenamiento", "RMSE parada (validación)", "RMSE prueba",
                "Brecha parada − entrenamiento (m/s)", "gap_ratio mediano / máximo", "Mincer–Zarnowitz (HAC)"],
               [fila("PADRE"), fila("TCN")])
    pr = lambda m, k: f"`{a[m]['proporcion_rechaza_5'][k] * 100:.0f} %`"
    t2 = tabla(["Prueba sobre los residuos (por estación, 61 estaciones)", "PADRE", "TCN"],
               [[n, pr("PADRE", k), pr("TCN", k)] for n, k in
                [("Ljung–Box(24): autocorrelación", "ljung_box"), ("ARCH-LM(24): heterocedasticidad", "arch_lm"),
                 ("Jarque–Bera: normalidad", "jarque_bera"), ("BDS: independencia no lineal", "bds"),
                 ("Mincer–Zarnowitz: a = 0, b = 1", "mz")]])
    return t1 + "\n" + t2


def equidad():
    c = pd.read_csv(f"{E3}/costo_cpu.csv").set_index("modelo")
    filas = []
    for m in ["PADRE"] + FAMILIA[:10]:
        csv = f"{E3}/confirmatorias/V6.csv" if m == "PADRE" else f"resultados/e2/finales/{m}.csv"
        d = pd.read_csv(csv)
        d = d[d.semilla.between(1000, 1009)]
        filas.append([m, f"`{int(c.loc[m, 'parametros']):,}`".replace(",", " "), f(c.loc[m, "mflops_ventana"], 1 if c.loc[m, "mflops_ventana"] >= 1 else 3),
                      f(c.loc[m, "ms_ventana_lote1024"] * 1e3, 0 if c.loc[m, "ms_ventana_lote1024"] >= 0.01 else 1), str(len(d)), f(d.epocas.mean(), 1),
                      f(d.segundos.mean() / 60, 1)])
    return tabla(["Modelo", "Parámetros", "MFLOPs por ventana", "µs por ventana (CPU, lote 1 024)",
                  "Corridas (semillas × orígenes)", "Épocas medias", "Minutos por corrida"], filas)


TABLAS = dict(diseno=diseno, complejidad=complejidad, verificacion=verificacion, h1=h1, h2a=h2a, pruebas=pruebas,
              h2b=h2b, mcs=mcs_tabla, potencia=potencia, r3=r3, adecuacion=adecuacion, equidad=equidad)

if __name__ == "__main__":
    os.makedirs("book/tablas", exist_ok=True)
    for n in (sys.argv[1:] or TABLAS):
        t = TABLAS[n]()
        open(f"book/tablas/e3_{n}.md", "w", encoding="utf-8").write(t)
        print(f"== book/tablas/e3_{n}.md\n{t}")
