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


# ---------------------------------------------------------------- capítulo 6: implementación, ablación e interpretabilidad

def ablacion():
    t = pd.read_csv(f"{EST}/ablacion.csv")
    q = pd.read_csv(f"{E3}/confirmatorias/V6.csv").parametros.iloc[0]
    params = {v: pd.read_csv(f"{E3}/confirmatorias/{v}.csv").parametros.iloc[0] for v in ["V3", "V4", "V5", "V1"]}
    filas = []
    for _, r in t.iterrows():
        filas.append([r.componente if "componente" in r else r.rival,
                      "RMSE" if r.metrica == "rmse" else "Winkler", f(r.media_base, 4), f(r.media_rival, 4),
                      f"`{r.diferencia:+.4f}`", fp(r.wilcoxon_p_holm), fp(r.dm_p_holm), fp(r.bootstrap_p_holm),
                      fp(r.p_max_holm), f"`{int(params.get(r.rival, q)):,}`".replace(",", " "),
                      "contribuye" if r.rechaza_las_tres else "sin efecto detectable"])
    return tabla(["Componente retirado", "Métrica", "V6 (completo)", "Variante", "Δ (V6 − variante)",
                  "p Wilcoxon (Holm)", "p DM-HLN (Holm)", "p bootstrap (Holm)", "p máx. (Holm)",
                  "Parámetros de la variante", "Conclusión"], filas)


def interpretabilidad():
    e2 = json.load(open("resultados/e2/f2a/gpu_resultados.json"))["interpretabilidad"]
    e3 = json.load(open(f"{E3}/interpretabilidad/resultados.json"))["interpretabilidad"]
    from scipy.stats import spearmanr
    L, anc = 168, [168 - 1 - 24 * j for j in range(7)]
    filas = []
    for m, d in [("PADRE-TCN", e3), ("Transformer", e2["Transformer"]), ("TCN", e2["TCN"]), ("KAN", e2["KAN"])]:
        h = np.array(d["importancia_hora"])
        rng = lambda xs: f"`{min(xs):.2f}`–`{max(xs):.2f}`"
        filas.append([m, rng([x["cambio_top"] for x in d["fidelidad"]]),
                      rng([x["cambio_azar"] for x in d["fidelidad"]]), rng(d["spearman_horas"]),
                      f"`{h[anc].sum() * 100:.0f} %`", f"`{d['importancia_variable'][0] * 100:.0f} %`",
                      "—" if d is e3 else f"`{spearmanr(e3['importancia_hora'], d['importancia_hora'])[0]:.2f}`"])
    return tabla(["Modelo", "Cambio del pronóstico al borrar el 10 % de horas más importantes (m/s)",
                  "Al borrar horas al azar", "Estabilidad entre semillas (Spearman)",
                  "Atribución en las 7 horas ancla (τ − 24 j)", "Peso de la velocidad",
                  "Spearman del perfil horario con PADRE-TCN"], filas)


def anclas():
    a = json.load(open(f"{E3}/interpretabilidad/resultados.json"))["anclas"]
    W = np.array([v["peso_medio"] for v in a.values()])
    Wf = np.array([v["peso_medio_fuerte"] for v in a.values()])
    F = np.array([v["frecuencia_mayor_peso"] for v in a.values()])
    filas = [[f"j = {j + 1} (hace {j + 1} día{'s' if j else ''})", f"`{W[:, j].mean():.3f}` ± `{W[:, j].std():.3f}`",
              f"`{Wf[:, j].mean():.3f}` ± `{Wf[:, j].std():.3f}`", f"`{F[:, j].mean() * 100:.1f} %`"]
             for j in range(W.shape[1])]
    return tabla(["Ancla", "Peso medio wⱼ (todas las ventanas)", "Peso medio con viento ≥ 6 m/s",
                  "Ventanas en que es el ancla de mayor peso"], filas)


# ---------------------------------------------------------------- capítulo 8: evaluación ampliada

AD = f"{E3}/adicionales/analisis.json"
AMP = f"{E3}/ampliada.json"
NOMC = {"etth1": "ETTh1", "sdwpf": "SDWPF", "noaa_kansas": "NOAA Kansas"}


def adicionales_resumen():
    a = json.load(open(AD, encoding="utf-8"))
    filas = []
    for c, n in NOMC.items():
        r = a[c]
        t = pd.DataFrame(r["contrastes"])
        dec = 1 if c == "sdwpf" else 3
        filas.append([n, f(r["rmse_padre"], dec), f"{r['rango_padre']} de {r['de']}", f"{nom(r['mejor'])} ({f(r['rmse_mejor'], dec)})",
                      f(r["rmse_persistencia"], dec), str(int((t.padre_mejor & t.significativo).sum())),
                      str(int((~t.padre_mejor & t.significativo).sum())),
                      ", ".join(nom(x) for x in r["peor_significativo"]) or "—"])
    return tabla(["Conjunto", "RMSE de PADRE-TCN", "Puesto entre 11", "Mejor modelo (RMSE)", "Persistencia",
                  "Rivales que supera (significativo)", "Rivales que lo superan (significativo)", "Cuáles lo superan"],
                 filas)


def adicionales_contrastes():
    a = json.load(open(AD, encoding="utf-8"))
    filas = []
    for c, n in NOMC.items():
        for r in a[c]["contrastes"]:
            concl = ("PADRE mejor" if r["padre_mejor"] else "rival mejor") if r["significativo"] else "sin diferencia"
            filas.append([n, nom(r["rival"]), str(r["pares"]), f"`{r['diferencia_relativa'] * 100:+.1f} %`",
                          fp(r["t_p_holm"]), fp(r["wilcoxon_p_holm"]), concl])
    return tabla(["Conjunto", "Rival", "Pares", "Δ RMSE relativo (PADRE − rival)", "p t pareado (Holm)",
                  "p Wilcoxon (Holm)", "Conclusión"], filas)


def friedman():
    fr = json.load(open(AD, encoding="utf-8"))["friedman"]
    filas = [[("PADRE-TCN" if m == "PADRE" else nom(m)), f(v, 2)] for m, v in fr["rangos_medios"].items()]
    return (tabla(["Modelo", "Rango medio en los 4 conjuntos (1 = mejor)"], filas) +
            f"\nFriedman: χ² = `{fr['chi2']:.2f}`; Iman–Davenport: F = `{fr['iman_davenport_F']:.2f}`, "
            f"p = `{fr['p']:.2f}`; diferencia crítica de Nemenyi al 95 %: `{fr['diferencia_critica_nemenyi']:.2f}` rangos.\n")


def sensibilidad():
    s = json.load(open(AMP, encoding="utf-8"))["sensibilidad_hp"]
    filas = [[("PADRE-TCN" if m == "PADRE" else m), f(v["minimo"], 4), f(v["mediana"], 4), f(v["maximo"], 4),
              f(v["rango"], 4), f"`{v['rho_lr']:+.2f}`"] for m, v in sorted(s.items(), key=lambda x: x[1]["rango"])]
    return tabla(["Modelo", "RMSE de validación mínimo", "Mediana", "Máximo", "Rango (máx. − mín.)",
                  "ρ de Spearman con la tasa de aprendizaje"], filas)


def estabilidad():
    s = json.load(open(AMP, encoding="utf-8"))["estabilidad_semillas"]
    filas = [[("PADRE-TCN" if m == "PADRE" else m), f(v["media"], 4), f(v["de"], 4), f"`{v['cv'] * 100:.2f} %`",
              fp(v["friedman_semillas_p"])] for m, v in sorted(s.items(), key=lambda x: x[1]["cv"])]
    return tabla(["Modelo", "RMSE medio", "DE entre semillas", "Coeficiente de variación",
                  "p de Friedman entre semillas (orígenes como bloques)"], filas)


def regimenes():
    r = json.load(open(AMP, encoding="utf-8"))["regimenes"]
    filas = []
    for k in ["0–2", "2–4", "4–6", "≥ 6"]:
        fila = [f"{k} m/s", f"`{r['PADRE']['por_rango'][k]['n']:,}`".replace(",", " ")]
        for m in ["PADRE", "TCN", "Transformer", "Persistencia"]:
            v = r[m]["por_rango"][k]
            fila.append(f"{f(v['rmse'])} / `{v['sesgo']:+.2f}`")
        filas.append(fila)
    return tabla(["Velocidad observada", "Ventanas", "PADRE-TCN: RMSE / sesgo", "TCN", "Transformer", "Persistencia"],
                 filas)


def trimestres():
    r = json.load(open(AMP, encoding="utf-8"))["regimenes"]
    cob = json.load(open(f"{EST}/resumen.json"))["exploratorias"]
    filas = [[o[:7], *[f(r[m]["por_trimestre"][o], 4) for m in ["PADRE", "TCN", "Transformer", "Persistencia"]],
              f(cob["cobertura_por_trimestre"][o], 3), f(cob["cobertura_std24_por_trimestre"][o], 3)]
             for o in ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]]
    return tabla(["Trimestre de prueba", "PADRE-TCN", "TCN", "Transformer", "Persistencia",
                  "Cobertura del intervalo al 90 % (PADRE-TCN)", "Cobertura con std24"], filas)


def perturbaciones():
    e2 = json.load(open("resultados/e2/f2a/gpu_resultados.json"))["robustez"]
    e3 = json.load(open(f"{E3}/interpretabilidad/resultados.json"))["robustez"]
    datos = {"PADRE-TCN": e3, "Transformer": e2["Transformer"], "TCN": e2["TCN"]}
    med = {m: pd.DataFrame(v).groupby(["tipo", "nivel"]).rmse.mean() for m, v in datos.items()}
    nombres = {"ruido": "Ruido gaussiano (σ en unidades escaladas)", "faltantes": "Horas faltantes (fracción)",
               "quiebre": "Quiebre de nivel (desviaciones)"}
    filas = []
    for (tipo, nivel) in med["PADRE-TCN"].index:
        if tipo == "ninguna":
            continue
        filas.append([nombres[tipo], f"{nivel}"] +
                     [f"{f(med[m][(tipo, nivel)])} (`{(med[m][(tipo, nivel)] / med[m][('ninguna', 0.0)] - 1) * 100:+.0f} %`)"
                      for m in datos])
    return tabla(["Perturbación de la entrada", "Severidad", "PADRE-TCN: RMSE (aumento)", "Transformer", "TCN"], filas)


def errores():
    e = json.load(open(AMP, encoding="utf-8"))["errores_compartidos"]
    filas = [[m, f(v["correlacion"], 3), f"`{v['solape_peor_5'] * 100:.0f} %`"] for m, v in e.items()]
    return tabla(["Modelo comparado con PADRE-TCN", "Correlación de los residuos",
                  "Peores 5 % de PADRE-TCN que también son peores 5 % del otro"], filas)


TABLAS = dict(diseno=diseno, complejidad=complejidad, verificacion=verificacion, h1=h1, h2a=h2a, pruebas=pruebas,
              h2b=h2b, mcs=mcs_tabla, potencia=potencia, r3=r3, adecuacion=adecuacion, equidad=equidad,
              ablacion=ablacion, interpretabilidad=interpretabilidad, anclas=anclas,
              adicionales_resumen=adicionales_resumen, adicionales_contrastes=adicionales_contrastes,
              friedman=friedman, sensibilidad=sensibilidad, estabilidad=estabilidad, regimenes=regimenes,
              trimestres=trimestres, perturbaciones=perturbaciones, errores=errores)

if __name__ == "__main__":
    os.makedirs("book/tablas", exist_ok=True)
    for n in (sys.argv[1:] or TABLAS):
        t = TABLAS[n]()
        open(f"book/tablas/e3_{n}.md", "w", encoding="utf-8").write(t)
        print(f"== book/tablas/e3_{n}.md\n{t}")
