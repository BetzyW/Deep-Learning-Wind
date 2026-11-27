"""Validación estadística confirmatoria del E3 (experiments/protocolo.md, tag e3-protocolo, secciones 3, 4, 7, 8 y 11).

Entradas (prueba de 2025, 4 orígenes):
  * PADRE-TCN y su ablación: resultados/e3/confirmatorias/{V6,V3,V4,V5,V1}/{origen}_{semilla}.npz (tau, pred, disp);
  * la familia de 12 rivales del E2: resultados/e2/finales/{modelo}/{origen}_{semilla}.npz, y {origen}.npz para Ridge y
    gradient boosting (deterministas: la misma predicción sirve para todas las semillas). Semillas 1000–1009.

Qué calcula:
  * H1 (12 contrastes, Holm por prueba): Wilcoxon sobre la diferencia de RMSE por estación; Diebold–Mariano con HLN
    sobre la serie horaria de pérdida cuadrática media entre estaciones y semillas (HAC, 23 rezagos); bootstrap
    estacionario pareado (bloque medio 168 h, 10 000 réplicas) de la diferencia de pérdida media. Decisión: el máximo de
    los tres p ajustados < 0.05 y diferencia mayor que δ. Tamaños del efecto (δ de Cliff, d de Cohen sobre semillas),
    robustez (t pareado sobre semillas con y sin Nadeau–Bengio, bootstrap por bloques fijos de una semana), supuestos
    (ADF y Ljung–Box) y el Model Confidence Set al 90 %.
  * H2a: lo mismo con el interval score de Winkler al 90 %. Intervalos conformales (split entre orígenes: el cuantil de
    cada origen se calibra con los residuos normalizados de los otros tres, misma semilla). Normalizador: PADRE
    s = √(disp² + std24²); rivales s = std24 (desviación del viento en las 24 h finales de la entrada). Para evitar s = 0
    en horas de calma se usa s + ε con ε = 0.1 × la media de s en la calibración (la misma regla de la ronda 2 de
    diseño, analisis_diseno_r2.py), igual para todos los modelos.
  * H2b: TOST (±δ) del RMSE de PADRE frente a la TCN sobre los 40 pares (semilla, origen).
  * Ablación (familia propia, 5 contrastes con Holm): V6 frente a V3, V4, V5 y V1 en RMSE con las tres pruebas, y la
    normalización combinada frente a std24 (V6, sin reentrenar) en Winkler.
  * Exploratorias del mecanismo: Spearman entre dispersión y |error|, cobertura total y por trimestre, sesgo con viento
    ≥ 6 m/s.

Decisiones de implementación (se declaran en el informe):
  * En la prueba no todas las estaciones tienen ventanas en cada origen (40 a 56 por origen). El RMSE por estación se
    calcula juntando las ventanas de los 4 orígenes por semilla y se promedia sobre las semillas; entran las estaciones
    con ventanas en la prueba.
  * La serie horaria es la del instante de emisión; las horas sin ventanas no existen en la serie (6 630 horas).
  * Nadeau–Bengio: factor (1/J + n_prueba/n_entrenamiento), con la razón media de los 4 orígenes.
Salida: resultados/e3/estadistica/*.csv y resumen.json.
Uso: python src/estadistica_e3.py      (ml_env: necesita arch y statsmodels; solo CPU)
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import diebold_mariano_hln, holm, tost  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*keyword-only.*")
DELTA, ALFA, ALFA_INT, H, B, BLOQUE = 0.02, 0.05, 0.10, 24, 10_000, 168
SEMILLAS = list(range(1000, 1010))
FAMILIA = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST",
           "Ridge", "Gradient_boosting"]
REFERENCIA = ["Persistencia"]                           # fuera de la familia (condición 5)
ABLACION = ["V3", "V4", "V5", "V1"]
SALIDA = "resultados/e3/estadistica"
FIN, CONF = "resultados/e2/finales", "resultados/e3/confirmatorias"


def cuantil_conformal(r, alfa=ALFA_INT):
    n = len(r)
    return np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alfa)) / n))


def winkler(y, lo, hi, alfa=ALFA_INT):
    return (hi - lo) + (2 / alfa) * (lo - y) * (y < lo) + (2 / alfa) * (y - hi) * (y > hi)


class Agregador:
    """Acumula, para un modelo, todo lo que piden las pruebas sin guardar las predicciones de todas las semillas."""

    def __init__(self, p, tiempos, estaciones):
        self.p, self.T, self.E = p, tiempos, estaciones
        self.e2_hora = np.zeros(len(tiempos)); self.w_hora = np.zeros(len(tiempos)); self.n_hora = np.zeros(len(tiempos))
        self.rmse_est, self.w_est = [], []                   # por semilla: vector por estación
        self.por_corrida = []                                # filas (semilla, origen) con RMSE, Winkler, cobertura...
        self.extra = []                                      # exploratorias (solo PADRE)

    def agregar(self, semilla, por_origen):
        """por_origen: {origen: (tau, pred, y, s)}, con s el normalizador del intervalo en cada ventana."""
        origenes = sorted(por_origen)
        e2_est = np.zeros(len(self.E)); w_est = np.zeros(len(self.E)); n_est = np.zeros(len(self.E))
        for o in origenes:
            tau, pred, y, s = por_origen[o]
            otros = [k for k in origenes if k != o]
            r_cal = np.concatenate([np.abs(por_origen[k][1] - por_origen[k][2]) for k in otros])
            s_cal = np.concatenate([por_origen[k][3] for k in otros])
            eps = 0.1 * s_cal.mean()
            q = cuantil_conformal(r_cal / (s_cal + eps))
            mitad = q * (s + eps)
            lo, hi = pred - mitad, pred + mitad
            e = pred - y
            w = winkler(y, lo, hi)
            it = np.searchsorted(self.T, self.p.tiempo[tau])
            ie = np.searchsorted(self.E, self.p.estacion[tau])
            np.add.at(self.e2_hora, it, e ** 2); np.add.at(self.w_hora, it, w); np.add.at(self.n_hora, it, 1)
            np.add.at(e2_est, ie, e ** 2); np.add.at(w_est, ie, w); np.add.at(n_est, ie, 1)
            dentro = (y >= lo) & (y <= hi)
            fuerte = y >= 6
            self.por_corrida.append(dict(semilla=semilla, origen=o, rmse=float(np.sqrt(np.mean(e ** 2))),
                                         mae=float(np.mean(np.abs(e))), winkler=float(w.mean()),
                                         cobertura=float(dentro.mean()), ancho=float(np.mean(hi - lo)),
                                         sesgo_fuerte=float(e[fuerte].mean()), n_fuerte=int(fuerte.sum()),
                                         cobertura_fuerte=float(dentro[fuerte].mean()),
                                         winkler_fuerte=float(w[fuerte].mean())))
        ok = n_est > 0
        self.rmse_est.append(np.where(ok, np.sqrt(e2_est / np.maximum(n_est, 1)), np.nan))
        self.w_est.append(np.where(ok, w_est / np.maximum(n_est, 1), np.nan))

    def cerrar(self):
        self.perdida_hora = self.e2_hora / self.n_hora
        self.winkler_hora = self.w_hora / self.n_hora
        self.rmse_estacion = np.nanmean(np.vstack(self.rmse_est), 0)
        self.winkler_estacion = np.nanmean(np.vstack(self.w_est), 0)
        self.corridas = pd.DataFrame(self.por_corrida)
        return self


def cargar_modelo(nombre, carpeta, p, y_de, std24_de, tiempos, estaciones, normalizador="std24"):
    """Carga todas las semillas de un modelo y devuelve su Agregador. normalizador: 'std24' o 'combinado' (PADRE)."""
    ag = Agregador(p, tiempos, estaciones)
    det = not glob.glob(f"{carpeta}/{nombre}/*_*.npz")       # baselines: un archivo por origen
    extra = []
    for s in SEMILLAS:
        por_origen = {}
        for a in sorted(glob.glob(f"{carpeta}/{nombre}/*.npz")):
            base = os.path.basename(a)[:-4]
            o, _, sem = base.partition("_")
            if not det and int(sem) != s:
                continue
            z = np.load(a)
            tau, pred = z["tau"].astype(np.int64), z["pred"].astype(np.float64)
            y, s24 = y_de(tau), std24_de(tau)
            if normalizador == "combinado":
                d = z["disp"].astype(np.float64)
                norm = np.sqrt(d ** 2 + s24 ** 2)
                fuerte = y >= 6
                extra.append(dict(semilla=s, origen=o,
                                  rho_disp_err=float(np.nan if d.std() == 0 else stats.spearmanr(d, np.abs(pred - y)).statistic),
                                  rho_disp_err_fuerte=float(np.nan if d.std() == 0 else stats.spearmanr(d[fuerte], np.abs(pred - y)[fuerte])
                                                            .statistic)))
            else:
                norm = s24
            por_origen[o] = (tau, pred, y, norm)
        if len(por_origen) != 4:
            raise SystemExit(f"{nombre}: semilla {s} con {len(por_origen)} orígenes (se esperan 4)")
        ag.agregar(s, por_origen)
    ag.extra = pd.DataFrame(extra)
    return ag.cerrar()


# ---------------------------------------------------------------- bootstrap pareado (mismas réplicas para todos)


def pesos_estacionario(n, B, bloque, rng, trozo=250):
    """Matrices (trozo, n) de conteos de remuestreo del bootstrap estacionario (Politis y Romano, 1994)."""
    p_salto = 1 / bloque
    for i in range(0, B, trozo):
        b = min(trozo, B - i)
        nuevo = rng.random((b, n)) < p_salto
        nuevo[:, 0] = True
        inicio = rng.integers(0, n, (b, n))
        # índice = inicio del bloque vigente + pasos desde ese inicio (circular)
        pos = np.arange(n)[None, :].repeat(b, 0)
        ult = np.where(nuevo, pos, 0)
        np.maximum.accumulate(ult, axis=1, out=ult)
        idx = (np.take_along_axis(inicio, ult, 1) + pos - ult) % n
        W = np.zeros((b, n))
        np.add.at(W, (np.arange(b)[:, None].repeat(n, 1), idx), 1)
        yield W


def pesos_bloques_fijos(n, B, bloque, rng, trozo=250):
    """Bootstrap por bloques móviles de longitud fija (una semana), circular."""
    k = int(np.ceil(n / bloque))
    for i in range(0, B, trozo):
        b = min(trozo, B - i)
        ini = rng.integers(0, n, (b, k))
        idx = ((ini[:, :, None] + np.arange(bloque)[None, None, :]).reshape(b, -1)[:, :n]) % n
        W = np.zeros((b, n))
        np.add.at(W, (np.arange(b)[:, None].repeat(n, 1), idx), 1)
        yield W


def bootstrap_pareado(D, generador):
    """D: (n, k) diferencias de pérdida por hora. Devuelve p bilateral (centrado) e IC 95 % de cada media."""
    n = D.shape[0]
    medias = np.vstack([W @ D / n for W in generador])
    m = D.mean(0)
    p = ((np.abs(medias - m) >= np.abs(m)).sum(0) + 1) / (len(medias) + 1)
    return p, np.quantile(medias, 0.025, 0), np.quantile(medias, 0.975, 0)


def cliff(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def pruebas_familia(nombre_fam, base, rivales, metrica, ags, n_ratio, rng_semilla):
    """Las tres pruebas del protocolo, con Holm por prueba sobre la familia. metrica: 'rmse' o 'winkler'.
    Diferencias = base − rival (negativas favorecen a la base)."""
    A = ags[base]
    serie = (lambda g: g.perdida_hora) if metrica == "rmse" else (lambda g: g.winkler_hora)
    porest = (lambda g: g.rmse_estacion) if metrica == "rmse" else (lambda g: g.winkler_estacion)
    D = np.column_stack([serie(A) - serie(ags[r]) for r in rivales])
    ok = np.isfinite(D).all(1)
    D = D[ok]
    p_be, lo_be, hi_be = bootstrap_pareado(D, pesos_estacionario(len(D), B, BLOQUE, np.random.default_rng(rng_semilla)))
    p_bf, lo_bf, hi_bf = bootstrap_pareado(D, pesos_bloques_fijos(len(D), B, BLOQUE, np.random.default_rng(rng_semilla + 1)))
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    filas = []
    for j, r in enumerate(rivales):
        R = ags[r]
        a_est, r_est = porest(A), porest(R)
        m = np.isfinite(a_est) & np.isfinite(r_est)
        d_est = a_est[m] - r_est[m]
        w = stats.wilcoxon(d_est, alternative="two-sided")
        # DM-HLN: diebold_mariano_hln eleva al cuadrado su entrada, así que se le pasa la raíz de la serie de pérdida
        dm = diebold_mariano_hln(np.sqrt(serie(A)[ok]), np.sqrt(serie(R)[ok]))
        # semillas: métrica media de los 4 orígenes por semilla (40 pares → 10 semillas)
        ca = A.corridas.groupby("semilla")[metrica].mean()
        cr = R.corridas.groupby("semilla")[metrica].mean().reindex(ca.index)
        dz = (ca - cr).to_numpy()
        J = len(dz)
        t_cl = dz.mean() / (dz.std(ddof=1) / np.sqrt(J)) if dz.std(ddof=1) > 0 else np.inf * np.sign(dz.mean())
        t_nb = dz.mean() / np.sqrt((1 / J + n_ratio) * dz.var(ddof=1)) if dz.std(ddof=1) > 0 else t_cl
        par = A.corridas.set_index(["semilla", "origen"])[metrica] - \
            R.corridas.set_index(["semilla", "origen"])[metrica]
        diff = float(A.corridas[metrica].mean() - R.corridas[metrica].mean())
        filas.append(dict(
            familia=nombre_fam, base=base, rival=r, metrica=metrica,
            media_base=float(A.corridas[metrica].mean()), media_rival=float(R.corridas[metrica].mean()),
            diferencia=diff, diferencia_hora=float(D[:, j].mean()),
            # IC 95 % del bootstrap estacionario de la diferencia de la serie horaria: en m²/s² (pérdida cuadrática)
            # para el RMSE y en m/s para el Winkler
            ic95_serie_inf=float(lo_be[j]), ic95_serie_sup=float(hi_be[j]),
            n_estaciones=int(m.sum()), wilcoxon_p=float(w.pvalue), diferencia_mediana_estacion=float(np.median(d_est)),
            cliff=cliff(a_est[m], r_est[m]),
            dm_estadistico=dm["estadistico"], dm_p=dm["p"], n_horas=int(len(D)),
            bootstrap_p=float(p_be[j]),
            d_cohen_semillas=float(dz.mean() / dz.std(ddof=1)) if dz.std(ddof=1) > 0 else np.nan,
            t_semillas_p=float(2 * stats.t.sf(abs(t_cl), J - 1)), t_nb_p=float(2 * stats.t.sf(abs(t_nb), J - 1)),
            bootstrap_semana_p=float(p_bf[j]), bootstrap_semana_ic95=[float(lo_bf[j]), float(hi_bf[j])],
            pares_semilla_origen=int(par.notna().sum()),
            adf_p=float(adfuller(D[:, j], autolag="AIC")[1]),
            ljungbox24_p=float(acorr_ljungbox(D[:, j] - D[:, j].mean(), lags=[24]).lb_pvalue.iloc[0])))
    t = pd.DataFrame(filas)
    for c in ["wilcoxon_p", "dm_p", "bootstrap_p", "t_semillas_p", "t_nb_p", "bootstrap_semana_p"]:
        t[c + "_holm"] = holm(t[c])
    t["p_max_holm"] = t[["wilcoxon_p_holm", "dm_p_holm", "bootstrap_p_holm"]].max(1)
    t["rechaza_las_tres"] = t.p_max_holm < ALFA
    t["mejor_con_relevancia"] = t.rechaza_las_tres & (t.diferencia < -DELTA)
    # robusta: la decisión (rechaza o no) es la misma con las tres pruebas alternativas (condición 6)
    t["robusta"] = (t[["t_semillas_p_holm", "t_nb_p_holm", "bootstrap_semana_p_holm"]].max(1) < ALFA) == \
        t.rechaza_las_tres
    t["efecto_cliff"] = pd.cut(t.cliff.abs(), [-1, .147, .33, .474, 1.01],
                               labels=["despreciable", "pequeño", "mediano", "grande"]).astype(str)
    return t


def mcs(perdidas, alfa=0.10, reps=2000, semilla=0):
    from arch.bootstrap import MCS
    m = MCS(perdidas, size=alfa, reps=reps, block_size=BLOQUE, method="max", seed=semilla)
    m.compute()
    return dict(nivel=1 - alfa, reps=reps, bloque=BLOQUE, incluidos=list(m.included), excluidos=list(m.excluded),
                pvalores=m.pvalues["Pvalue"].round(4).to_dict())


if __name__ == "__main__":
    from origen_movil import leer_config, pliegues, preparar_panel
    cfg = leer_config()
    p = preparar_panel(cfg)
    fs = pliegues(p, cfg, "prueba")
    ev_todas = np.concatenate([np.asarray(f["evaluacion"]) for f in fs])
    n_ratio = float(np.mean([len(f["evaluacion"]) / len(f["entrenamiento"]) for f in fs]))
    tiempos = np.unique(p.tiempo[ev_todas])
    estaciones = np.unique(p.estacion[ev_todas])
    h = p.horizontes[0]
    y_de = lambda tau: p.X[tau + h, 0].astype(np.float64)                      # noqa: E731
    v = p.X[:, 0].astype(np.float64)
    std24_de = lambda tau: np.std(v[tau[:, None] + np.arange(-23, 1)[None, :]], axis=1)  # noqa: E731
    os.makedirs(SALIDA, exist_ok=True)

    ags = {}
    for m in FAMILIA + REFERENCIA:
        print("cargando", m, flush=True)
        ags[m] = cargar_modelo(m, FIN, p, y_de, std24_de, tiempos, estaciones)
    for var in ["V6"] + ABLACION:
        print("cargando", var, flush=True)
        ags[var] = cargar_modelo(var, CONF, p, y_de, std24_de, tiempos, estaciones, normalizador="combinado")
    ags["V6_std24"] = cargar_modelo("V6", CONF, p, y_de, std24_de, tiempos, estaciones)  # ablación de la normalización
    ags["PADRE"] = ags["V6"]

    # ---------- H1, H2a, ablación
    print("pruebas H1", flush=True)
    h1 = pruebas_familia("H1", "PADRE", FAMILIA, "rmse", ags, n_ratio, 1)
    print("pruebas H2a", flush=True)
    h2a = pruebas_familia("H2a", "PADRE", FAMILIA, "winkler", ags, n_ratio, 2)
    print("pruebas de la ablación", flush=True)
    ab_rmse = pruebas_familia("ablacion", "V6", ABLACION, "rmse", ags, n_ratio, 3)
    ab_norm = pruebas_familia("ablacion", "V6", ["V6_std24"], "winkler", ags, n_ratio, 4)
    ab = pd.concat([ab_rmse, ab_norm], ignore_index=True)
    for c in ["wilcoxon_p", "dm_p", "bootstrap_p", "t_semillas_p", "t_nb_p", "bootstrap_semana_p"]:
        ab[c + "_holm"] = holm(ab[c])                         # Holm sobre los 5 contrastes de la familia
    ab["p_max_holm"] = ab[["wilcoxon_p_holm", "dm_p_holm", "bootstrap_p_holm"]].max(1)
    ab["rechaza_las_tres"] = ab.p_max_holm < ALFA
    ab["componente"] = ab.rival.map({"V3": "anclas (K = 1)", "V4": "pesos uniformes", "V5": "sin antisimetría",
                                     "V1": "pérdida por ancla (λ = 0.5)", "V6_std24": "normalización solo std24"})
    ref = pruebas_familia("referencia", "PADRE", REFERENCIA, "rmse", ags, n_ratio, 5)

    # ---------- H2b: TOST frente a la TCN sobre los 40 pares (semilla, origen)
    a = ags["PADRE"].corridas.set_index(["semilla", "origen"]).rmse
    t = ags["TCN"].corridas.set_index(["semilla", "origen"]).rmse.reindex(a.index)
    d = (a - t).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t90 = stats.t.ppf(0.95, len(d) - 1)
    h2b = dict(n_pares=len(d), delta=DELTA, **tost(d), ic90=[float(d.mean() - t90 * se), float(d.mean() + t90 * se)])

    # ---------- MCS (90 %) sobre la serie horaria de pérdida cuadrática
    perd = pd.DataFrame({m: ags[m].perdida_hora for m in ["PADRE"] + FAMILIA}).dropna()
    res_mcs = mcs(perd)

    # ---------- exploratorias del mecanismo y condición 5
    c6 = ags["PADRE"].corridas
    conf_csv = pd.read_csv(f"{CONF}/V6.csv")
    exploratorias = dict(
        rho_disp_err_medio=float(ags["PADRE"].extra.rho_disp_err.mean()),
        rho_disp_err_min=float(ags["PADRE"].extra.rho_disp_err.min()),
        rho_disp_err_fuerte_medio=float(ags["PADRE"].extra.rho_disp_err_fuerte.mean()),
        cobertura_total=float(c6.cobertura.mean()),
        cobertura_por_trimestre=c6.groupby("origen").cobertura.mean().round(4).to_dict(),
        cobertura_std24_por_trimestre=ags["V6_std24"].corridas.groupby("origen").cobertura.mean().round(4).to_dict(),
        cobertura_fuerte=float(c6.cobertura_fuerte.mean()),
        sesgo_fuerte=float(np.average(c6.sesgo_fuerte, weights=c6.n_fuerte)),
        sesgo_fuerte_tcn=float(np.average(ags["TCN"].corridas.sesgo_fuerte, weights=ags["TCN"].corridas.n_fuerte)),
        brecha_train_parada=float((conf_csv.rmse_parada - conf_csv.rmse_train).mean()),
        rmse_parada_medio=float(conf_csv.rmse_parada.mean()), rmse_prueba_medio=float(conf_csv.rmse.mean()))
    cobert = pd.concat([ags[m].corridas.assign(modelo=m) for m in ["PADRE", "V6_std24"] + FAMILIA + REFERENCIA])
    cobert.groupby("modelo")[["rmse", "mae", "winkler", "cobertura", "ancho", "winkler_fuerte", "cobertura_fuerte",
                              "sesgo_fuerte"]].mean().sort_values("rmse").to_csv(f"{SALIDA}/resumen_modelos.csv")

    # ---------- regla de decisión de R3 (sección 8); la condición 4 depende de los conjuntos adicionales
    r3 = dict(
        c1_menor_rmse_que_los_12=bool((h1.diferencia < 0).all()),
        c2_tres_pruebas_y_delta=bool(h1.mejor_con_relevancia.all()),
        c3_ic_inferior_mayor_que_0=bool((h1.ic95_serie_sup < 0).all()),
        c4_conjuntos_adicionales="pendiente (modelos pesados en Colab)",
        c5_supera_persistencia=bool(ref.rechaza_las_tres.iloc[0] and ref.diferencia.iloc[0] < 0),
        c5_brecha_menor_que_delta=bool(abs(exploratorias["brecha_train_parada"]) < DELTA),
        c6_robusta=bool(h1.robusta.all()))
    falla = [k for k, v in r3.items() if v is False]
    h1_estado = "rechazada" if falla else "pendiente de la condición 4"
    h2a_ok = bool(h2a.rechaza_las_tres.all() and (h2a.diferencia < 0).all())
    resumen = dict(
        protocolo="experiments/protocolo.md (tag e3-protocolo)", replicas_bootstrap=B, bloque_horas=BLOQUE,
        n_horas=int(h1.n_horas.iloc[0]), n_estaciones=int(h1.n_estaciones.iloc[0]), razon_nadeau_bengio=n_ratio,
        H1=dict(contrastes=len(h1), rechazos_las_tres=int(h1.rechaza_las_tres.sum()),
                mejor_con_relevancia=int(h1.mejor_con_relevancia.sum()), estado=h1_estado,
                condiciones_que_fallan=falla, regla_R3=r3),
        H2a=dict(contrastes=len(h2a), rechazos_las_tres=int(h2a.rechaza_las_tres.sum()),
                 menor_winkler_que_todos=bool((h2a.diferencia < 0).all()), aceptada=h2a_ok),
        H2b=h2b, H2=dict(aceptada=bool(h2a_ok and h2b["equivalentes"])),
        ablacion=ab[["componente", "diferencia", "p_max_holm", "rechaza_las_tres"]].to_dict("records"),
        persistencia=ref[["diferencia", "p_max_holm", "rechaza_las_tres"]].to_dict("records"),
        mcs=res_mcs, exploratorias=exploratorias)
    h1.to_csv(f"{SALIDA}/H1.csv", index=False)
    h2a.to_csv(f"{SALIDA}/H2a.csv", index=False)
    ab.to_csv(f"{SALIDA}/ablacion.csv", index=False)
    ref.to_csv(f"{SALIDA}/referencia_persistencia.csv", index=False)
    json.dump(resumen, open(f"{SALIDA}/resumen.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=str)
    pd.set_option("display.width", 220)
    cols = ["rival", "media_base", "media_rival", "diferencia", "ic95_serie_inf", "ic95_serie_sup", "wilcoxon_p_holm",
            "dm_p_holm", "bootstrap_p_holm", "rechaza_las_tres", "cliff"]
    print("\nH1 (RMSE)\n", h1[cols].round(4).to_string(index=False))
    print("\nH2a (Winkler)\n", h2a[cols].round(4).to_string(index=False))
    print("\nAblación\n", ab[["componente"] + cols[1:]].round(4).to_string(index=False))
    print(json.dumps({k: resumen[k] for k in ["H2b", "H2", "mcs", "exploratorias"]}, indent=1, ensure_ascii=False,
                     default=str))
