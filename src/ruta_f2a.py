"""Ruta F2a del E2 (pasos S7 y S8): residuos, calibración, incertidumbre, estabilidad, robustez e interpretabilidad.

Funciones que operan sobre las predicciones de prueba guardadas por `finales.py` (y, para robustez e
interpretabilidad, sobre los modelos entrenados .pt). Se aplican a los 3 mejores modelos del benchmark.

* Residuos: Ljung–Box (autocorrelación), ARCH-LM (heterocedasticidad condicional), Jarque–Bera (normalidad) y BDS
  (dependencia no lineal), por estación, con la proporción de estaciones que rechazan.
* Calibración de la media: regresión de Mincer–Zarnowitz y = a + b ŷ con H0: a = 0, b = 1 (prueba F).
* Incertidumbre: intervalos por predicción conforme partida (residuos absolutos del último origen de validación, sin
  tocar la prueba) a niveles 50, 80 y 90 %; cobertura empírica, ancho medio, pérdida pinball y CRPS aproximado con los
  cuantiles; PIT a partir de la distribución empírica de residuos.
* Estabilidad (sobreajuste en el tiempo): prueba de fluctuación de Giacomini y Rossi (2010) sobre el estadístico de
  Diebold–Mariano en ventanas móviles, y razón de error fuera/dentro de muestra.
* Robustez: tres niveles de severidad de ruido gaussiano, horas faltantes y quiebre de nivel en la ENTRADA de prueba.
* Interpretabilidad: gradientes integrados por hora de la ventana y por variable, con prueba de fidelidad (borrado de
  las horas más importantes frente a horas al azar) y estabilidad entre semillas (correlación de Spearman).
"""
import numpy as np
import pandas as pd
from scipy import stats

NIVELES = [0.5, 0.8, 0.9]


# ---------------------------------------------------------------- residuos y calibración


def pruebas_residuos(e, rezagos=24):
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import bds
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    lb = acorr_ljungbox(e, lags=[rezagos], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(e, nlags=rezagos)[1]
    jb_p = stats.jarque_bera(e).pvalue
    bds_p = float(np.atleast_1d(bds(e[:5000], max_dim=2)[1])[0])          # BDS es costoso: primeras 5 000 horas
    return dict(ljung_box_p=float(lb), arch_lm_p=float(arch_p), jarque_bera_p=float(jb_p), bds_p=bds_p)


def mincer_zarnowitz(y, yhat):
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(yhat))
    r = sm.OLS(np.asarray(y), X).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
    f = r.f_test("const = 0, x1 = 1")
    return dict(a=float(r.params[0]), b=float(r.params[1]), p_conjunto=float(f.pvalue))


# ---------------------------------------------------------------- incertidumbre


def cuantiles_conformes(residuos_validacion, niveles=NIVELES):
    """Cuantiles de |e| (con la corrección de muestra finita) para intervalos simétricos ŷ ± q."""
    r = np.sort(np.abs(np.asarray(residuos_validacion)))
    n = len(r)
    return {a: float(r[min(n - 1, int(np.ceil((n + 1) * a)) - 1)]) for a in niveles}


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def evaluar_intervalos(y, yhat, residuos_validacion):
    y, yhat = np.asarray(y), np.asarray(yhat)
    qs = cuantiles_conformes(residuos_validacion)
    filas = []
    for a, q in qs.items():
        lo, hi = yhat - q, yhat + q
        filas.append(dict(nivel=a, cobertura=float(np.mean((y >= lo) & (y <= hi))), ancho=float(2 * q),
                          pinball_inferior=pinball(y, lo, (1 - a) / 2), pinball_superior=pinball(y, hi, (1 + a) / 2)))
    # CRPS con la distribución predictiva ŷ + e (e: residuos de validación), fórmula de energía con una submuestra
    rv = np.asarray(residuos_validacion)
    rv = np.random.default_rng(0).choice(rv, min(500, len(rv)), replace=False)
    crps = np.mean(np.abs(yhat[:, None] + rv[None] - y[:, None]).mean(1) - 0.5 * np.abs(rv[:, None] - rv[None]).mean())
    pit = np.searchsorted(np.sort(np.asarray(residuos_validacion)), y - yhat) / len(residuos_validacion)
    ks = stats.kstest(pit, "uniform")
    return pd.DataFrame(filas), dict(crps=float(crps), pit_ks_p=float(ks.pvalue))


# ---------------------------------------------------------------- estabilidad


def giacomini_rossi(e1, e2, ventana=24 * 30, h=24):
    """Fluctuación de Giacomini y Rossi (2010): DM reescalado en ventanas móviles; se compara su máximo absoluto con
    el valor crítico tabulado para μ = ventana/n (α = 0.05). Devuelve la serie y el máximo."""
    from estadistica_e2 import hac_var
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    s = np.sqrt(hac_var(d, h - 1))
    serie = np.array([d[i:i + ventana].mean() * np.sqrt(ventana) / s for i in range(0, n - ventana + 1, 24)])
    mu = ventana / n
    # valores críticos de Giacomini y Rossi (2010, tabla 1), bilaterales al 5 %. PENDIENTE: verificar cada valor contra
    # el artículo (Review of Economic Studies 77(2)) antes de reportar; están escritos de memoria
    tabla = {0.1: 3.393, 0.2: 3.170, 0.3: 3.011, 0.4: 2.882, 0.5: 2.770, 0.6: 2.662, 0.7: 2.546, 0.8: 2.406, 0.9: 2.193}
    clave = min(tabla, key=lambda m: abs(m - mu))
    return dict(maximo=float(np.abs(serie).max()), critico=tabla[clave], mu=float(mu),
                inestable=bool(np.abs(serie).max() > tabla[clave])), serie


# ---------------------------------------------------------------- robustez


def perturbar(x, tipo, nivel, rng):
    """x: (B, L, d) estandarizada. Ruido: σ ∈ {0.1, 0.3, 0.5}; faltantes: fracción de horas puestas en 0 (la media del
    escalado) ∈ {0.1, 0.3, 0.5}; quiebre: desplazamiento de nivel de la velocidad en las últimas 72 h ∈ {0.5, 1, 2} σ."""
    import torch
    x = x.clone()
    if tipo == "ruido":
        x = x + nivel * torch.randn(x.shape, generator=rng, device=x.device)
    elif tipo == "faltantes":
        m = torch.rand(x.shape[:2], generator=rng, device=x.device) < nivel
        x[m] = 0.0
    elif tipo == "quiebre":
        x[:, -72:, 0] += nivel
    return x


SEVERIDADES = {"ruido": [0.1, 0.3, 0.5], "faltantes": [0.1, 0.3, 0.5], "quiebre": [0.5, 1.0, 2.0]}


# ---------------------------------------------------------------- interpretabilidad


def gradientes_integrados(modelo, x, pasos=32):
    """Atribución de Sundararajan et al. (2017) con línea base 0 (la media del escalado). x: (B, L, d)."""
    import torch
    x = x.detach()
    total = torch.zeros_like(x)
    for k in range(1, pasos + 1):
        xi = (k / pasos * x).requires_grad_(True)
        modelo(xi).sum().backward()
        total += xi.grad
    return (x * total / pasos).detach()


def fidelidad_borrado(modelo, x, atrib, fraccion=0.1, rng=None):
    """Borra (pone en 0) la fracción de horas con mayor |atribución| y la misma cantidad al azar; una atribución fiel
    cambia más la predicción con el primer borrado."""
    import torch
    imp = atrib.abs().sum(-1)                                        # (B, L)
    k = max(1, int(fraccion * x.shape[1]))
    top = imp.topk(k, dim=1).indices
    azar = torch.stack([torch.randperm(x.shape[1], generator=rng, device=x.device)[:k] for _ in range(x.shape[0])])
    with torch.no_grad():
        base = modelo(x)
        def borrar(ind):
            xb = x.clone()
            xb.scatter_(1, ind.unsqueeze(-1).expand(-1, -1, x.shape[2]), 0.0)
            return (modelo(xb) - base).abs().mean().item()
        return dict(cambio_top=borrar(top), cambio_azar=borrar(azar))
