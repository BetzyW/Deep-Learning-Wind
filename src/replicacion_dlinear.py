"""Replicación del paper guía (sección 8 del preregistro): DLinear (Zeng et al., AAAI 2023) en ETTh1.

Condiciones originales, verificadas contra el artículo (arXiv 2205.13504v3, tabla 2 y apéndice B.2) y el código oficial
(github.com/cure-lab/LTSF-Linear: scripts/EXP-LongForecasting/Linear/etth1.sh, data_loader.py, exp_main.py, tools.py):
  * ETTh1 multivariado (7 variables, features = M), ventana L = 336, horizonte T = 96.
  * Partición 12/4/4 meses: entrenamiento filas 0–8639, validación 8640–11519, prueba 11520–14399 (validación y prueba
    empiezan L filas antes para tener ventana); estandarización ajustada solo con el entrenamiento.
  * DLinear compartido entre canales (individual = False), media móvil k = 25 con relleno replicado en los bordes.
  * Adam, lr = 0.005, lote 32, máximo 10 épocas, paciencia 3 sobre el MSE de validación, lr × 0.5 por época desde la 2.ª,
    pérdida MSE; se evalúa el mejor punto de validación. MSE y MAE en la escala estandarizada (como el código oficial).
Resultado reportado por el artículo (tabla 2, DLinear, ETTh1, T = 96): MSE 0.375, MAE 0.399.
Criterio de éxito fijado antes (preregistro, sección 8): el MSE reportado cae dentro del IC 95 % de 5 réplicas o la
diferencia es menor que 0.01. Uso: python src/replicacion_dlinear.py → resultados/e2/replicacion_dlinear.json
"""
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

RUTA = "datos/externos/ETTh1/ETTh1.csv"
L, T, K = 336, 96, 25
LR, LOTE, EPOCAS, PACIENCIA = 0.005, 32, 10, 3
SEMILLAS = [2021, 2022, 2023, 2024, 2025]      # 2021 es la semilla del código oficial
REPORTADO = dict(mse=0.375, mae=0.399)


class DLinearOficial(nn.Module):
    """y = W_t · tendencia + W_s · estacional, con capas lineales L → T compartidas entre canales."""

    def __init__(self):
        super().__init__()
        self.tend, self.est = nn.Linear(L, T), nn.Linear(L, T)

    def forward(self, x):                                         # x: (B, L, C)
        pad = (K - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], 1)
        t = nn.functional.avg_pool1d(xp.transpose(1, 2), K, 1).transpose(1, 2)
        s = x - t
        return (self.tend(t.transpose(1, 2)) + self.est(s.transpose(1, 2))).transpose(1, 2)   # (B, T, C)


def ventanas(Z, a, b):
    """Índices de inicio de ventana para el tramo de filas [a, b)."""
    return np.arange(a, b - L - T + 1)


def evaluar(m, Z, idx):
    m.eval()
    se, ae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            j = idx[i:i + 256]
            x = torch.stack([Z[k:k + L] for k in j]); y = torch.stack([Z[k + L:k + L + T] for k in j])
            p = m(x)
            se += ((p - y) ** 2).sum().item(); ae += (p - y).abs().sum().item(); n += y.numel()
    return se / n, ae / n


def una_corrida(Z, semilla):
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    m = DLinearOficial()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    tr = ventanas(Z, 0, 8640)
    va = ventanas(Z, 8640 - L, 11520)
    te = ventanas(Z, 11520 - L, 14400)
    rng = np.random.default_rng(semilla)
    mejor, estado, sin_mejora = np.inf, None, 0
    for ep in range(1, EPOCAS + 1):
        m.train()
        orden = rng.permutation(tr)
        for i in range(0, len(orden) - LOTE + 1, LOTE):             # drop_last = True en el entrenamiento
            j = orden[i:i + LOTE]
            x = torch.stack([Z[k:k + L] for k in j]); y = torch.stack([Z[k + L:k + L + T] for k in j])
            opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
        mse_va, _ = evaluar(m, Z, va)
        if mse_va < mejor:
            mejor, sin_mejora = mse_va, 0
            estado = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
        for g in opt.param_groups:                                  # tipo 1: lr = 0.005 · 0.5^(época − 1)
            g["lr"] = LR * 0.5 ** ep
    m.load_state_dict(estado)
    mse, mae = evaluar(m, Z, te)
    return dict(semilla=semilla, mse=mse, mae=mae, mse_validacion=mejor, epocas=ep)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    datos = np.loadtxt(RUTA, delimiter=",", skiprows=1, usecols=range(1, 8))
    mu, sd = datos[:8640].mean(0), datos[:8640].std(0)
    Z = torch.tensor((datos - mu) / sd, dtype=torch.float32)
    res = []
    for s in SEMILLAS:
        t0 = time.time()
        r = una_corrida(Z, s)
        print(f"semilla {s}: MSE {r['mse']:.4f}  MAE {r['mae']:.4f}  ({r['epocas']} épocas, {time.time() - t0:.0f} s)",
              flush=True)
        res.append(r)
    mse = np.array([r["mse"] for r in res]); mae = np.array([r["mae"] for r in res])
    t = stats.t.ppf(0.975, len(mse) - 1)
    ic = [float(mse.mean() - t * mse.std(ddof=1) / np.sqrt(len(mse))),
          float(mse.mean() + t * mse.std(ddof=1) / np.sqrt(len(mse)))]
    dif = float(mse.mean() - REPORTADO["mse"])
    exito = bool(ic[0] <= REPORTADO["mse"] <= ic[1] or abs(dif) < 0.01)
    resumen = dict(reportado=REPORTADO, mse_media=float(mse.mean()), mse_de=float(mse.std(ddof=1)), ic95_mse=ic,
                   mae_media=float(mae.mean()), diferencia_mse=dif, exito=exito,
                   criterio="MSE reportado dentro del IC 95 % de 5 réplicas o |diferencia| < 0.01")
    os.makedirs("resultados/e2", exist_ok=True)
    json.dump(dict(corridas=res, resumen=resumen), open("resultados/e2/replicacion_dlinear.json", "w"), indent=1)
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
