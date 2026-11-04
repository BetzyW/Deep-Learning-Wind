"""Los tres modelos base (MLP, LSTM y CNN 1D) y la rutina de entrenamiento común."""
import random, time
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, L, d, h1=256, h2=128, dropout=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(L * d, h1), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h2, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, L, d, h=64, capas=1, dropout=0.0):
        super().__init__()
        self.r = nn.LSTM(d, h, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)
        self.o = nn.Linear(h, 1)

    def forward(self, x):
        return self.o(self.r(x)[0][:, -1]).squeeze(-1)       # estado oculto de la última hora


class CNN(nn.Module):
    def __init__(self, L, d, c1=32, c2=64, k=5, dropout=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(d, c1, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2),
                               nn.Conv1d(c1, c2, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2),
                               nn.Conv1d(c2, c2, k, padding=k // 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                               nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, 1))

    def forward(self, x):
        return self.f(x.transpose(1, 2)).squeeze(-1)          # Conv1d espera (lote, canales, tiempo)


MODELOS = {"MLP": MLP, "LSTM": LSTM, "CNN": CNN}


def fijar_semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def n_parametros(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def predecir(m, lotes, idx, bs=8192):
    m.eval()
    return np.concatenate([m(lotes(b)[0]).float().cpu().numpy() for b in np.array_split(idx, max(1, len(idx) // bs))])


def entrenar(nombre, d, lotes, semilla, hp, epocas=50, paciencia=5, bs=1024, evaluar_prueba=False):
    """Entrena con Adam y MSE; parada temprana por RMSE de validación. Devuelve métricas y predicciones."""
    fijar_semilla(semilla)
    hp = dict(hp); lr = hp.pop("lr", 1e-3)
    m = MODELOS[nombre](d["L"], len(d["cols"]), **hp).to(lotes.dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr, va = d["idx"]["entrenamiento"], d["idx"]["validacion"]
    y_va = d["y"][va + d["L"] - 1 + d["H"]]
    rng = np.random.default_rng(semilla)
    mejor, estado, sin_mejora, hist, t0 = np.inf, None, 0, [], time.time()
    for ep in range(epocas):
        m.train()
        for b in np.array_split(rng.permutation(tr), len(tr) // bs):
            x, y = lotes(b)
            opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
        rmse_va = float(np.sqrt(np.mean((predecir(m, lotes, va) - y_va) ** 2)))
        hist.append(rmse_va)
        if rmse_va < mejor - 1e-4:
            mejor, sin_mejora = rmse_va, 0
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    m.load_state_dict(estado)
    out = dict(modelo=nombre, semilla=semilla, rmse_val=mejor, epocas=len(hist), segundos=time.time() - t0,
               parametros=n_parametros(m), historia=hist, pred_val=predecir(m, lotes, va))
    if evaluar_prueba:
        out["pred_prueba"] = predecir(m, lotes, d["idx"]["prueba"])
    return out
