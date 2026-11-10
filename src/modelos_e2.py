"""Modelos del Top 10 (E2) y entrenamiento común sobre los pliegues de origen móvil.

Arquitectura de búsqueda (sección 14): 3 bloques de anchos w = (w1, w2, w3) y patrón de dropout b = (b1, b2, b3)
con p = 0.2. La referencia del preregistro es w = (64, 64, 64), b = (0, 1, 0).
DLinear usa su arquitectura publicada (Zeng et al., 2023): descomposición por media móvil de 25 pasos y una capa
lineal para la tendencia y otra para el resto, aplicadas a la serie de velocidad.
"""
import math, os, random, time
import numpy as np
import torch
import torch.nn as nn

from origen_movil import Escalador, objetivos

P_DROPOUT = 0.2


class DLinear(nn.Module):
    """Arquitectura publicada: tendencia = media móvil (k = 25, con relleno en los bordes); dos capas lineales L → 1."""

    def __init__(self, L, d, kernel=25, **_):
        super().__init__()
        self.k = kernel
        self.tendencia, self.resto = nn.Linear(L, 1), nn.Linear(L, 1)

    def forward(self, x):
        v = x[..., 0]                                                    # canal de velocidad (independiente)
        pad = (self.k - 1) // 2
        vp = torch.cat([v[:, :1].repeat(1, pad), v, v[:, -1:].repeat(1, self.k - 1 - pad)], 1)
        t = vp.unfold(1, self.k, 1).mean(-1)
        return (self.tendencia(t) + self.resto(v - t)).squeeze(-1)


class BloqueTCN(nn.Module):
    def __init__(self, c_in, c_out, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil                                         # causal: solo mira al pasado
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dil)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dil)
        self.drop = nn.Dropout(drop)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return self.drop(torch.relu(y + self.res(x)))


class TCN(nn.Module):
    """Bai et al. (2018): bloques residuales de convoluciones causales dilatadas; aquí 3 bloques (w_i canales)."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), k=7, **_):
        super().__init__()
        canales = [d] + list(w)
        # dilataciones 1, 4, 16 → campo receptivo 1 + 2(k−1)(1+4+16) = 253 ≥ 168 con k = 7
        self.bloques = nn.Sequential(*[BloqueTCN(canales[i], canales[i + 1], k, 4 ** i, P_DROPOUT * b[i])
                                       for i in range(3)])
        self.salida = nn.Linear(w[-1], 1)

    def forward(self, x):
        return self.salida(self.bloques(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class Transformer(nn.Module):
    """Codificador de Vaswani et al. (2017) con 3 bloques de dimensión w_i (cabezas = w_i / 16) y proyecciones
    entre bloques cuando cambia el ancho; codificación posicional sinusoidal; salida desde la última hora."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), **_):
        super().__init__()
        self.entrada = nn.Linear(d, w[0])
        pe = torch.zeros(L, w[0])
        pos = torch.arange(L).unsqueeze(1)
        div = torch.exp(torch.arange(0, w[0], 2) * (-math.log(10000.0) / w[0]))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        capas, proy = [], []
        for i in range(3):
            capas.append(nn.TransformerEncoderLayer(w[i], max(1, w[i] // 16), 2 * w[i], P_DROPOUT * b[i],
                                                    batch_first=True, norm_first=True))
            proy.append(nn.Linear(w[i], w[i + 1]) if i < 2 and w[i] != w[i + 1] else nn.Identity())
        self.capas, self.proy = nn.ModuleList(capas), nn.ModuleList(proy)
        self.salida = nn.Linear(w[-1], 1)

    def forward(self, x):
        h = self.entrada(x) + self.pe
        for capa, p in zip(self.capas, self.proy):
            h = p(capa(h))
        return self.salida(h[:, -1]).squeeze(-1)


class MLP(nn.Module):
    """Perceptrón de 3 capas ocultas (w_i neuronas); dropout tras la activación de la capa i si b_i = 1."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), **_):
        super().__init__()
        capas, n = [nn.Flatten()], L * d
        for wi, bi in zip(w, b):
            capas += [nn.Linear(n, wi), nn.ReLU(), nn.Dropout(P_DROPOUT * bi)]
            n = wi
        self.f = nn.Sequential(*capas, nn.Linear(n, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class Recurrente(nn.Module):
    """3 capas recurrentes apiladas (w_i unidades); dropout sobre la secuencia de salida de la capa i si b_i = 1."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), celda=nn.LSTM, **_):
        super().__init__()
        self.capas = nn.ModuleList([celda(d if i == 0 else w[i - 1], w[i], batch_first=True) for i in range(3)])
        self.drop = nn.ModuleList([nn.Dropout(P_DROPOUT * bi) for bi in b])
        self.salida = nn.Linear(w[-1], 1)

    def forward(self, x):
        for capa, dr in zip(self.capas, self.drop):
            x = dr(capa(x)[0])
        return self.salida(x[:, -1]).squeeze(-1)


class LSTM(Recurrente):
    def __init__(self, L, d, **k):
        super().__init__(L, d, celda=nn.LSTM, **k)


class GRU(Recurrente):
    def __init__(self, L, d, **k):
        super().__init__(L, d, celda=nn.GRU, **k)


class CNN(nn.Module):
    """3 bloques convolucionales 1D (w_i filtros, k = 7, ReLU, agrupamiento máximo en los dos primeros); dropout
    tras el bloque i si b_i = 1; promedio global y salida lineal (misma estructura que la CNN del E1)."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), k=7, **_):
        super().__init__()
        capas, c = [], d
        for i, (wi, bi) in enumerate(zip(w, b)):
            capas += [nn.Conv1d(c, wi, k, padding=k // 2), nn.ReLU()]
            capas += [nn.MaxPool1d(2)] if i < 2 else [nn.AdaptiveAvgPool1d(1)]
            capas += [nn.Dropout(P_DROPOUT * bi)]
            c = wi
        self.f = nn.Sequential(*capas, nn.Flatten(), nn.Linear(c, 1))

    def forward(self, x):
        return self.f(x.transpose(1, 2)).squeeze(-1)


class CapaKAN(nn.Module):
    """Capa de Kolmogorov-Arnold (Liu et al., 2024) en la forma eficiente: φ(x) = W_b·SiLU(x) + W_s·B-splines(x),
    con splines cúbicos sobre una rejilla de 5 intervalos en [−3, 3] (entradas estandarizadas)."""

    def __init__(self, n_in, n_out, rejilla=5, orden=3, rango=3.0):
        super().__init__()
        h = 2 * rango / rejilla
        g = torch.arange(-orden, rejilla + orden + 1) * h - rango
        self.register_buffer("g", g)
        self.orden = orden
        self.base = nn.Linear(n_in, n_out)
        self.spline = nn.Parameter(torch.randn(n_out, n_in * (rejilla + orden)) * 0.1 / math.sqrt(n_in))

    def bases(self, x):
        g, x = self.g, x.unsqueeze(-1)
        B = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.orden + 1):
            B = ((x - g[:-(k + 1)]) / (g[k:-1] - g[:-(k + 1)]) * B[..., :-1]
                 + (g[k + 1:] - x) / (g[k + 1:] - g[1:-k]) * B[..., 1:])
        return B.flatten(1)

    def forward(self, x):
        return self.base(nn.functional.silu(x)) + nn.functional.linear(self.bases(x), self.spline)


class KAN(nn.Module):
    """3 capas KAN ocultas (w_i salidas) sobre la ventana aplanada; dropout tras la capa i si b_i = 1."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), **_):
        super().__init__()
        dims = [L * d] + list(w)
        self.capas = nn.ModuleList([CapaKAN(dims[i], dims[i + 1]) for i in range(3)])
        self.drop = nn.ModuleList([nn.Dropout(P_DROPOUT * bi) for bi in b])
        self.salida = nn.Linear(w[-1], 1)

    def forward(self, x):
        h = x.flatten(1)
        for capa, dr in zip(self.capas, self.drop):
            h = dr(capa(h))
        return self.salida(h).squeeze(-1)


class BloqueMamba(nn.Module):
    """Bloque de Mamba (Gu y Dao, 2023) en PyTorch puro: proyección, convolución causal, barrido selectivo de un
    espacio de estados (d_estado = 16, Δ, B y C dependientes de la entrada) y compuerta; recorrido secuencial."""

    def __init__(self, d_in, d_modelo, d_estado=16, k_conv=4):
        super().__init__()
        e = 2 * d_modelo
        self.entrada = nn.Linear(d_in, 2 * e)
        self.conv = nn.Conv1d(e, e, k_conv, groups=e, padding=k_conv - 1)
        self.x_proj = nn.Linear(e, 2 * d_estado + 1)
        self.dt = nn.Linear(1, e)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_estado + 1).float()).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.salida = nn.Linear(e, d_modelo)
        self.n = d_estado

    def forward(self, x):
        Lx = x.shape[1]
        u, z = self.entrada(x).chunk(2, -1)
        u = nn.functional.silu(self.conv(u.transpose(1, 2))[..., :Lx].transpose(1, 2))
        dbc = self.x_proj(u)
        delta = nn.functional.softplus(self.dt(dbc[..., :1]))               # (B, L, e)
        Bm, Cm = dbc[..., 1:1 + self.n], dbc[..., 1 + self.n:]              # (B, L, n)
        A = -torch.exp(self.A_log)                                          # (e, n)
        h = u.new_zeros(u.shape[0], u.shape[2], self.n)
        ys = []
        for a in range(0, Lx, 24):                                          # tramos de 24 h con recomputación
            sl = slice(a, min(Lx, a + 24))
            if self.training:
                yt, h = torch.utils.checkpoint.checkpoint(_barrido, h, delta[:, sl], A, Bm[:, sl], Cm[:, sl], u[:, sl],
                                                          use_reentrant=False)
            else:
                yt, h = _barrido(h, delta[:, sl], A, Bm[:, sl], Cm[:, sl], u[:, sl])
            ys.append(yt)
        y = torch.cat(ys, 1) + u * self.D
        return self.salida(y * nn.functional.silu(z))


def _barrido(h, delta, A, Bm, Cm, u):
    """Barrido selectivo secuencial sobre un tramo: h_t = exp(Δ_t A) h_{t−1} + Δ_t B_t u_t, y_t = C_t h_t."""
    ys = []
    for t in range(delta.shape[1]):
        dt = delta[:, t].unsqueeze(-1)
        h = torch.exp(dt * A) * h + dt * Bm[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1), h


class Mamba(nn.Module):
    """3 bloques Mamba (dimensión w_i) con normalización previa; dropout tras el bloque i si b_i = 1."""

    def __init__(self, L, d, w=(64, 64, 64), b=(0, 1, 0), **_):
        super().__init__()
        dims = [d] + list(w)
        self.normas = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(3)])
        self.bloques = nn.ModuleList([BloqueMamba(dims[i], dims[i + 1]) for i in range(3)])
        self.drop = nn.ModuleList([nn.Dropout(P_DROPOUT * bi) for bi in b])
        self.salida = nn.Linear(w[-1], 1)

    def forward(self, x):
        for nrm, blq, dr in zip(self.normas, self.bloques, self.drop):
            x = dr(blq(nrm(x)))
        return self.salida(x[:, -1]).squeeze(-1)


K_VECINOS = 8


class ConvTemporalGLU(nn.Module):
    """Convolución temporal con compuerta (GLU) de STGCN: (P + x) ⊙ σ(Q), con Kt = 3 y residual alineado."""

    def __init__(self, ci, co, kt=3):
        super().__init__()
        self.kt, self.conv = kt, nn.Conv2d(ci, 2 * co, (kt, 1))
        self.res = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):                                                     # x: (B, C, T, N)
        P, Q = self.conv(x).chunk(2, 1)
        return (P + self.res(x)[:, :, self.kt - 1:]) * torch.sigmoid(Q)


class ConvCheb(nn.Module):
    """Convolución de grafo de Chebyshev de orden Ks = 3 con el L̃ de cada muestra: Σ_k Θ_k T_k(L̃) x."""

    def __init__(self, ci, co, ks=3):
        super().__init__()
        self.ks, self.lin = ks, nn.Linear(ci * ks, co)

    def forward(self, x, Lt):                                                 # x: (B, C, T, N), Lt: (B, N, N)
        T = [x, torch.einsum("bnm,bctm->bctn", Lt, x)]
        for _ in range(2, self.ks):
            T.append(2 * torch.einsum("bnm,bctm->bctn", Lt, T[-1]) - T[-2])
        return torch.relu(self.lin(torch.cat(T, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class BloqueST(nn.Module):
    def __init__(self, ci, c=(64, 16, 64), p=0.0):
        super().__init__()
        self.t1, self.s, self.t2 = ConvTemporalGLU(ci, c[0]), ConvCheb(c[0], c[1]), ConvTemporalGLU(c[1], c[2])
        self.norma, self.drop = nn.LayerNorm(c[2]), nn.Dropout(p)

    def forward(self, x, Lt):
        x = self.t2(self.s(self.t1(x), Lt))
        return self.drop(self.norma(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class STGCN(nn.Module):
    """Arquitectura publicada (Yu et al., IJCAI 2018), escrita desde su formulación: 2 bloques ST-Conv (convolución
    temporal con compuerta, Kt = 3 → convolución de Chebyshev, Ks = 3 → convolución temporal con compuerta), canales
    (64, 16, 64), y capa de salida (convolución temporal sobre el resto de la ventana + capa lineal). Cada pronóstico
    usa el subgrafo de la estación y sus k = 8 vecinas más cercanas (adaptación declarada en la enmienda 6); la salida
    es la del nodo de la estación."""

    def __init__(self, L, d, **_):
        super().__init__()
        self.b1, self.b2 = BloqueST(d), BloqueST(64)
        Tr = L - 8
        self.t_out = ConvTemporalGLU(64, 64, kt=Tr)
        self.norma = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        X, Lt = x                                                             # X: (B, L, N, d)
        h = self.b2(self.b1(X.permute(0, 3, 1, 2), Lt), Lt)
        h = self.norma(self.t_out(h)[:, :, -1].transpose(1, 2))               # (B, N, 64)
        return self.fc(h[:, 0]).squeeze(-1)


class AtencionProbSparse(nn.Module):
    """Atención ProbSparse de Informer (Zhou et al., AAAI 2021), escrita desde su formulación: para cada consulta se
    estima la medida de dispersión M(q) = max_j(q·k_j/√d) − media_j(q·k_j/√d) con U = c·ln L_K claves muestreadas; solo
    las u = c·ln L_Q consultas más dispersas reciben atención completa. Las demás reciben la media de los valores
    (o su suma acumulada, si la atención es causal)."""

    def __init__(self, D, H, c=5, p=0.05, causal=False):
        super().__init__()
        self.H, self.dk, self.c, self.causal = H, D // H, c, causal
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.drop = nn.Dropout(p)

    def forward(self, xq, xkv):
        B, Lq, _ = xq.shape
        Lk = xkv.shape[1]
        Q = self.q(xq).view(B, Lq, self.H, self.dk).transpose(1, 2)            # (B, H, Lq, dk)
        K = self.k(xkv).view(B, Lk, self.H, self.dk).transpose(1, 2)
        V = self.v(xkv).view(B, Lk, self.H, self.dk).transpose(1, 2)
        U = min(Lk, int(self.c * math.ceil(math.log(Lk + 1))))
        u = min(Lq, int(self.c * math.ceil(math.log(Lq + 1))))
        idx = torch.randint(Lk, (Lq, U), device=xq.device)                     # claves muestreadas por consulta
        # se calculan los productos q·k completos (B, H, Lq, Lk) y se toman las U columnas muestreadas: es el mismo
        # valor que multiplicar por las claves muestreadas, pero sin materializar un tensor (B, H, Lq, U, dk), que
        # con lotes grandes ocupa decenas de GB
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)
        qk = S.gather(-1, idx.expand(B, self.H, Lq, U))
        M = qk.max(-1).values - qk.sum(-1) / Lk
        top = M.topk(u, dim=-1).indices                                        # (B, H, u)
        if self.causal:
            salida = V.cumsum(2)[:, :, :Lq] / torch.arange(1, Lq + 1, device=xq.device).view(1, 1, -1, 1)
        else:
            salida = V.mean(2, keepdim=True).expand(-1, -1, Lq, -1).clone()
        s = S.gather(2, top.unsqueeze(-1).expand(-1, -1, -1, Lk))              # (B, H, u, Lk)
        if self.causal:
            s = s.masked_fill(top.unsqueeze(-1) < torch.arange(Lk, device=xq.device), float("-inf"))
        salida = salida.scatter(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk), self.drop(s.softmax(-1)) @ V)
        return self.o(salida.transpose(1, 2).reshape(B, Lq, -1))


class CapaInformer(nn.Module):
    def __init__(self, D, H, F, p, causal=False, cruzada=False):
        super().__init__()
        self.att = AtencionProbSparse(D, H, p=p, causal=causal)
        self.cruz = nn.MultiheadAttention(D, H, dropout=p, batch_first=True) if cruzada else None
        self.ff = nn.Sequential(nn.Conv1d(D, F, 1), nn.GELU(), nn.Dropout(p), nn.Conv1d(F, D, 1))
        self.normas = nn.ModuleList([nn.LayerNorm(D) for _ in range(3 if cruzada else 2)])
        self.drop = nn.Dropout(p)

    def forward(self, x, mem=None):
        x = self.normas[0](x + self.drop(self.att(x, x)))
        if self.cruz is not None:
            x = self.normas[1](x + self.drop(self.cruz(x, mem, mem, need_weights=False)[0]))
        return self.normas[-1](x + self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2)))


class Destilacion(nn.Module):
    """Destilación entre capas del codificador: Conv1d (k = 3) + BatchNorm + ELU + MaxPool (k = 3, paso 2)."""

    def __init__(self, D):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(D, D, 3, padding=1, padding_mode="circular"), nn.BatchNorm1d(D), nn.ELU(),
                               nn.MaxPool1d(3, 2, 1))

    def forward(self, x):
        return self.f(x.transpose(1, 2)).transpose(1, 2)


class Informer(nn.Module):
    """Arquitectura publicada (Zhou et al., AAAI 2021), configuración por defecto del artículo: D = 512, H = 8 cabezas,
    feed-forward 2048, 2 capas de codificador con destilación, 1 capa de decodificador, factor c = 5, dropout 0.05.
    Decodificador generativo: recibe las últimas 48 h (label_len) y 24 posiciones en cero; la salida en la última
    posición es el pronóstico a τ + 24. Incrustación por convolución (k = 3) más posición sinusoidal; sin incrustación
    de calendario (el panel no la trae), adaptación declarada en la enmienda 5."""

    def __init__(self, L, d, D=512, H=8, F=2048, p=0.05, etiqueta=48, h=24, **_):
        super().__init__()
        self.etiqueta, self.h = etiqueta, h
        self.inc_enc = nn.Conv1d(d, D, 3, padding=1, padding_mode="circular")
        self.inc_dec = nn.Conv1d(d, D, 3, padding=1, padding_mode="circular")
        n = max(L, etiqueta + h)
        pe = torch.zeros(n, D)
        pos = torch.arange(n).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.enc = nn.ModuleList([CapaInformer(D, H, F, p) for _ in range(2)])
        self.dest = Destilacion(D)
        self.norma_enc = nn.LayerNorm(D)
        self.dec = CapaInformer(D, H, F, p, causal=True, cruzada=True)
        self.norma_dec = nn.LayerNorm(D)
        self.salida = nn.Linear(D, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, L, d = x.shape
        e = self.drop(self.inc_enc(x.transpose(1, 2)).transpose(1, 2) + self.pe[:L])
        e = self.dest(self.enc[0](e))
        mem = self.norma_enc(self.enc[1](e))
        xd = torch.cat([x[:, -self.etiqueta:], x.new_zeros(B, self.h, d)], 1)
        z = self.drop(self.inc_dec(xd.transpose(1, 2)).transpose(1, 2) + self.pe[:self.etiqueta + self.h])
        z = self.norma_dec(self.dec(z, mem))
        return self.salida(z[:, -1]).squeeze(-1)


class CapaPatchTST(nn.Module):
    """Capa del codificador de PatchTST: atención multicabeza y red feed-forward, cada una con conexión residual y
    normalización por lotes (BatchNorm) en lugar de LayerNorm, como en el artículo."""

    def __init__(self, D, H, F, p):
        super().__init__()
        self.att = nn.MultiheadAttention(D, H, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, F), nn.GELU(), nn.Dropout(p), nn.Linear(F, D))
        self.n1, self.n2 = nn.BatchNorm1d(D), nn.BatchNorm1d(D)
        self.d1, self.d2 = nn.Dropout(p), nn.Dropout(p)

    @staticmethod
    def _bn(bn, h):                                    # BatchNorm1d espera (B, D, N)
        return bn(h.transpose(1, 2)).transpose(1, 2)

    def forward(self, h):
        h = self._bn(self.n1, h + self.d1(self.att(h, h, h, need_weights=False)[0]))
        return self._bn(self.n2, h + self.d2(self.ff(h)))


class PatchTST(nn.Module):
    """Arquitectura publicada (Nie et al., ICLR 2023), versión supervisada para conjuntos grandes, escrita desde su
    formulación: independencia de canal (solo la serie de velocidad, como DLinear), normalización por instancia
    (RevIN sin parámetros afines), parches de longitud P = 16 con paso S = 8 (relleno del final con el último valor,
    N = 21 parches para L = 168), proyección lineal a D = 128 con posición aprendida, 3 capas de codificador con
    H = 16 cabezas, feed-forward F = 256 y dropout 0.2, y cabezal lineal sobre los parches aplanados."""

    def __init__(self, L, d, P=16, S=8, D=128, H=16, F=256, capas=3, p=0.2, mu_y=0.0, sd_y=1.0, **_):
        super().__init__()
        self.P, self.S = P, S
        N = (L - P) // S + 2
        self.proy = nn.Linear(P, D)
        self.pos = nn.Parameter(torch.empty(N, D).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(p)
        self.cod = nn.ModuleList([CapaPatchTST(D, H, F, p) for _ in range(capas)])
        self.cabezal = nn.Linear(N * D, 1)
        # la entrada llega estandarizada con el escalador del origen y el objetivo en m/s: la salida desnormalizada por
        # RevIN se lleva a m/s con la media y la desviación de la velocidad de ese escalador (2 parámetros afines que
        # parten de esos valores exactos y pueden ajustarse)
        self.escala, self.sesgo = nn.Parameter(torch.tensor([float(sd_y)])), nn.Parameter(torch.tensor([float(mu_y)]))

    def forward(self, x):
        v = x[..., 0]                                                       # (B, L)
        mu, sd = v.mean(1, keepdim=True), v.std(1, keepdim=True) + 1e-5     # RevIN
        v = (v - mu) / sd
        v = torch.cat([v, v[:, -1:].repeat(1, self.S)], 1)                  # relleno del final
        z = self.drop(self.proy(v.unfold(1, self.P, self.S)) + self.pos)   # (B, N, D)
        for capa in self.cod:
            z = capa(z)
        out = self.cabezal(z.flatten(1)).squeeze(-1) * sd[:, 0] + mu[:, 0]
        return out * self.escala + self.sesgo


# Chronos y Moirai: el modelo preentrenado queda congelado y su representación de cada ventana se calcula una vez
# (src/embeber.py); lo que se ajusta es un cabezal de 3 capas densas sobre esa representación, con el mismo espacio de
# arquitecturas de la sección 14 (sección 14.2 del enunciado: "solo el cabezal de 3 capas densas").
PREENTRENADOS = {"Chronos", "Moirai"}
EMB_DIR = os.environ.get("EMB_DIR", "/content/emb")

MODELOS = {"Chronos": MLP, "Moirai": MLP, "DLinear": DLinear, "TCN": TCN, "Transformer": Transformer, "MLP": MLP, "LSTM": LSTM, "GRU": GRU,
           "CNN": CNN, "KAN": KAN, "Mamba": Mamba, "PatchTST": PatchTST, "Informer": Informer, "STGCN": STGCN}


def fijar_semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Datos:
    """Panel escalado con el entrenamiento de un pliegue, en la GPU; arma lotes (B, L, d) por índices τ."""

    def __init__(self, panel, pliegue, dispositivo="cuda"):
        esc = Escalador().ajustar(panel, pliegue["entrenamiento"])
        self.X = torch.tensor(esc.transformar(panel.X), device=dispositivo)
        self.mu_y, self.sd_y = float(esc.mu[0]), float(esc.sd[0])     # escala de la velocidad (para PatchTST)
        self.y = torch.tensor(np.nan_to_num(panel.X[:, 0]), device=dispositivo)
        self.off = torch.arange(-panel.L + 1, 1, device=dispositivo)
        self.h, self.dev, self.L, self.d = panel.horizontes[0], dispositivo, panel.L, panel.X.shape[1]

        self.grafo, self._grafo = None, None
        self._emb, self._entrenamiento = {}, pliegue["entrenamiento"]

    def con_embeddings(self, nombre):
        """Vista de estos datos con la representación congelada del modelo preentrenado (L = 1, d = dimensión),
        estandarizada con las ventanas de entrenamiento de este origen."""
        if nombre not in self._emb:
            E = np.load(f"{EMB_DIR}/{nombre}.npy", mmap_mode="r")
            tr = np.asarray(self._entrenamiento)
            mu, sd = E[tr].astype(np.float32).mean(0), E[tr].astype(np.float32).std(0) + 1e-6
            vista = Datos.__new__(Datos)
            vista.X = torch.tensor((np.asarray(E, dtype=np.float32) - mu) / sd, device=self.dev)
            vista.y, vista.h, vista.dev, vista.L, vista.d = self.y, self.h, self.dev, 1, E.shape[1]
            vista.off = torch.zeros(1, dtype=torch.long, device=self.dev)
            vista.grafo, vista._grafo, vista._emb = None, None, {}
            self._emb[nombre] = vista
        return self._emb[nombre]

    def activar_grafo(self, panel, ruta_panel, k=K_VECINOS):
        """Prepara la entrada de STGCN: para cada fila, las filas de sus k estaciones más cercanas en la misma hora
        (−1 si esa estación no tiene la ventana completa) y, por estación, la matriz L̃ de su subgrafo (ella + k
        vecinas). Pesos w_ij = exp(−d_ij² / σ²) con d en km (haversine) y σ = mediana de las distancias a las k
        vecinas; L̃ = −D^(−1/2) W D^(−1/2) (laplaciano normalizado reescalado con λ_max ≈ 2, como en el artículo)."""
        import pandas as pd
        coords = (pd.read_csv(ruta_panel, usecols=["CodigoEstacion", "lat", "lon"], dtype={"CodigoEstacion": str})
                  .drop_duplicates("CodigoEstacion").set_index("CodigoEstacion"))
        est = list(dict.fromkeys(panel.estacion.tolist()))                     # orden de aparición en el panel
        lat, lon = np.radians(coords.loc[est, "lat"].to_numpy()), np.radians(coords.loc[est, "lon"].to_numpy())
        dl, dn = lat[:, None] - lat[None], lon[:, None] - lon[None]
        a = np.sin(dl / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dn / 2) ** 2
        D = 2 * 6371 * np.arcsin(np.sqrt(a))
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, 1)[:, :k]                                          # (S, k)
        sigma = np.median(np.take_along_axis(D, vec, 1))
        nodos = np.concatenate([np.arange(len(est))[:, None], vec], 1)           # (S, k+1), el nodo 0 es la estación
        Dn = D.copy(); np.fill_diagonal(Dn, 0)
        sub = Dn[nodos[:, :, None], nodos[:, None, :]]                          # (S, k+1, k+1)
        W = np.exp(-(sub / sigma) ** 2) * (1 - np.eye(k + 1))
        g = 1 / np.sqrt(W.sum(-1))
        Lt = -(g[:, :, None] * W * g[:, None, :])
        # fila de cada estación en la hora t: inicio + (t − t0); válida si la estación tiene la ventana completa
        pos = {e: i for i, e in enumerate(est)}
        inicio = np.array([np.flatnonzero(panel.estacion == e)[0] for e in est])
        fin = np.array([np.flatnonzero(panel.estacion == e)[-1] + 1 for e in est])
        t0 = panel.tiempo[inicio]
        s_fila = np.array([pos[e] for e in panel.estacion])
        filas_vec = np.full((len(panel.estacion), k), -1, dtype=np.int64)
        for j in range(k):
            sj = vec[s_fila, j]
            r = inicio[sj] + ((panel.tiempo - t0[sj]) // np.timedelta64(1, "h")).astype(np.int64)
            ok = (r - (panel.L - 1) >= inicio[sj]) & (r < fin[sj])
            filas_vec[ok, j] = r[ok]
        self._grafo = dict(vec=torch.tensor(filas_vec, device=self.dev),
                          Lt=torch.tensor(Lt, dtype=torch.float32, device=self.dev),
                          est=torch.tensor(s_fila, device=self.dev))

    def __call__(self, idx):
        idx = torch.as_tensor(idx, device=self.dev)
        x = self.X[idx[:, None] + self.off]
        if self.grafo is not None:                                            # (B, L, k+1, d) y L̃ de cada subgrafo
            fv = self.grafo["vec"][idx]                                        # (B, k)
            xv = self.X[fv.clamp(min=0)[:, :, None] + self.off] * (fv >= 0)[:, :, None, None]
            x = (torch.cat([x[:, None], xv], 1).transpose(1, 2), self.grafo["Lt"][self.grafo["est"][idx]])
        return x, self.y[idx + self.h]


@torch.no_grad()
def predecir(m, datos, idx, bs=4096):
    m.eval()
    return np.concatenate([m(datos(b)[0]).float().cpu().numpy() for b in np.array_split(idx, max(1, len(idx) // bs))])


def entrenar(nombre, datos, pliegue, semilla, hp=None, lr=1e-3, epocas=50, paciencia=5, bs=1024, wd=0.0,
             frac=1.0, semilla_muestra=7):
    """Adam + MSE; ajusta con `ajuste`, parada temprana con `parada`, evalúa en `evaluacion` del pliegue.
    Con frac < 1 se usa una submuestra fija de las ventanas de ajuste (la misma para todas las arquitecturas)."""
    if nombre in PREENTRENADOS:                 # el cabezal ve la representación congelada, no la serie
        datos = datos.con_embeddings(nombre)
    if nombre == "STGCN":                       # solo STGCN recibe las ventanas de las estaciones vecinas
        assert datos._grafo is not None, "llamar datos.activar_grafo(panel, ruta_panel) antes de entrenar STGCN"
        datos.grafo = datos._grafo
        try:
            return _entrenar(nombre, datos, pliegue, semilla, hp, lr, epocas, paciencia, bs, wd, frac, semilla_muestra)
        finally:
            datos.grafo = None
    return _entrenar(nombre, datos, pliegue, semilla, hp, lr, epocas, paciencia, bs, wd, frac, semilla_muestra)


def _entrenar(nombre, datos, pliegue, semilla, hp, lr, epocas, paciencia, bs, wd, frac, semilla_muestra):
    fijar_semilla(semilla)
    extra = dict(mu_y=datos.mu_y, sd_y=datos.sd_y) if nombre == "PatchTST" else {}
    m = MODELOS[nombre](datos.L, datos.d, **(hp or {}), **extra).to(datos.dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    aj, pa, ev = pliegue["ajuste"], pliegue["parada"], pliegue["evaluacion"]
    if frac < 1:
        aj = np.sort(np.random.default_rng(semilla_muestra).choice(aj, int(frac * len(aj)), replace=False))
    y_pa = datos.y[torch.as_tensor(pa + datos.h, device=datos.dev)].cpu().numpy()
    rng = np.random.default_rng(semilla)
    mejor, estado, sin_mejora, hist, hist_tr, t0 = np.inf, None, 0, [], [], time.time()
    for ep in range(epocas):
        m.train()
        sse, n = 0.0, 0
        for b in np.array_split(rng.permutation(aj), len(aj) // bs):
            x, y = datos(b)
            opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
            sse += loss.item() * len(b); n += len(b)
        rmse_pa = float(np.sqrt(np.mean((predecir(m, datos, pa) - y_pa) ** 2)))
        hist_tr.append(float(np.sqrt(sse / n))); hist.append(rmse_pa)
        if not np.isfinite(rmse_pa):
            break
        if rmse_pa < mejor - 1e-4:
            mejor, sin_mejora = rmse_pa, 0
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if estado is not None:
        m.load_state_dict(estado)
    pred = predecir(m, datos, ev)
    return dict(modelo=nombre, semilla=semilla, origen=pliegue["origen"], rmse_parada=mejor, epocas=len(hist),
                segundos=time.time() - t0, parametros=sum(p.numel() for p in m.parameters()),
                historia=hist, historia_train=hist_tr, pred=pred, estado=m.state_dict())
