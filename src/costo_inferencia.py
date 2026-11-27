"""Costo de cada modelo en su configuración final (secciones 11.3, 11.4 y plantilla A.2 del enunciado), medido igual
para los 13 modelos del benchmark y en el mismo equipo:
  * parámetros entrenables,
  * FLOPs por ventana con el contador de PyTorch (`torch.utils.flop_counter`), sobre una ventana de 168 h,
  * latencia de inferencia por ventana: mediana de 50 repeticiones con lote 1 y con lote 1 024 (por ventana),
  * memoria pico de la GPU en el lote de 1 024 (solo con --gpu).
En Chronos y Moirai se mide el cabezal; el modelo base congelado se mide aparte en Colab (preentrenados.py). La
LSTM no la ve el contador, así que en ella los FLOPs se calculan con la fórmula. Los FLOPs se cuentan en modo de
entrenamiento, porque en evaluación la atención de PyTorch usa un núcleo fusionado que el contador no registra.
Los pesos son aleatorios: el costo no depende de los valores, solo de la arquitectura.
Salida: resultados/e2/costo_{cpu|gpu}.csv
E3 (--e3): los mismos 13 más el modelo original (PADRE-TCN, configuración V6 de resultados/e3/PADRE_configuracion.json),
medidos juntos en la misma sesión para que las latencias sean comparables; salida en resultados/e3/costo_{cpu|gpu}.csv
(el archivo del E2 no se toca).
Uso: python src/costo_inferencia.py [--gpu] [--e3]      (dl_env)
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from modelos_e2 import K_VECINOS, MODELOS  # noqa: E402
from fase2 import arquitectura  # noqa: E402

L, D = 168, 4                           # ventana y variables por hora del panel (velocidad, dirección sen/cos, temperatura)
EMB = {"Chronos": 512, "Moirai": 384}   # dimensión de la representación congelada
TOP13 = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST",
         "MLP", "CNN", "LSTM"]


def entrada(nombre, B, dev):
    if nombre in EMB:
        return torch.randn(B, 1, EMB[nombre], device=dev)
    if nombre == "STGCN":
        return (torch.randn(B, L, K_VECINOS + 1, D, device=dev), torch.randn(B, K_VECINOS + 1, K_VECINOS + 1, device=dev))
    return torch.randn(B, L, D, device=dev)


def construir(nombre, dev):
    if nombre == "PADRE":
        from original_model import PadreTCN
        arq = json.load(open("resultados/e3/PADRE_configuracion.json"))["arquitectura"]
        arq = {**arq, "w": tuple(arq["w"]), "b": tuple(arq["b"])}
        return PadreTCN(L, D, **arq).to(dev).eval()
    c = json.load(open(f"resultados/e2/finales/{nombre}_configuracion.json"))
    hp = arquitectura(c["arquitectura"]["id"]) if c["arquitectura"] else {}
    extra = dict(mu_y=0.0, sd_y=1.0) if nombre == "PatchTST" else {}
    l, d = (1, EMB[nombre]) if nombre in EMB else (L, D)
    return MODELOS[nombre](l, d, **hp, **extra).to(dev).eval()


def flops_recurrente(m):
    """FLOPs de una pasada de LSTM/GRU (4 o 3 compuertas): 2·g·(d_in·h + h·h) por paso y capa, más la capa de salida."""
    total = 0
    for mod in m.modules():
        if isinstance(mod, (torch.nn.LSTM, torch.nn.GRU)):
            g = 4 if isinstance(mod, torch.nn.LSTM) else 3
            for k in range(mod.num_layers):
                d_in = mod.input_size if k == 0 else mod.hidden_size
                total += 2 * g * (d_in * mod.hidden_size + mod.hidden_size ** 2) * L
    return total


def medir(nombre, dev):
    from torch.utils.flop_counter import FlopCounterMode
    m = construir(nombre, dev)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    with torch.no_grad():
        x1 = entrada(nombre, 1, dev)
        m.train()                                   # en eval, la atención usa un núcleo fusionado que el contador no ve
        with FlopCounterMode(display=False) as fc:
            m(x1)
        m.eval()
        flops = fc.get_total_flops()
        if nombre == "LSTM":                        # la GRU sí la cuenta; la LSTM no
            flops += flops_recurrente(m)
        r = dict(modelo=nombre, parametros=params, mflops_ventana=flops / 1e6)
        for B in (1, 1024):
            x = entrada(nombre, B, dev)
            for _ in range(5):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = []
            for _ in range(50 if B == 1 else 10):
                t0 = time.perf_counter(); m(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t.append(time.perf_counter() - t0)
            r[f"ms_ventana_lote{B}"] = float(np.median(t)) * 1e3 / B
            if dev == "cuda" and B == 1024:
                r["memoria_pico_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return r


if __name__ == "__main__":
    import pandas as pd
    dev = "cuda" if "--gpu" in sys.argv else "cpu"
    torch.manual_seed(0)
    if dev == "cpu":
        torch.set_num_threads(4)
    filas = []
    e3 = "--e3" in sys.argv
    for nombre in TOP13 + (["PADRE"] if e3 else []):
        r = medir(nombre, dev)
        r["equipo"] = torch.cuda.get_device_name() if dev == "cuda" else f"CPU, {torch.get_num_threads()} hilos"
        print(r, flush=True)
        filas.append(r)
    pd.DataFrame(filas).to_csv(f"resultados/{'e3' if e3 else 'e2'}/costo_{dev}.csv", index=False)
