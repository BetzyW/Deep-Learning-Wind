"""Representación congelada de cada ventana con Chronos o Moirai (pesos oficiales), para ajustar después el cabezal.

Para cada fila τ del panel que cierra una ventana válida se toma el contexto de 168 h de la velocidad (ambos modelos
son univariados) y se guarda el promedio temporal de la salida del codificador del modelo preentrenado. Los pesos no
se modifican: el ajuste fino (sección 14.2) entrena solo el cabezal de 3 capas densas (`MLP` sobre esta
representación) con el mismo protocolo de los demás modelos. La representación de τ solo usa datos hasta τ.
Se corre en Colab, en una sesión aparte (las bibliotecas exigen otras versiones de torch):
  Chronos: pip install chronos-forecasting  → python src/embeber.py Chronos
  Moirai : pip install uni2ts               → python src/embeber.py Moirai
Salida: $EMB_DIR/{modelo}.npy, float16, (filas del panel, dimensión); filas sin ventana válida quedan en cero.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, preparar_panel  # noqa: E402

EMB_DIR = os.environ.get("EMB_DIR", "/content/emb")
PESOS = {"Chronos": "amazon/chronos-t5-small", "Moirai": "Salesforce/moirai-1.0-R-small"}


def codificador(nombre, dev):
    if nombre == "Chronos":
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained(PESOS[nombre], device_map=dev, torch_dtype=torch.bfloat16)

        def f(ctx):
            emb, _ = pipe.embed(torch.tensor(ctx))                           # (B, T+1, 512), incluye el token EOS
            return emb[:, :-1].float().mean(1)
        return f
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    modulo = MoiraiModule.from_pretrained(PESOS[nombre])
    capturado = {}
    modulo.encoder.register_forward_hook(lambda m, i, o: capturado.__setitem__("h", o))
    modelo = None

    def f(ctx):
        nonlocal modelo
        if modelo is None:
            modelo = MoiraiForecast(module=modulo, prediction_length=24, context_length=ctx.shape[1], patch_size=32,
                                    num_samples=1, target_dim=1, feat_dynamic_real_dim=0,
                                    past_feat_dynamic_real_dim=0).to(dev).eval()
        x = torch.tensor(np.nan_to_num(ctx), device=dev).unsqueeze(-1)
        obs = torch.tensor(~np.isnan(ctx), device=dev).unsqueeze(-1)
        with torch.no_grad():
            modelo(past_target=x, past_observed_target=obs, past_is_pad=torch.zeros(ctx.shape, dtype=torch.bool,
                                                                                     device=dev))
        return capturado["h"].float().mean(1)                                 # promedio sobre los parches
    return f


if __name__ == "__main__":
    nombre = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = leer_config()
    p = preparar_panel(cfg)
    v = p.X[:, 0].astype(np.float32)
    f = codificador(nombre, dev)
    os.makedirs(EMB_DIR, exist_ok=True)
    salida = None
    t0 = time.time()
    lote = 1024
    with torch.no_grad():
        for i in range(0, len(p.tau), lote):
            tau = p.tau[i:i + lote]
            ctx = v[tau[:, None] + np.arange(-p.L + 1, 1)]
            e = f(ctx).cpu().numpy().astype(np.float16)
            if salida is None:
                salida = np.lib.format.open_memmap(f"{EMB_DIR}/{nombre}.npy", mode="w+", dtype=np.float16,
                                                   shape=(len(v), e.shape[1]))
            salida[tau] = e
            if i % (200 * lote) == 0:
                print(f"{nombre}: {i:,}/{len(p.tau):,} ventanas, {time.time() - t0:.0f} s", flush=True)
    salida.flush()
    print(f"{nombre}: listo, {len(p.tau):,} ventanas en {time.time() - t0:.0f} s → {EMB_DIR}/{nombre}.npy", flush=True)
