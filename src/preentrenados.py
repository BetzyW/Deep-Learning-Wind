"""Chronos y Moirai (modelos preentrenados del Top 10) con sus pesos oficiales, sin reentrenar (cero disparos).

Se corre en Colab en una sesión aparte, porque sus bibliotecas exigen otras versiones de torch, numpy y pandas:
  Chronos: pip install chronos-forecasting      → python src/preentrenados.py Chronos  --medir 2000
  Moirai : pip install uni2ts                   → python src/preentrenados.py Moirai   --medir 2000
Con --medir N se pronostican N ventanas al azar (semilla 7) del último origen de validación y se reporta el tiempo por
ventana, para estimar el costo antes de la corrida completa (sin --medir). Ventana de contexto L = 168 h de la serie de
velocidad (la misma de todos los modelos; ambos son univariados), horizonte 24 h; el pronóstico puntual es la mediana
de las trayectorias muestreadas. Las predicciones se guardan por ventana para las pruebas estadísticas.
Salida: resultados/e2/preentrenados/{modelo}_{etapa}_{origen}.npz y un resumen .json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402

H = 24
PESOS = {"Chronos": "amazon/chronos-t5-small", "Moirai": "Salesforce/moirai-1.0-R-small"}


def contextos(p, idx):
    """(n, L) con la velocidad de las L horas que terminan en τ (NaN donde falta)."""
    v = p.X[:, 0].astype(np.float32)
    return v[idx[:, None] + np.arange(-p.L + 1, 1)]


def cargar(nombre, dispositivo):
    if nombre == "Chronos":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(PESOS[nombre], device_map=dispositivo, torch_dtype=torch.bfloat16)

        def predecir(ctx):
            q, _ = pipe.predict_quantiles(torch.tensor(ctx), prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
            return q[:, H - 1, :].float().cpu().numpy()                 # (n, 3): cuantiles 0.1, 0.5, 0.9 a τ + 24
        return predecir
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    L = None

    def predecir(ctx):
        nonlocal L
        if L is None or L.context_length != ctx.shape[1]:
            L = MoiraiForecast(module=MoiraiModule.from_pretrained(PESOS[nombre]), prediction_length=H,
                               context_length=ctx.shape[1], patch_size="auto", num_samples=100, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(dispositivo).eval()
        x = torch.tensor(np.nan_to_num(ctx), device=dispositivo).unsqueeze(-1)
        obs = torch.tensor(~np.isnan(ctx), device=dispositivo).unsqueeze(-1)
        pad = torch.zeros(ctx.shape, dtype=torch.bool, device=dispositivo)
        with torch.no_grad():
            s = L(past_target=x, past_observed_target=obs, past_is_pad=pad)   # (n, muestras, H)
        s = s[..., H - 1].float().cpu().numpy()
        return np.quantile(s, [0.1, 0.5, 0.9], axis=1).T
    return predecir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", choices=list(PESOS))
    ap.add_argument("--medir", type=int, default=0)
    ap.add_argument("--etapa", choices=["validacion", "prueba"], default="validacion")
    ap.add_argument("--lote", type=int, default=256)
    a = ap.parse_args()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = leer_config()
    p = preparar_panel(cfg)
    folds = pliegues(p, cfg, a.etapa)
    if a.medir:
        folds = folds[-1:]
    predecir = cargar(a.modelo, dispositivo)
    os.makedirs("resultados/e2/preentrenados", exist_ok=True)
    resumen = []
    for f in folds:
        ev = f["evaluacion"]
        if a.medir:
            ev = np.sort(np.random.default_rng(7).choice(ev, min(a.medir, len(ev)), replace=False))
        y = objetivos(p, ev)[:, 0]
        ctx = contextos(p, ev)
        t0, q = time.time(), []
        for i in range(0, len(ev), a.lote):
            q.append(predecir(ctx[i:i + a.lote]))
        q = np.concatenate(q)
        seg = time.time() - t0
        rmse = float(np.sqrt(np.mean((q[:, 1] - y) ** 2)))
        cob = float(np.mean((y >= q[:, 0]) & (y <= q[:, 2])))
        sufijo = "medicion" if a.medir else a.etapa
        np.savez_compressed(f"resultados/e2/preentrenados/{a.modelo}_{sufijo}_{f['origen']}.npz", tau=ev, y=y, q=q)
        fila = dict(modelo=a.modelo, origen=f["origen"], ventanas=len(ev), rmse=rmse, cobertura_80=cob,
                    segundos=round(seg, 1), seg_por_1000=round(1000 * seg / len(ev), 2),
                    gpu=torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu")
        print(json.dumps(fila, ensure_ascii=False), flush=True)
        resumen.append(fila)
    json.dump(resumen, open(f"resultados/e2/preentrenados/{a.modelo}_{sufijo}.json", "w"), indent=1)
