"""Exportación del modelo original (PADRE-TCN, diseño congelado V6) a ONNX, con su ficha de metadatos y una prueba de
ida y vuelta.

Qué se exporta: el modelo de la semilla 1000 en el último origen de prueba (2025-10-01), que `confirmatorias_padre.py`
guarda en resultados/e3/confirmatorias/V6/2025-10-01_1000.pt. El grafo ONNX recibe la ventana ya escalada
x (B, 168, 4) y devuelve dos salidas: el pronóstico a 24 h en m/s y la dispersión ponderada entre anclas (la base de
los intervalos conformales). El lote es dinámico.

Ficha (metadata.json): columnas y orden de la entrada, ventana, horizonte, media y desviación del escalador (ajustado
con el entrenamiento de ese origen, como en el entrenamiento), arquitectura e hiperparámetros, versiones, opset y
SHA-256 del .pt y del .onnx. Con la ficha, cualquiera puede escalar sus datos y usar el .onnx sin el repositorio.

Prueba de ida y vuelta, sobre todas las ventanas reales de prueba de ese origen:
  1. onnxruntime (CPU) frente a PyTorch (CPU, mismos pesos): diferencia máxima < 1e-4 m/s en ambas salidas;
  2. onnxruntime frente al pronóstico guardado en el entrenamiento (2025-10-01_1000.npz, calculado en GPU con TF32):
     se reporta la diferencia (no se exige 1e-4, porque TF32 redondea distinto).
Salida: artifacts/{padre_v6.onnx, metadata.json, ida_y_vuelta.json} (carpeta que pide el enunciado)
Uso: python src/exportar_onnx.py      (dl_env; no necesita GPU)
"""
import hashlib, json, os, platform, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import Escalador, entradas, leer_config, pliegues, preparar_panel  # noqa: E402
from original_model import PadreTCN  # noqa: E402

PT = "resultados/e3/confirmatorias/V6/2025-10-01_1000.pt"
NPZ = "resultados/e3/confirmatorias/V6/2025-10-01_1000.npz"
SALIDA = "artifacts"
OPSET = 17
TOL = 1e-4


class PadreExportable(torch.nn.Module):
    """Envuelve PadreTCN para que una sola pasada devuelva el pronóstico y la dispersión entre anclas."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y, y_j, w = self.m.componentes(x)
        return y, (w * (y_j - y.unsqueeze(-1)) ** 2).sum(-1).sqrt()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


if __name__ == "__main__":
    import onnx, onnxruntime as ort
    os.makedirs(SALIDA, exist_ok=True)
    ck = torch.load(PT, map_location="cpu", weights_only=False)
    arq = ck["arquitectura"]
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = next(f for f in pliegues(p, cfg, "prueba") if f["origen"] == ck["origen"])
    esc = Escalador().ajustar(p, f["entrenamiento"])
    m = PadreTCN(p.L, p.X.shape[1], **arq, mu_y=float(esc.mu[0]), sd_y=float(esc.sd[0]))
    m.load_state_dict(ck["estado"])
    m.eval()
    modelo = PadreExportable(m).eval()

    # ventanas reales de prueba del origen, escaladas igual que en el entrenamiento
    ev = np.asarray(f["evaluacion"])
    X = entradas(esc.transformar(p.X), ev, p.L)
    guardado = np.load(NPZ)
    assert np.array_equal(guardado["tau"], ev.astype(np.int32)), "las ventanas no coinciden con las del entrenamiento"

    ruta_onnx = f"{SALIDA}/padre_v6.onnx"
    ejemplo = torch.tensor(X[:2])
    torch.onnx.export(modelo, (ejemplo,), ruta_onnx, input_names=["x"], output_names=["pronostico", "dispersion"],
                      dynamic_axes={"x": {0: "lote"}, "pronostico": {0: "lote"}, "dispersion": {0: "lote"}},
                      opset_version=OPSET, dynamo=False)
    onnx.checker.check_model(onnx.load(ruta_onnx))

    with torch.no_grad():
        y_pt, s_pt = (t.numpy() for t in modelo(torch.tensor(X)))
    sesion = ort.InferenceSession(ruta_onnx, providers=["CPUExecutionProvider"])
    y_ox, s_ox = sesion.run(None, {"x": X})
    prueba = dict(
        ventanas=int(len(ev)),
        origen=ck["origen"],
        max_dif_pronostico_onnx_vs_pytorch=float(np.abs(y_ox - y_pt).max()),
        max_dif_dispersion_onnx_vs_pytorch=float(np.abs(s_ox - s_pt).max()),
        max_dif_pronostico_onnx_vs_entrenamiento=float(np.abs(y_ox - guardado["pred"]).max()),
        max_dif_dispersion_onnx_vs_entrenamiento=float(np.abs(s_ox - guardado["disp"]).max()),
        rmse_onnx=float(np.sqrt(np.nanmean((y_ox - p.X[ev + p.horizontes[0], 0]) ** 2))),
        tolerancia=TOL,
        fecha=time.strftime("%Y-%m-%d %H:%M"),
    )
    prueba["aprobada"] = bool(prueba["max_dif_pronostico_onnx_vs_pytorch"] < TOL
                              and prueba["max_dif_dispersion_onnx_vs_pytorch"] < TOL)
    json.dump(prueba, open(f"{SALIDA}/ida_y_vuelta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    meta = dict(
        modelo="PADRE-TCN, diseño congelado V6 (experiments/design_log.md)",
        tarea="velocidad del viento a 24 h por estación, panel IDEAM del Caribe colombiano",
        entrada=dict(nombre="x", forma=["lote", p.L, p.X.shape[1]], tipo="float32", columnas=cfg["columnas"],
                     orden_temporal="fila 0 = hora τ − 167, fila 167 = hora de emisión τ",
                     escalado="(valor − media) / desviación por columna; faltantes → 0 después de escalar",
                     media=[float(v) for v in esc.mu], desviacion=[float(v) for v in esc.sd]),
        salidas=dict(pronostico="velocidad en τ + 24 h, m/s, forma (lote,)",
                     dispersion="desviación ponderada entre anclas, m/s, forma (lote,); base de los intervalos "
                                "conformales (s = √(dispersión² + std24²), ver src/estadistica_e3.py)"),
        horizonte_horas=int(p.horizontes[0]), ventana_horas=int(p.L),
        entrenamiento=dict(origen=ck["origen"], semilla=int(ck["semilla"]), hp=ck["hp"], arquitectura=arq,
                           escalador_ajustado_con="filas de entrada del entrenamiento de ese origen"),
        versiones=dict(python=platform.python_version(), torch=torch.__version__, onnx=onnx.__version__,
                       onnxruntime=ort.__version__, opset=OPSET),
        sha256={os.path.basename(PT): sha256(PT), "padre_v6.onnx": sha256(ruta_onnx)},
        ida_y_vuelta="ida_y_vuelta.json",
        licencia_datos="datos del IDEAM (ver e2_02_datos.md)",
    )
    json.dump(meta, open(f"{SALIDA}/metadata.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(prueba, indent=1, ensure_ascii=False))
    if not prueba["aprobada"]:
        sys.exit("La prueba de ida y vuelta no pasó")
