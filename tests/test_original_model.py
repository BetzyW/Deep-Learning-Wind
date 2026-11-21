"""Pruebas unitarias del modelo original (sección 6.5 del enunciado del E3). Se corren con `pytest tests/`.

(i) sobreajusta un lote diminuto sintético, (ii) reproducibilidad con la misma semilla, (iii) formas de salida y
entradas inválidas, (iv) todos los parámetros reciben gradiente finito, (v) verificación numérica de la propiedad
demostrada (antisimetría exacta y pronóstico dentro del rango de las anclas) y gradcheck, (vi) guardar, cargar y
exportar a ONNX conserva las salidas.
"""
import io, os, sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from original_model import DiferenciaAntisimetrica, PadreTCN  # noqa: E402

L, D = 168, 4


def lote(n=32, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randn(n, L, D, generator=g)
    y = 2.0 + x[:, -1, 0] + 0.5 * x[:, -25, 0]                            # depende del nivel actual y de ayer
    return x, y


def modelo(semilla=0, **kw):
    torch.manual_seed(semilla)
    return PadreTCN(L, D, w=(16, 16, 8), b=(0, 0, 0), K=6, **kw)


def test_sobreajusta_lote_pequeno():
    x, y = lote()
    m = modelo()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        perdida = ((m(x) - y) ** 2).mean() + m.perdida_aux(y)
        perdida.backward()
        opt.step()
    assert ((m(x) - y) ** 2).mean().item() < 0.05


def test_reproducible_con_misma_semilla():
    x, _ = lote()
    m1, m2 = modelo(1).eval(), modelo(1).eval()
    torch.testing.assert_close(m1(x), m2(x))


def test_formas_y_entradas_invalidas():
    x, _ = lote(5)
    m = modelo().eval()
    y, y_j, w = m.componentes(x)
    assert y.shape == (5,) and y_j.shape == (5, 6) and w.shape == (5, 6)
    torch.testing.assert_close(w.sum(-1), torch.ones(5))
    with pytest.raises(ValueError):
        m(torch.randn(5, L - 1, D))                                       # ventana de otra longitud
    with pytest.raises(ValueError):
        PadreTCN(L, D, K=7)                                               # más anclas de las que caben en 168 h
    with pytest.raises(RuntimeError):
        m(torch.randn(5, L, D + 1))                                       # número de variables distinto


def test_todos_los_parametros_reciben_gradiente():
    x, y = lote()
    m = modelo()
    (((m(x) - y) ** 2).mean() + m.perdida_aux(y)).backward()
    for nombre, p in m.named_parameters():
        assert p.grad is not None, nombre
        assert torch.isfinite(p.grad).all(), nombre
        assert p.grad.abs().sum() > 0, nombre


def test_antisimetria_exacta():
    """Proposición 1: D(a, b) = −D(b, a) y D(a, a) = 0 para cualquier valor de los parámetros."""
    torch.manual_seed(3)
    Dm = DiferenciaAntisimetrica(8).double()
    a, b = torch.randn(1000, 8, dtype=torch.float64), torch.randn(1000, 8, dtype=torch.float64)
    assert (Dm(a, b) + Dm(b, a)).abs().max().item() < 1e-12
    assert Dm(a, a).abs().max().item() < 1e-12


def test_pronostico_entre_anclas():
    """Proposición 2: ŷ es combinación convexa de los ŷ_j, así que min_j ŷ_j ≤ ŷ ≤ max_j ŷ_j."""
    x, _ = lote(200)
    m = modelo().eval()
    y, y_j, _ = m.componentes(x)
    assert (y >= y_j.min(-1).values - 1e-5).all() and (y <= y_j.max(-1).values + 1e-5).all()


def test_gradcheck_de_la_diferencia():
    torch.manual_seed(4)
    Dm = DiferenciaAntisimetrica(4, oculta=6).double()
    a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u, v: Dm(u, v), (a, b))


def test_guardar_cargar_y_exportar(tmp_path):
    x, _ = lote(8)
    m1, m2 = modelo(2).eval(), modelo(99).eval()
    buf = io.BytesIO(); torch.save(m1.state_dict(), buf); buf.seek(0)
    m2.load_state_dict(torch.load(buf))
    torch.testing.assert_close(m1(x), m2(x))
    ort = pytest.importorskip("onnxruntime")
    ruta = str(tmp_path / "padre.onnx")
    torch.onnx.export(m1, (x,), ruta, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "lote"}},
                      dynamo=False)
    y_onnx = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    assert abs(y_onnx - m1(x).detach().numpy()).max() < 1e-4
