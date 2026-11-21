"""Modelo original del E3: PADRE-TCN, regresión por diferencias con anclas diarias causales.

Idea. En vez de predecir directamente la velocidad dentro de 24 h, el modelo predice **cuánto cambia** respecto de
situaciones parecidas ya observadas (anclas) y suma ese cambio al valor conocido de cada ancla. Las anclas son las
ventanas que terminan a la misma hora de los K días anteriores, dentro de la misma ventana de 168 h, así que no usa
ningún dato que los modelos del benchmark no vean. El ancla 1 es la persistencia (su objetivo es la velocidad actual),
de modo que el modelo generaliza la corrección sobre la persistencia a varios días.

Fuentes del mecanismo (transferido y adaptado):
* Regresión por diferencias entre pares (PADRE): Tynes et al. (2021), J. Chem. Inf. Model. 61(8), 3846–3857,
  doi:10.1021/acs.jcim.1c00670 (química, con bosques aleatorios).
* Red gemela de regresión (TNNR): Wetzel et al. (2022), Applied AI Letters 3(4), e78, doi:10.1002/ail2.78.
* Ensamble de análogos: Delle Monache et al. (2013), Monthly Weather Review 141(10), 3498–3516.
Modificaciones algorítmicas propias (ver experiments/procedencia.md):
  (M1) anclas temporales causales dentro de la misma ventana, en lugar de puntos del conjunto de entrenamiento;
  (M2) codificador causal compartido: una sola pasada de la TCN da la representación de la consulta y de todas las
       anclas (cada ancla ve solo su pasado dentro de la ventana);
  (M3) diferencia **antisimétrica por construcción**, D(a, b) = g(a, b) − g(b, a), en vez de una penalización;
  (M4) regla de fusión aprendida: pesos softmax por similitud en un espacio proyectado más un sesgo por día, en
       lugar del promedio uniforme de TNNR; la dispersión ponderada entre anclas da la incertidumbre.

Formulación (notación del capítulo de diseño):
  h_t = TCN(x)_t,  q = h_{L−1},  a_j = h_{L−1−24 j},  y_j = v_{L−1−24 (j−1)}   (j = 1, …, K)
  D(q, a_j) = g([q, a_j, q − a_j, q ⊙ a_j]) − g([a_j, q, a_j − q, a_j ⊙ q])
  ŷ_j = y_j + D(q, a_j),  s_j = −‖P q − P a_j‖² / T + β_j,  w = softmax(s),  ŷ = Σ_j w_j ŷ_j
  pérdida = (ŷ − y)² + λ · Σ_j w̄_j (ŷ_j − y)²,  w̄ = w sin gradiente
Escrito desde estas ecuaciones; no se usó código de terceros.
"""
import torch
import torch.nn as nn

from modelos_e2 import TCN

H_DIA = 24


class DiferenciaAntisimetrica(nn.Module):
    """D(a, b) = g(φ(a, b)) − g(φ(b, a)) con φ(a, b) = [a, b, a − b, a ⊙ b]. Antisimétrica para cualquier g."""

    def __init__(self, k, oculta=32):
        super().__init__()
        # sin sesgo en la última capa: en g(a, b) − g(b, a) se cancelaría y nunca recibiría gradiente
        self.g = nn.Sequential(nn.Linear(4 * k, oculta), nn.ReLU(), nn.Linear(oculta, 1, bias=False))

    @staticmethod
    def phi(a, b):
        return torch.cat([a, b, a - b, a * b], -1)

    def forward(self, a, b):
        return (self.g(self.phi(a, b)) - self.g(self.phi(b, a))).squeeze(-1)


class DiferenciaLibre(nn.Module):
    """Ablación de M3: la misma red sin restar el orden inverso (no garantiza D(a, b) = −D(b, a))."""

    def __init__(self, k, oculta=32):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(4 * k, oculta), nn.ReLU(), nn.Linear(oculta, 1))

    def forward(self, a, b):
        return self.g(DiferenciaAntisimetrica.phi(a, b)).squeeze(-1)


class PadreTCN(nn.Module):
    """PADRE-TCN. Entrada x: (B, L, d) escalada; la columna 0 es la velocidad. Salida: pronóstico (B,) en m/s.

    Hiperparámetros propios: K (anclas, 1 ≤ K ≤ ⌊(L − 1)/24⌋), lam (peso de la pérdida por ancla), oculta (ancho de g)
    y proy (dimensión de la proyección P). Banderas de ablación: antisimetrica, pesos ("aprendidos" o "uniformes").
    """

    def __init__(self, L, d, w=(64, 64, 16), b=(1, 1, 0), K=6, lam=0.5, oculta=32, proy=8, mu_y=0.0, sd_y=1.0,
                 antisimetrica=True, pesos="aprendidos", **_):
        super().__init__()
        if not 1 <= K <= (L - 1) // H_DIA:
            raise ValueError(f"K = {K} fuera de [1, {(L - 1) // H_DIA}] con L = {L}")
        self.L, self.K, self.lam = L, K, lam
        self.register_buffer("mu_y", torch.tensor(float(mu_y)))
        self.register_buffer("sd_y", torch.tensor(float(sd_y)))
        tcn = TCN(L, d, w=w, b=b)
        self.codificador = tcn.bloques                                   # solo los bloques causales, sin la salida
        k = w[-1]
        self.D = (DiferenciaAntisimetrica if antisimetrica else DiferenciaLibre)(k, oculta)
        self.pesos = pesos
        self.P = nn.Linear(k, proy, bias=False)
        self.log_T = nn.Parameter(torch.zeros(()))                       # temperatura T = exp(log_T)
        self.beta = nn.Parameter(torch.zeros(K))                          # sesgo por día de ancla
        q_idx = L - 1
        self.register_buffer("idx_ancla", torch.tensor([q_idx - H_DIA * j for j in range(1, K + 1)]))
        self.register_buffer("idx_obj", torch.tensor([q_idx - H_DIA * (j - 1) for j in range(1, K + 1)]))
        self._ultimo = None

    def componentes(self, x):
        """Devuelve (ŷ, ŷ_j, w): pronóstico, pronóstico de cada ancla (B, K) y pesos (B, K)."""
        if x.dim() != 3 or x.shape[1] != self.L:
            raise ValueError(f"se esperaba x de forma (B, {self.L}, d) y llegó {tuple(x.shape)}")
        h = self.codificador(x.transpose(1, 2)).transpose(1, 2)           # (B, L, k): h_t ve solo x_{≤t}
        q = h[:, -1]                                                      # consulta
        a = h[:, self.idx_ancla]                                          # (B, K, k) anclas
        y_a = x[:, self.idx_obj, 0] * self.sd_y + self.mu_y               # objetivos conocidos de las anclas (m/s)
        qk = q.unsqueeze(1).expand_as(a)
        y_j = y_a + self.D(qk, a)                                         # (B, K)
        if self.pesos == "uniformes":
            w = torch.full_like(y_j, 1.0 / self.K)
        else:
            dist = ((self.P(qk) - self.P(a)) ** 2).sum(-1)
            w = torch.softmax(-dist / self.log_T.exp() + self.beta, -1)
        return (w * y_j).sum(-1), y_j, w

    def forward(self, x):
        y, y_j, w = self.componentes(x)
        self._ultimo = (y_j, w)
        return y

    def perdida_aux(self, y):
        """λ · Σ_j w̄_j (ŷ_j − y)² del último forward: entrena cada diferencia, no solo la combinación."""
        if self._ultimo is None or self.lam == 0:
            return 0.0
        y_j, w = self._ultimo
        return self.lam * (w.detach() * (y_j - y.unsqueeze(-1)) ** 2).sum(-1).mean()

    @torch.no_grad()
    def dispersion(self, x):
        """Desviación ponderada entre anclas, √Σ_j w_j (ŷ_j − ŷ)²: base de los intervalos conformes normalizados."""
        y, y_j, w = self.componentes(x)
        return (w * (y_j - y.unsqueeze(-1)) ** 2).sum(-1).sqrt()
