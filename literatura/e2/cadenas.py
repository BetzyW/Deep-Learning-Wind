"""Cadenas de búsqueda del Entregable 2 (revisión ampliada a modelos avanzados y búsqueda transversal de mecanismos).

Cada cadena se define una sola vez como listas de términos y se traduce a la sintaxis de cada base, para que las
consultas de Scopus, Web of Science, IEEE Xplore, Semantic Scholar y arXiv sean equivalentes.
Uso: python cadenas.py  → imprime todas las consultas (para copiarlas en las bases que se consultan a mano).
"""

PERIODO = (2018, 2026)

# --- Cadenas sobre el problema: viento en el título + una familia de arquitecturas en título, resumen o palabras clave
PROBLEMA = {
    "C1_atencion": ["transformer", "informer", "autoformer", "fedformer", "patchtst", "itransformer",
                    "temporal fusion transformer", "self-attention", "attention mechanism"],
    "C2_convolucional_y_mlp": ["temporal convolutional network", "TCN", "dilated causal convolution", "wavenet",
                               "N-BEATS", "N-HiTS", "DLinear", "TSMixer", "TimesNet"],
    "C3_espacio_de_estados": ["mamba", "state space model", "xLSTM", "Kolmogorov-Arnold", "liquid neural network",
                              "neural ODE"],
    "C4_preentrenados": ["foundation model", "pretrained", "pre-trained", "transfer learning", "fine-tuning",
                         "chronos", "timesfm", "moirai", "lag-llama", "timegpt", "large language model"],
    "C5_grafos": ["graph neural network", "graph convolutional", "spatio-temporal graph", "spatiotemporal graph",
                  "graph attention"],
    "C6_probabilisticos": ["DeepAR", "probabilistic forecasting", "quantile regression neural network",
                           "mixture density network", "normalizing flow", "diffusion model"],
}

# --- Cadenas transversales: (i) mecanismo, (ii) modalidad distinta al viento, (iii) transferencia o adaptación
TRANSVERSAL = {
    "T1_espacio_de_estados": (["mamba", "selective state space"], ["vision", "image", "language", "speech", "audio"]),
    "T2_parches": (["patch embedding", "vision transformer", "patching"], ["time series"]),
    "T3_llm_series": (["large language model", "foundation model"], ["time series"]),
    "T4_grafos_trafico": (["graph neural network", "graph convolutional"], ["traffic", "air quality", "load"]),
    "T5_pares": (["siamese", "twin neural network", "pairwise difference"], ["regression"]),
}
ADAPTACION = ["transfer", "adaptation", "adapt*", "fine-tun*"]


def _o(terminos, f=lambda t: t):
    return " OR ".join(f(f'"{t}"' if (" " in t or "-" in t) else t) for t in terminos)


def scopus(nombre):
    a, b = PERIODO
    if nombre in PROBLEMA:
        return (f'TITLE("wind speed" AND (forecast* OR predict*)) AND TITLE-ABS-KEY({_o(PROBLEMA[nombre])}) '
                f"AND PUBYEAR > {a - 1} AND PUBYEAR < {b + 1}")
    mec, mod = TRANSVERSAL[nombre]
    return (f"TITLE({_o(mec)}) AND TITLE-ABS-KEY({_o(mod)}) AND TITLE-ABS-KEY({_o(ADAPTACION)}) "
            f"AND PUBYEAR > {a - 1} AND PUBYEAR < {b + 1}")


def wos(nombre):
    a, b = PERIODO
    if nombre in PROBLEMA:
        return f'TI=("wind speed" AND (forecast* OR predict*)) AND TS=({_o(PROBLEMA[nombre])}) AND PY=({a}-{b})'
    mec, mod = TRANSVERSAL[nombre]
    return f"TI=({_o(mec)}) AND TS=({_o(mod)}) AND TS=({_o(ADAPTACION)}) AND PY=({a}-{b})"


def ieee(nombre):
    if nombre in PROBLEMA:
        return ('("Document Title":"wind speed") AND ("Document Title":forecast* OR "Document Title":predict*) AND ('
                + _o(PROBLEMA[nombre], lambda t: f'"All Metadata":{t}') + ")")
    mec, mod = TRANSVERSAL[nombre]
    return ("(" + _o(mec, lambda t: f'"Document Title":{t}') + ") AND (" + _o(mod, lambda t: f'"All Metadata":{t}')
            + ") AND (" + _o(ADAPTACION, lambda t: f'"All Metadata":{t}') + ")")


def s2(nombre):
    """Sintaxis de Semantic Scholar: + es AND, | es OR, comillas para frases, * para prefijos (título y resumen)."""
    o = lambda ts: " | ".join(f'"{t}"' if (" " in t or "-" in t) else t for t in ts)
    if nombre in PROBLEMA:
        return f'"wind speed" + (forecast* | predict*) + ({o(PROBLEMA[nombre])})'
    mec, mod = TRANSVERSAL[nombre]
    return f"({o(mec)}) + ({o(mod)}) + ({o(ADAPTACION)})"


def arxiv(nombre):
    if nombre in PROBLEMA:
        return ('ti:"wind speed" AND (ti:forecast* OR ti:predict*) AND ('
                + " OR ".join(f'abs:"{t}"' for t in PROBLEMA[nombre]) + ")")
    mec, mod = TRANSVERSAL[nombre]
    return ("(" + " OR ".join(f'ti:"{t}"' for t in mec) + ") AND (" + " OR ".join(f'abs:"{t}"' for t in mod)
            + ") AND (" + " OR ".join(f'abs:{t.rstrip("*")}' for t in ADAPTACION) + ")")


TODAS = list(PROBLEMA) + list(TRANSVERSAL)

if __name__ == "__main__":
    for n in TODAS:
        print(f"## {n}\nScopus: {scopus(n)}\nWoS:    {wos(n)}\nIEEE:   {ieee(n)}\nS2:     {s2(n)}\narXiv:  {arxiv(n)}\n")
