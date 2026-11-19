"""Metadatos del artículo original de cada arquitectura candidata (Semantic Scholar, búsqueda por título exacto).

Uso (desde literatura/e2): python originales.py  → originales.json
Se guarda lo que devuelve la API (título, año, venue, DOI, arXiv, citas) para verificar a mano cada coincidencia.
"""
import json, time, urllib.parse, urllib.request, urllib.error

CANDIDATOS = {
    "Transformer": "Attention Is All You Need",
    "Informer": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
    "Autoformer": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
    "FEDformer": "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
    "PatchTST": "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
    "iTransformer": "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
    "TFT": "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
    "TCN": "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling",
    "WaveNet": "WaveNet: A Generative Model for Raw Audio",
    "TimesNet": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
    "N-BEATS": "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting",
    "N-HiTS": "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting",
    "DLinear": "Are Transformers Effective for Time Series Forecasting?",
    "TSMixer": "TSMixer: An All-MLP Architecture for Time Series Forecasting",
    "TiDE": "Long-term Forecasting with TiDE: Time-series Dense Encoder",
    "GRU": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
    "xLSTM": "xLSTM: Extended Long Short-Term Memory",
    "DeepAR": "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks",
    "Mamba": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    "KAN": "KAN: Kolmogorov-Arnold Networks",
    "STGCN": "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
    "Graph WaveNet": "Graph WaveNet for Deep Spatial-Temporal Graph Modeling",
    "Chronos": "Chronos: Learning the Language of Time Series",
    "TimesFM": "A decoder-only foundation model for time-series forecasting",
    "Moirai": "Unified Training of Universal Time Series Forecasting Transformers",
    "Lag-Llama": "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting",
    "Time-LLM": "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
    "TimeGrad": "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting",
}
CAMPOS = "title,year,venue,publicationVenue,externalIds,citationCount,authors,publicationDate"


def buscar(titulo):
    u = "https://api.semanticscholar.org/graph/v1/paper/search/match?" + urllib.parse.urlencode(
        {"query": titulo, "fields": CAMPOS})
    for i in range(6):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                u, headers={"User-Agent": "revision-sistematica-viento/1.0"}), timeout=60).read())
            return (r.get("data") or [None])[0]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(15 * (i + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):   # corte de red o DNS: reintentar
            time.sleep(15 * (i + 1))


if __name__ == "__main__":
    import os
    out = json.load(open("originales.json", encoding="utf-8")) if os.path.exists("originales.json") else {}
    for k, t in CANDIDATOS.items():
        if out.get(k, {}).get("titulo"):                 # ya consultado: se reanuda donde quedó
            continue
        p = buscar(t)
        time.sleep(3)
        if p:
            out[k] = dict(buscado=t, titulo=p["title"], anio=p.get("year"), venue=p.get("venue"),
                          doi=(p.get("externalIds") or {}).get("DOI"), arxiv=(p.get("externalIds") or {}).get("ArXiv"),
                          citas=p.get("citationCount"), autores=[a["name"] for a in (p.get("authors") or [])][:3])
        else:
            out[k] = dict(buscado=t, titulo=None)
        print(f"{k:14s} {out[k].get('anio')} {out[k].get('citas')!s:>7} | {out[k].get('titulo')}", flush=True)
        json.dump(out, open("originales.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
