"""4.13 Learned embeddings: a fingerprint, kept away from the measurements.

193 hand-engineered scalars are ideal for interpretability and poor at timbre
similarity -- two records can agree on every column in this tool and sound
nothing alike.  One embedding vector per track (and per section) gives
nearest-neighbour search across a corpus, which is a useful capability on its
own and independent of any model anyone might later train.

An embedding is opaque, and opacity is against the grain of this tool, so three
rules hold and are enforced by where the block lives:

1. It is stored in its own block with the model name and version.
2. It is never mixed in with measured quantities.
3. No measured value is ever derived from it.

The nearest-neighbour search that consumes these vectors is a corpus-level
question and lives in `mtx cohort`, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import AudioSource
from ..params import PARAMS
from ..util import Collector

BACKENDS = ("laion_clap", "openl3", "transformers:MERT")


def _try_laion_clap(src: AudioSource) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import laion_clap  # type: ignore
    except Exception:
        return None
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    from ..audio import resample_to
    y = resample_to(src.mono.astype(np.float32), src.sr, 48000)
    vec = model.get_audio_embedding_from_data(x=y[None, :], use_tensor=False)[0]
    return np.asarray(vec, dtype=float), {
        "backend": "laion_clap", "input_sr_hz": 48000,
        "version": getattr(laion_clap, "__version__", "unknown")}


def _try_openl3(src: AudioSource) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import openl3  # type: ignore
    except Exception:
        return None
    emb, _ = openl3.get_audio_embedding(src.mono, src.sr, content_type="music",
                                        embedding_size=512, verbose=0)
    return np.asarray(emb, dtype=float).mean(axis=0), {
        "backend": "openl3", "content_type": "music", "embedding_size": 512,
        "version": getattr(openl3, "__version__", "unknown"),
        "pooling": "mean over the model's own frames"}


def _try_mert(src: AudioSource) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
    except Exception:
        return None
    name = "m-a-p/MERT-v1-95M"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
    from ..audio import resample_to
    sr = int(extractor.sampling_rate)
    y = resample_to(src.mono.astype(np.float32), src.sr, sr)
    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
    vec = torch.stack(hidden).mean(dim=0).mean(dim=1).squeeze(0).numpy()
    import transformers  # type: ignore
    return np.asarray(vec, dtype=float), {
        "backend": "transformers:MERT", "model": name,
        "input_sr_hz": sr, "version": transformers.__version__,
        "pooling": "mean over layers, then over time"}


def _section_vectors(src: AudioSource, sections: list[dict[str, Any]],
                     fn) -> list[dict[str, Any]]:
    import soundfile as sf
    import os
    import tempfile

    out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mtx_embed_") as tmp:
        for s in sections[:40]:
            t0, t1 = float(s["start_s"]), float(s["end_s"])
            a, b = int(t0 * src.sr), int(t1 * src.sr)
            if b - a < src.sr:
                continue
            p = os.path.join(tmp, f"sec{s.get('index')}.wav")
            sf.write(p, src.x[a:b], src.sr)
            sub = AudioSource(p, Collector())
            got = fn(sub)
            if got is None:
                break
            vec, _ = got
            out.append({"section_index": s.get("index"), "start_s": t0,
                        "vector": [float(v) for v in vec]})
    return out


def analyse(src: AudioSource, sections: list[dict[str, Any]],
            collector: Collector, enabled: bool = False) -> dict[str, Any]:
    """One vector per track, and optionally one per section."""
    P = PARAMS["embedding"]
    if not enabled:
        return {
            "available": False,
            "reason": "not requested; pass --embed to compute an embedding",
            "backends": list(BACKENDS),
            "note": P["note"],
        }
    for fn in (_try_laion_clap, _try_openl3, _try_mert):
        try:
            got = fn(src)
        except Exception as exc:
            collector.warn("embedding", f"{fn.__name__} failed: {exc!r}")
            continue
        if got is None:
            continue
        vec, meta = got
        out: dict[str, Any] = {
            "available": True,
            "source": "model",
            "dimensions": int(vec.size),
            "vector": [float(v) for v in vec],
            "l2_norm": float(np.linalg.norm(vec)),
            "note": P["note"],
            "nearest_neighbour_note": "similarity search over a folder of these "
                                      "vectors is `mtx cohort --neighbours`",
        }
        out.update(meta)
        if P.get("section_embeddings") and sections:
            try:
                out["per_section"] = _section_vectors(src, sections, fn)
            except Exception as exc:
                collector.warn("embedding", f"section embeddings failed: {exc!r}")
        return out
    return {
        "available": False,
        "reason": "no embedding backend is installed",
        "backends": list(BACKENDS),
        "install": ["pip install laion-clap", "pip install openl3",
                    "pip install transformers torch"],
        "note": P["note"],
    }
