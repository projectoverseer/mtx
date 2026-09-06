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

# Windowing for the transformer backends.  5 s at 24 kHz is 120,000 samples,
# which attention handles in milliseconds; a whole track does not.  A 30 s hop
# over a 4-minute song is 8 views, enough to average out an intro and an
# outro without paying for every frame of the middle.
# A 30 s hop over a 4-minute song is 8 windows of 5 s each: 40 seconds looked
# at out of 240, 17% of the record.  That is enough to tell two masters of one
# recording apart (`Get Lucky` matches itself at 0.98) and thin enough that a
# track whose sampled seconds are unrepresentative lands beside the wrong
# neighbours.  15 s doubles the coverage to a third of the track for twice the
# GPU time, which for an unattended overnight job is the right trade.
WINDOW_S = 5.0
HOP_S = 15.0
MAX_WINDOWS = 24
BATCH = 8


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


_MERT_CACHE: dict[str, Any] = {}


def _mert_model(name: str):
    """Load once per process.  Loading per track dominated everything else."""
    if name not in _MERT_CACHE:
        import torch  # type: ignore
        from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            name, trust_remote_code=True)
        # Asked for on the config, not only on the call: in transformers 5.x
        # passing `output_hidden_states=True` to `forward` alone comes back
        # with `hidden_states=None`, and `torch.stack(None)` then raises a
        # TypeError that `analyse()` catches and reports as "no embedding
        # backend is installed" -- a message about the environment for a fault
        # in the call.
        model = AutoModel.from_pretrained(
            name, trust_remote_code=True, output_hidden_states=True).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _MERT_CACHE[name] = (extractor, model, device)
    return _MERT_CACHE[name]


def _try_mert(src: AudioSource) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from transformers import AutoModel, Wav2Vec2FeatureExtractor  # noqa: F401
    except Exception:
        return None
    name = "m-a-p/MERT-v1-95M"
    extractor, model, device = _mert_model(name)
    from ..audio import resample_to
    sr = int(extractor.sampling_rate)
    y = resample_to(src.mono.astype(np.float32), src.sr, sr)

    # In windows, never as one sequence.  MERT is a transformer, so attention
    # over a whole track is quadratic in its length: a four-minute song at
    # 24 kHz is ~18,000 frames, and one track took over ten minutes -- 220
    # hours for this corpus, against a few seconds a track windowed.  The
    # window is also the honest unit: a song is not one texture, and a mean
    # over five-second views of it is a better summary than one attention
    # pass that has to attend across the whole arrangement at once.
    win = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)
    if y.size < win:
        y = np.pad(y, (0, win - y.size))
    starts = list(range(0, max(1, y.size - win + 1), hop))[:MAX_WINDOWS]
    vecs = []
    layers_used = "layers"
    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            chunk = [y[a:a + win] for a in starts[i:i + BATCH]]
            inputs = extractor(chunk, sampling_rate=sr, return_tensors="pt",
                               padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            hidden = getattr(out, "hidden_states", None)
            if hidden:
                # Mean over layers, then over time: MERT's lower layers carry
                # timbre and its upper ones carry structure, and a similarity
                # question wants both.
                stacked = torch.stack(list(hidden)).mean(dim=0).mean(dim=1)
                pooled = "layers"
            else:
                # Still a usable vector, and saying which one was taken
                # matters more than always taking the same one: two tracks
                # embedded under different poolings are not comparable, and
                # nothing downstream could tell without this.
                stacked = out.last_hidden_state.mean(dim=1)
                pooled = "last layer"
            layers_used = pooled
            vecs.append(stacked.float().cpu().numpy())
    vec = np.concatenate(vecs, axis=0).mean(axis=0)
    return np.asarray(vec, dtype=float), {
        "backend": "transformers:MERT", "model": name,
        "input_sr_hz": sr, "version": transformers.__version__,
        "device": device,
        "window_s": WINDOW_S, "hop_s": HOP_S, "windows": len(starts),
        "pooling": (f"mean over {layers_used}, then over time within each "
                    f"{WINDOW_S:g}s window, then over {len(starts)} windows")}


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
