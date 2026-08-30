"""4.12 Lyrics: acquired, located in time, and read rather than only counted.

Three separate things were missing, and they need separate answers.

*Acquisition.*  Lyrics came from file tags only, so any lyric feature was
absent on whatever share of a corpus happened to be tagged -- its presence
correlated with the tagger, not with the song.  The fix is a priority order:
a **declared** lyric (you wrote the song, you hold the authoritative text)
beats a **tag**, and a **transcript** from the vocal stem is the fallback that
covers everything else.  A transcript is a guess at a lyric and mishears, so it
is labelled as an inference and never merged into a declared text.

*Alignment.*  Nothing could be located in time -- not the hook, not the title,
not the first line.  Word timings come from the transcript backend, and their
payoff is rhythmic rather than textual: syllables per second against the beat
grid is a **delivery rate**.

*Semantics.*  The existing statistics are shape, not meaning.  What can be
measured from the text alone is measured here.  What needs a lexicon nobody
ships (valence, concreteness) reports `available: false` with what to install,
because a made-up sentiment number is worse than none.

The syllable counter is an English heuristic.  It now detects the language
first and declines rather than producing a meaningless number.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from typing import Any

import numpy as np

from ..params import PARAMS
from ..util import Collector, fmt_time

# Small stop-word sets, used only to pick a language -- never for filtering.
STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the a an and or but if of to in it is was you i me my we "
                    "they he she that this with for on at be have do not no my "
                    "your all so what when know like just".split()),
    "es": frozenset("el la los las de que y en un una por con no me te se lo "
                    "para mi si como mas pero yo tu su ya".split()),
    "fr": frozenset("le la les de des et en un une que qui pour dans pas je tu "
                    "il elle nous vous sur ne plus mais comme".split()),
    "de": frozenset("der die das und ist ich nicht du mit sie es ein eine den "
                    "dem auf fur war wir aber wie noch nur".split()),
    "pt": frozenset("o a os as de que e em um uma por com nao me te se para meu "
                    "mais mas eu voce ja como".split()),
    "it": frozenset("il lo la i gli le di che e in un una per con non mi ti si "
                    "ma piu come sono anche".split()),
    "vi": frozenset("va la cua co khong nguoi mot nhung den cho toi anh em minh "
                    "nay duoc trong ra ve khi con".split()),
    "id": frozenset("yang dan di ke dari untuk dengan tidak aku kamu ini itu "
                    "ada akan saya kita bisa sudah".split()),
}
SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("han", 0x4E00, 0x9FFF), ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF), ("hangul", 0xAC00, 0xD7AF),
    ("cyrillic", 0x0400, 0x04FF), ("arabic", 0x0600, 0x06FF),
    ("devanagari", 0x0900, 0x097F), ("thai", 0x0E00, 0x0E7F),
    ("hebrew", 0x0590, 0x05FF), ("greek", 0x0370, 0x03FF),
)
SCRIPT_LANGUAGE = {"hangul": "ko", "thai": "th", "arabic": "ar",
                   "cyrillic": "ru", "devanagari": "hi", "hebrew": "he",
                   "greek": "el"}

PRONOUNS = {
    "first_singular": ("i", "me", "my", "mine", "myself", "i'm", "i'll", "i've", "i'd"),
    "first_plural": ("we", "us", "our", "ours", "ourselves", "we're", "we'll"),
    "second": ("you", "your", "yours", "yourself", "you're", "you'll", "u"),
    "third": ("he", "she", "him", "her", "his", "hers", "they", "them",
              "their", "theirs", "it", "its"),
}
PAST_MARKERS = ("was", "were", "had", "did", "went", "said", "used to", "would")
EXPLICIT_TERMS = ("fuck", "shit", "bitch", "nigga", "cunt", "motherfucker",
                  "dick", "pussy", "cock", "whore", "asshole")
VOWEL_GROUPS = re.compile(r"[aeiouy]+")
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def detect_language(text: str) -> dict[str, Any]:
    """Script share first, then stop-word frequency over a fixed table."""
    if not text.strip():
        return {"available": False, "reason": "empty text"}
    counts = Counter()
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        for name, lo, hi in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
        else:
            counts["latin"] += 1
    if letters == 0:
        return {"available": False, "reason": "no letters in the text"}
    script, n = counts.most_common(1)[0]
    share = n / letters
    if script in SCRIPT_LANGUAGE and share > 0.3:
        return {"available": True, "language": SCRIPT_LANGUAGE[script],
                "basis": "script", "script": script, "script_share": share,
                "confidence": "medium"}
    if script in ("han", "hiragana", "katakana"):
        kana = counts["hiragana"] + counts["katakana"]
        lang = "ja" if kana > 0 else "zh"
        return {"available": True, "language": lang, "basis": "script",
                "script": script, "script_share": share, "confidence": "medium"}
    words = [_fold(w) for w in WORD.findall(text)]
    if not words:
        return {"available": False, "reason": "no words after tokenising"}
    scored = {lang: sum(1 for w in words if w in sw) / len(words)
              for lang, sw in STOPWORDS.items()}
    best = max(scored, key=lambda k: scored[k])
    ordered = sorted(scored.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    conf = "medium" if (scored[best] > 0.08 and margin > 0.02) else "low"
    return {"available": True, "language": best, "basis": "stop-word frequency",
            "stopword_share": scored[best], "margin": margin,
            "per_language_share": scored, "script": "latin",
            "confidence": conf}


def _syllables_en(word: str) -> int:
    w = _fold(word).strip("'")
    if not w:
        return 0
    groups = VOWEL_GROUPS.findall(w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


def _rime(word: str) -> tuple[str, str] | None:
    """Crude English rime: the last vowel group and everything after it."""
    w = _fold(word).strip("'")
    if not w:
        return None
    groups = list(VOWEL_GROUPS.finditer(w))
    if not groups:
        return None
    last = groups[-1]
    return w[last.start():last.end()], w[last.end():]


def _rhymes(lines: list[str]) -> dict[str, Any]:
    """End-rhyme scheme, density, and the perfect/slant split."""
    ends = []
    for ln in lines:
        ws = WORD.findall(ln)
        ends.append(_rime(ws[-1]) if ws else None)
    scheme: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    letters = "abcdefghijklmnopqrstuvwxyz"
    perfect = slant = 0
    for i, r in enumerate(ends):
        if r is None:
            scheme.append("-")
            continue
        if r in seen:
            scheme.append(seen[r])
        else:
            tag = letters[len(seen) % 26]
            seen[r] = tag
            scheme.append(tag)
    for i in range(len(ends)):
        for j in range(i + 1, min(i + 5, len(ends))):
            a, b = ends[i], ends[j]
            if a is None or b is None:
                continue
            if a == b:
                perfect += 1
            elif a[0] == b[0] or a[1] == b[1]:
                slant += 1
    internal = 0
    for ln in lines:
        ws = [w for w in WORD.findall(ln)]
        rimes = [_rime(w) for w in ws[:-1]]
        end = _rime(ws[-1]) if ws else None
        if end is not None:
            internal += sum(1 for r in rimes if r == end)
    pairs = perfect + slant
    return {
        "available": True,
        "scheme": "".join(scheme)[:400],
        "rhyming_line_pairs_within_4_lines": pairs,
        "perfect": perfect, "slant": slant,
        "perfect_share": (perfect / pairs) if pairs else None,
        "internal_rhymes": internal,
        "density_per_line": (pairs / len(lines)) if lines else None,
        "method": "the last vowel group of the final word of each line, with "
                  "everything after it, compared within a four-line window; "
                  "identical rime is perfect, a shared nucleus or a shared coda "
                  "is slant",
        "confidence": "low",
        "confidence_reason": "spelling-based rimes, not phonemes; English "
                             "orthography makes this an approximation",
    }


def _text_stats(text: str, language: str | None, title: str | None
                ) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    words = WORD.findall(text)
    folded = [_fold(w) for w in words]
    unique = set(folded)
    line_counts = Counter(_fold(" ".join(WORD.findall(ln))) for ln in lines)
    repeated_lines = sum(c for l, c in line_counts.items() if c > 1 and l)
    raw = text.encode("utf-8")
    compressed = len(zlib.compress(raw, 9))

    ngram_max = int(PARAMS["lyrics"]["ngram_max"])
    longest: dict[str, Any] = {"n": 0, "text": None, "occurrences": 0}
    # The longest repeated phrase and the most repeated one are different
    # questions, and for a lyric the second is usually the hook.
    most: dict[str, Any] = {"n": 0, "text": None, "occurrences": 0}
    for n in range(2, min(ngram_max, max(2, len(folded))) + 1):
        grams = Counter(tuple(folded[i:i + n]) for i in range(len(folded) - n + 1))
        best = grams.most_common(1)
        if best and best[0][1] > 1:
            longest = {"n": n, "text": " ".join(best[0][0]),
                       "occurrences": best[0][1]}
            if (best[0][1] > most["occurrences"]
                    or (best[0][1] == most["occurrences"] and n > most["n"])):
                most = {"n": n, "text": " ".join(best[0][0]),
                        "occurrences": best[0][1]}
    pronouns = {k: sum(1 for w in folded if w in v) for k, v in PRONOUNS.items()}
    total_pron = sum(pronouns.values()) or 1

    out: dict[str, Any] = {
        "characters": len(text),
        "words": len(words),
        "unique_words": len(unique),
        "type_token_ratio": (len(unique) / len(words)) if words else None,
        "lines": len(lines),
        "words_per_line": (len(words) / len(lines)) if lines else None,
        "repeated_line_pct": (100.0 * repeated_lines / len(lines)) if lines else None,
        "compression_ratio": (len(raw) / compressed) if compressed else None,
        "compression_note": "zlib level 9 over the UTF-8 text; a higher ratio "
                            "means more repetition",
        "longest_repeated_ngram": longest,
        "most_repeated_ngram": most,
        "ngram_note": "the longest phrase that repeats, and the phrase that "
                      "repeats most; for a lyric the second is usually the hook",
        "pronoun_counts": pronouns,
        "pronoun_share": {k: v / total_pron for k, v in pronouns.items()},
        "explicit_terms": {
            "count": sum(1 for w in folded if w in EXPLICIT_TERMS),
            "distinct": sorted({w for w in folded if w in EXPLICIT_TERMS}),
            "wordlist_size": len(EXPLICIT_TERMS),
            "note": "an English wordlist counted in the text, independent of "
                    "the Explicit flag a database returns",
        },
    }
    if title:
        t = _fold(" ".join(WORD.findall(title)))
        body = _fold(" ".join(folded))
        out["title_in_lyric"] = {
            "title_used": title,
            "occurrences": body.count(t) if t else 0,
            "first_line_index": next((i for i, ln in enumerate(lines)
                                      if t and t in _fold(ln)), None),
        }
    if language == "en":
        syls = [_syllables_en(w) for w in words]
        per_line = [sum(_syllables_en(w) for w in WORD.findall(ln)) for ln in lines]
        out["syllables"] = {
            "total": int(sum(syls)),
            "per_line_mean": float(np.mean(per_line)) if per_line else None,
            "per_line_median": float(np.median(per_line)) if per_line else None,
            "per_word_mean": float(np.mean(syls)) if syls else None,
            "counter": "English vowel-group heuristic",
        }
        # Flesch reading ease, over the same counts.
        n_sent = max(1, len(lines))
        if words:
            out["readability_flesch"] = (
                206.835 - 1.015 * (len(words) / n_sent)
                - 84.6 * (sum(syls) / len(words)))
    else:
        out["syllables"] = {
            "available": False,
            "reason": f"the syllable counter is an English heuristic and the "
                      f"detected language is {language or 'unknown'}; a number "
                      "here would be meaningless",
        }
        out["readability_flesch"] = None
    return out


def _sentiment(text: str, lines: list[str]) -> dict[str, Any]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return {
            "available": False,
            "reason": "no valence lexicon is installed and mtx ships none; a "
                      "sentiment number without a stated lexicon would not be "
                      "reproducible",
            "install": "pip install vaderSentiment",
            "lexicon_param": "params.lyrics.lexicons.valence",
        }
    an = SentimentIntensityAnalyzer()
    whole = an.polarity_scores(text)
    arc = [{"line": i, "compound": an.polarity_scores(ln)["compound"]}
           for i, ln in enumerate(lines)]
    vals = [a["compound"] for a in arc]
    return {
        "available": True, "backend": "vaderSentiment",
        "whole_text": whole,
        "arc_per_line": arc[:400],
        "arc_summary": {
            "mean": float(np.mean(vals)) if vals else None,
            "first_quarter_mean": float(np.mean(vals[:max(1, len(vals) // 4)])) if vals else None,
            "last_quarter_mean": float(np.mean(vals[-max(1, len(vals) // 4):])) if vals else None,
            "range": (float(max(vals) - min(vals)) if vals else None),
        },
        "caveat": "a lexicon score over song lyrics, not a reading of them",
    }


def _lexicon_block(name: str) -> dict[str, Any]:
    path = (PARAMS["lyrics"]["lexicons"] or {}).get(name)
    if not path:
        return {"available": False,
                "reason": f"no {name} lexicon is configured; set "
                          f"params.lyrics.lexicons.{name} to a two-column "
                          "word,score file to enable it"}
    return {"available": False,
            "reason": f"the configured {name} lexicon at {path} was not loaded",
            "path": path}


def transcribe(vocal_path: str, collector: Collector) -> dict[str, Any]:
    """A time-aligned transcript from the vocal stem, if a backend is installed."""
    P = PARAMS["lyrics"]["transcript"]
    try:
        import whisper_timestamped  # type: ignore
        import whisper  # type: ignore
    except Exception:
        pass
    else:
        try:
            model = whisper.load_model("base")
            result = whisper_timestamped.transcribe(model, vocal_path, vad=False)
            words = [{"word": w["text"], "start_s": float(w["start"]),
                      "end_s": float(w["end"])}
                     for seg in result.get("segments", [])
                     for w in seg.get("words", [])]
            return {"available": True, "backend": "whisper_timestamped",
                    "model": "base", "text": result.get("text", "").strip(),
                    "language": result.get("language"), "words": words,
                    "source": "transcript",
                    "caveat": "a transcription of a separated vocal stem: an "
                              "inference, which mishears, and never a lyric sheet"}
        except Exception as exc:
            collector.warn("lyrics.transcript", f"whisper_timestamped failed: {exc!r}")
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return {"available": False,
                "reason": "no transcription backend is installed",
                "backends": list(P["backends"]),
                "install": "pip install whisper-timestamped  # or faster-whisper"}
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(vocal_path, word_timestamps=True)
        words, chunks = [], []
        for seg in segments:
            chunks.append(seg.text)
            for w in (seg.words or []):
                words.append({"word": w.word, "start_s": float(w.start),
                              "end_s": float(w.end)})
        return {"available": True, "backend": "faster_whisper", "model": "base",
                "text": "".join(chunks).strip(), "language": info.language,
                "words": words, "source": "transcript",
                "caveat": "a transcription of a separated vocal stem: an "
                          "inference, which mishears, and never a lyric sheet"}
    except Exception as exc:
        collector.warn("lyrics.transcript", f"faster_whisper failed: {exc!r}")
        return {"available": False, "reason": repr(exc)}


def _alignment(transcript: dict[str, Any], tempo: dict[str, Any] | None,
               language: str | None, title: str | None) -> dict[str, Any]:
    """Word timings, and the rhythmic measurement they make possible."""
    if not transcript.get("available") or not transcript.get("words"):
        return {"available": False,
                "reason": "no word timings; alignment needs a transcript "
                          "backend (see lyrics.transcript)"}
    words = transcript["words"]
    span = float(words[-1]["end_s"]) - float(words[0]["start_s"])
    voiced = sum(float(w["end_s"]) - float(w["start_s"]) for w in words)
    syl = (sum(_syllables_en(w["word"]) for w in words)
           if language == "en" else None)
    bpm = (tempo or {}).get("bpm") if isinstance(tempo, dict) else None
    out: dict[str, Any] = {
        "available": True,
        "source": "transcript",
        "word_count": len(words),
        "first_word_s": float(words[0]["start_s"]),
        "first_word": fmt_time(float(words[0]["start_s"])),
        "span_s": span,
        "voiced_time_s": voiced,
        "words_per_second_of_voicing": (len(words) / voiced) if voiced > 0 else None,
        "delivery_rate": {
            "syllables_per_second": (syl / voiced) if (syl and voiced > 0) else None,
            "syllables_per_beat": ((syl / voiced) * (60.0 / bpm)
                                   if (syl and voiced > 0 and bpm) else None),
            "bpm_used": bpm,
            "note": "syllables per second of voicing is a rhythmic measurement, "
                    "not a text one; the beat-relative form needs a tempo",
        },
        "caveat": transcript.get("caveat"),
    }
    if title:
        t = _fold(" ".join(WORD.findall(title)))
        stream = [(_fold(w["word"]).strip(), float(w["start_s"])) for w in words]
        joined = " ".join(s for s, _ in stream)
        idx = joined.find(t) if t else -1
        first = None
        if idx >= 0:
            spoken = joined[:idx].count(" ")
            if spoken < len(stream):
                first = stream[spoken][1]
        out["title_first_sung_s"] = first
        out["title_first_sung"] = fmt_time(first) if first is not None else None
    return out


def analyse(tags: dict[str, Any], declared: dict[str, Any] | None,
            stems: dict[str, Any] | None, structure: dict[str, Any] | None,
            collector: Collector, want_transcript: bool = False) -> dict[str, Any]:
    """Pick a lyric source, then measure the text and (if aligned) its timing."""
    from ..declared import declared_value

    named = (tags or {}).get("named") or {}
    allt = (tags or {}).get("all") or {}
    title = named.get("title")

    text = None
    source = None
    declared_text = declared_value(declared or {}, "lyrics")
    if isinstance(declared_text, str) and declared_text.strip():
        text, source = declared_text, "declared"
    else:
        for key, value in allt.items():
            if "lyric" in str(key).lower() and isinstance(value, str) and value.strip():
                text, source = value, "file:tag"
                break

    transcript: dict[str, Any] = {"available": False,
                                  "reason": "not requested; pass --transcribe "
                                            "to run a transcription backend "
                                            "over the vocal stem"}
    if want_transcript:
        vocal = ((stems or {}).get("stems") or {}).get("vocals") or {}
        vpath = vocal.get("path")
        if not vpath:
            transcript = {"available": False,
                          "reason": "no vocals stem; transcription needs --stems"}
        else:
            transcript = transcribe(vpath, collector)
        if text is None and transcript.get("available"):
            text, source = transcript.get("text"), "transcript"

    out: dict[str, Any] = {
        "source": source,
        "source_priority": list(PARAMS["lyrics"]["sources_in_priority_order"]),
        "transcript": transcript,
    }
    if not text or not text.strip():
        out.update({
            "available": False,
            "reason": "no lyric from any source: nothing declared, no lyric tag, "
                      "and no transcript",
            "coverage_note": "lyric coverage that depends on how a file was "
                             "tagged is coverage of the tagger, not of the "
                             "corpus; a declared sidecar or a transcript closes it",
        })
        return out

    lang = detect_language(text)
    language = lang.get("language") if lang.get("available") else None
    lang_declared = declared_value(declared or {}, "lyrics_language")
    if lang_declared:
        lang["declared_language"] = lang_declared
        lang["agrees_with_declared"] = bool(lang_declared == language)
        language = lang_declared
        lang["language_used"] = lang_declared
        lang["basis"] = "declared"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    out.update({
        "available": True,
        "text_available": True,
        "is_inference": bool(source == "transcript"),
        "language": lang,
        "statistics": _text_stats(text, language, title),
        "rhyme": (_rhymes(lines) if language == "en" else
                  {"available": False,
                   "reason": f"the rhyme rules here are English-specific and the "
                             f"language is {language or 'unknown'}"}),
        "sentiment": _sentiment(text, lines),
        "concreteness": _lexicon_block("concreteness"),
        "alignment": _alignment(transcript, (structure or {}).get("tempo"),
                                language, title),
    })
    if source == "transcript":
        collector.low_confidence("lyrics", "low",
                                 "the lyric is a transcript of a separated "
                                 "vocal stem, not a lyric sheet")
    return out
