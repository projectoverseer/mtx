"""The musical half: harmony, melody, rhythm, form, masking.

Synthetic signals with known answers where the question is a DSP one, and
direct calls to the pure functions where it is not.  The two regression tests
at the top are the ones that matter most: both encode a bug that was found by
measuring this tool's output against published, human-transcribed references,
and both would be invisible to a test that only checked the code ran.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from mtx.audio import AudioSource
from mtx.params import PARAMS
from mtx.util import Collector

SR = 44100
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def tone(freq: float, dur: float, sr: int = SR, partials: int = 5) -> np.ndarray:
    """A harmonic tone, so a chroma transform has something to bite on."""
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros(t.size)
    for k in range(1, partials + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    return y / max(np.max(np.abs(y)), 1e-9)


def chord_audio(chords, seconds_each=2.0, sr=SR):
    """Render a list of (root_pc, [semitone offsets]) as sustained chords."""
    out = []
    for root, tones in chords:
        seg = np.zeros(int(seconds_each * sr))
        for off in tones:
            midi = 60 + root + off          # voiced around middle C
            seg += tone(midi_hz(midi), seconds_each, sr, partials=4)
        seg /= max(np.max(np.abs(seg)), 1e-9)
        # A short fade stops the chord change from being a click.
        n = int(0.02 * sr)
        seg[:n] *= np.linspace(0, 1, n)
        seg[-n:] *= np.linspace(1, 0, n)
        out.append(seg)
    y = np.concatenate(out) * 0.7
    return np.stack([y, y], axis=1)


# --------------------------------------------------------------------- harmony
def test_a_seventh_template_does_not_beat_the_triad_it_contains():
    """Regression: the whole reason every chord came back as a seventh.

    With plain cosine against uncentred binary masks, a four-tone mask that
    contains a triad can never score lower than the triad, because the fourth
    tone only ever adds to the numerator.  Measured against published chord
    charts that turned C into Cmaj7 or C6 nearly everywhere.
    """
    from mtx.metrics.harmony import _templates

    templates, extra_tones, meta = _templates()
    chroma = np.zeros(12)
    for pc in (0, 4, 7):                    # a clean C major triad
        chroma[pc] = 1.0
    chroma = chroma - chroma.mean()
    chroma /= np.linalg.norm(chroma)
    scores = templates @ chroma
    by_label = {m["label"]: float(s) for m, s in zip(meta, scores)}

    assert by_label["C"] == pytest.approx(max(by_label.values()))
    for richer in ("Cmaj7", "C6", "C7", "Csus4", "Csus2"):
        assert by_label["C"] > by_label[richer], (
            f"{richer} scored at least as high as the triad it contains")
    # And the complexity prior is real, so the margin survives a noisy chroma.
    assert PARAMS["harmony"]["complexity_penalty"] > 0
    assert (extra_tones[[i for i, m in enumerate(meta)
                         if m["label"] == "Cmaj7"][0]]) == 1.0


def test_chords_and_key_are_recovered_from_a_synthetic_progression(tmp_path):
    from mtx.metrics import harmony as m_harmony

    # I - vi - IV - V in C major, twice round.
    prog = [(0, [0, 4, 7]), (9, [0, 3, 7]), (5, [0, 4, 7]), (7, [0, 4, 7])] * 2
    path = tmp_path / "prog.flac"
    sf.write(path, chord_audio(prog), SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    res = m_harmony.analyse(src, {"available": False}, None, Collector())

    assert res["available"]
    roots = {c["label"][:2] if len(c["label"]) > 1 and c["label"][1] == "#"
             else c["label"][:1]
             for c in res["chords"] if c.get("quality") != "no_chord"}
    # The four roots that were played must dominate what came back.
    assert {"C", "A", "F", "G"} <= roots, roots
    key = res["key_from_chords"]
    assert key["available"]
    assert key["key"] == "C major", key
    assert key["diatonic_time_pct"] > 90.0


def test_key_from_chords_reports_the_relative_key_trap():
    """A scale fit cannot separate relative keys; the block has to say so."""
    from mtx.metrics.harmony import _key_from_chords

    # All-diatonic-to-C chords, but every one of them an A minor tonic.
    segs = [{"root_pc": 9, "quality": "min", "tones": [9, 0, 4], "duration_s": 4.0},
            {"root_pc": 5, "quality": "maj", "tones": [5, 9, 0], "duration_s": 2.0},
            {"root_pc": 7, "quality": "maj", "tones": [7, 11, 2], "duration_s": 2.0},
            {"root_pc": 9, "quality": "min", "tones": [9, 0, 4], "duration_s": 4.0}]
    got = _key_from_chords(segs)
    assert got["key"] == "A minor"
    assert got["runner_up"] == "C major"
    assert got["runner_up_is_relative_key"] is True
    assert got["confidence"] == "low"
    assert "relative" in got["confidence_reason"]


# ---------------------------------------------------------------------- melody
def test_the_vocal_range_excludes_the_trackers_octave_errors():
    """Regression: one octave error was setting the reported range.

    Checked against published vocal ranges, the raw extremes of a pyin track on
    a separated stem came out 40-58 semitones wide for singers whose range in
    the song is under two octaves.
    """
    from mtx.metrics.melody import _range_block

    def note(midi, start, dur):
        from mtx.metrics.melody import _midi_name
        return {"midi": float(midi), "note": _midi_name(midi), "start_s": start,
                "duration_s": dur, "end_s": start + dur}

    notes = [note(60 + (i % 7), i * 1.0, 1.0) for i in range(20)]   # C4..F#4
    notes.append(note(24, 20.0, 0.2))       # an octave-error outlier, C1
    notes.append(note(96, 21.0, 0.2))       # and one at C7
    block, kept = _range_block(notes, Collector())

    assert block["semitones"] < 12, "an outlier still set the range"
    assert block["lowest"]["note"] == "C4"
    assert block["highest"]["note"] == "F#4"
    assert block["octave_outliers"]["notes"] == 2
    assert block["absolute"]["semitones"] == pytest.approx(72.0)
    assert block["read_this_one"].startswith("p5_p95")
    assert len(kept) == 20


def test_range_percentiles_are_weighted_by_note_duration():
    from mtx.metrics.melody import _weighted_percentile

    values = np.array([60.0, 72.0])
    # One long low note and one short high one: the median is the long one.
    assert _weighted_percentile(values, np.array([10.0, 0.1]), 50.0) < 62.0
    assert _weighted_percentile(values, np.array([0.1, 10.0]), 50.0) > 70.0


def test_pitch_track_finds_a_synthetic_note(tmp_path):
    from mtx.metrics import melody as m_melody

    y = tone(midi_hz(69), 4.0)              # a steady A4
    path = tmp_path / "a4.flac"
    sf.write(path, np.stack([y, y], axis=1) * 0.5, SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    track = m_melody.track_f0(src, 65.4, 2093.0, Collector(), "test")
    assert track is not None
    notes = m_melody.segment_notes(track)
    assert notes, "no note segmented from a four-second steady tone"
    midis = [n["midi"] for n in notes]
    assert abs(float(np.median(midis)) - 69.0) < 0.5


# ---------------------------------------------------------------------- rhythm
def test_grid_deviation_separates_a_sequencer_from_a_player():
    from mtx.metrics.rhythm import _deviations, _grid, _timing_stats

    beats = np.arange(0, 32, 0.5)           # 120 BPM
    grid = _grid(beats, 4)
    rng = np.random.default_rng(0)
    tight = grid[:-1] + rng.normal(0, 0.002, grid.size - 1)
    loose = grid[:-1] + rng.normal(0, 0.030, grid.size - 1)

    t_stats = _timing_stats(_deviations(tight, grid, 120.0))
    l_stats = _timing_stats(_deviations(loose, grid, 120.0))
    assert t_stats["std_ms"] < PARAMS["rhythm"]["programmed_tightness_ms"]
    assert l_stats["std_ms"] > PARAMS["rhythm"]["programmed_tightness_ms"]
    assert t_stats["share_within_10ms"] > l_stats["share_within_10ms"]


def test_a_four_four_accent_pattern_yields_four_beats_per_bar(tmp_path):
    from mtx.metrics.rhythm import _downbeats

    bpm, bars = 120.0, 16
    beat = 60.0 / bpm
    n = int(bars * 4 * beat * SR)
    y = np.zeros(n)
    for i in range(bars * 4):
        at = int(i * beat * SR)
        strong = (i % 4 == 0)
        # A low thump on the downbeat, a quiet tick elsewhere.
        d = int(0.09 * SR)
        env = np.exp(-np.linspace(0, 9, d))
        f = 55.0 if strong else 900.0
        amp = 1.0 if strong else 0.25
        seg = amp * env * np.sin(2 * np.pi * f * np.arange(d) / SR)
        y[at:at + d] += seg[:max(0, min(d, n - at))]
    y *= 0.6
    path = tmp_path / "four.flac"
    sf.write(path, np.stack([y, y], axis=1), SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    beats = np.arange(bars * 4) * beat
    got = _downbeats(src, beats, Collector())

    assert got["available"]
    assert got["beats_per_bar"] == 4, got["per_meter"]
    assert got["phase_beat_index"] == 0
    assert got["bar_count"] == bars
    assert got["downbeat_accent_contrast"] > 0


def test_the_tempo_octave_check_survives_a_plain_click_track(tmp_path):
    """The reported level is not always the level the record is on.

    The click track is the case an autocorrelation test gets wrong: a periodic
    pulse train correlates with itself as well at twice its period as at its
    own, so the naive check calls a correct tempo halved.
    """
    from mtx.metrics.rhythm import _tempo_octave

    bpm = 120.0
    beat = 60.0 / bpm
    n = int(24 * SR)
    y = np.zeros(n)
    d = int(0.05 * SR)
    env = np.exp(-np.linspace(0, 9, d))
    click = env * np.sin(2 * np.pi * 200 * np.arange(d) / SR)
    for i in range(int(24 / beat)):
        at = int(i * beat * SR)
        if at + d < n:
            y[at:at + d] += click
    path = tmp_path / "click.flac"
    sf.write(path, np.stack([y, y], axis=1) * 0.6, SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    beats = np.arange(int(24 / beat)) * beat

    ok = _tempo_octave(src, beats, bpm, Collector())
    assert ok["available"], ok
    assert ok["suggested_factor"] == 1.0, ok
    assert ok["reported_level_is_best_supported"]

    # Read at half the real rate, the midpoints are full of beats.
    collector = Collector()
    half = _tempo_octave(src, beats[::2], bpm / 2.0, collector)
    assert half["suggested_factor"] == 2.0, half
    assert half["midpoint_ratio"] > 0.8
    assert any("tempo_octave" in w for w in collector.warnings)


def test_the_tempo_octave_check_spots_an_empty_alternate_beat(tmp_path):
    """Read at double the real rate, every other beat lands on nothing."""
    from mtx.metrics.rhythm import _tempo_octave

    real_beat = 1.0                        # 60 BPM of actual events
    n = int(32 * SR)
    y = np.zeros(n)
    d = int(0.05 * SR)
    env = np.exp(-np.linspace(0, 9, d))
    click = env * np.sin(2 * np.pi * 200 * np.arange(d) / SR)
    for i in range(32):
        at = int(i * real_beat * SR)
        if at + d < n:
            y[at:at + d] += click
    path = tmp_path / "slow.flac"
    sf.write(path, np.stack([y, y], axis=1) * 0.6, SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())

    doubled = np.arange(64) * 0.5          # 120 BPM read over a 60 BPM track
    collector = Collector()
    got = _tempo_octave(src, doubled, 120.0, collector)
    assert got["suggested_factor"] == 0.5, got
    assert got["alternation_ratio"] < 0.5
    assert got["suggested_bpm"] == pytest.approx(60.0)


# ------------------------------------------------------------------------ form
def test_consecutive_sections_of_one_letter_become_one_part():
    """A novelty boundary fires wherever the texture turns; a chorus is a part."""
    from mtx.metrics.form import _label, _parts

    sections = [{"index": i, "start_s": float(i * 10), "end_s": float(i * 10 + 10),
                 "duration_s": 10.0, "lufs_i": lufs}
                for i, lufs in enumerate([-20, -9, -9, -9, -14, -9, -9, -25])]
    letters = [0, 1, 1, 1, 2, 1, 1, 0]
    vocal = [False, True, True, True, True, True, True, False]

    parts = _parts(sections, letters, vocal)
    assert [p["letter"] for p in parts] == ["A", "B", "C", "B", "A"]
    assert parts[1]["members"] == [1, 2, 3]
    assert parts[1]["duration_s"] == pytest.approx(30.0)

    _label(sections, parts, Collector())
    labels = [p["label"] for p in parts]
    assert labels[0] == "intro" and labels[-1] == "outro"
    assert labels.count("chorus") == 2, labels
    # Every section inherits its part's label, and carries its evidence.
    assert sections[2]["label"] == "chorus"
    assert sections[2]["part_index"] == 1
    assert sections[2]["label_evidence"]


# --------------------------------------------------------------------- masking
def test_the_masking_index_is_a_signed_ratio_in_the_targets_own_bands():
    from mtx.metrics.masking import _masking_index_db, _overlap

    target = np.array([0.0, 1.0, 1.0, 0.0])      # lives in the middle two bands
    quiet = np.array([1.0, 0.1, 0.1, 1.0])       # loud only where the target is not
    loud = np.array([0.0, 4.0, 4.0, 0.0])        # four times the target, in its bands

    assert _masking_index_db(target, quiet) < 0
    assert _masking_index_db(target, loud) == pytest.approx(10 * math.log10(4.0), abs=1e-6)
    # Overlap is symmetric and normalised.
    a = _overlap(target, loud)
    b = _overlap(loud, target)
    assert a["cosine"] == pytest.approx(b["cosine"])
    assert a["cosine"] == pytest.approx(1.0)
    assert _overlap(target, np.array([1.0, 0.0, 0.0, 1.0]))["cosine"] == pytest.approx(0.0)


def test_masking_needs_at_least_two_stems():
    from mtx.metrics import masking as m_masking

    out = m_masking.analyse(None, {}, [], {}, Collector())
    assert out["available"] is False and "two stems" in out["reason"]


# ----------------------------------------------------------------- the tuning

def _detuned_progression(a4: float, sr: int = SR) -> np.ndarray:
    """The same I-vi-IV-V, built on a chosen reference pitch."""
    prog = [(0, [0, 4, 7]), (9, [0, 3, 7]), (5, [0, 4, 7]), (7, [0, 4, 7])] * 2
    out = []
    for root, tones in prog:
        seg = np.zeros(int(2.0 * sr))
        for off in tones:
            midi = 60 + root + off
            freq = a4 * (2.0 ** ((midi - 69.0) / 12.0))
            seg += tone(freq, 2.0, sr, partials=4)
        seg /= max(np.max(np.abs(seg)), 1e-9)
        n = int(0.02 * sr)
        seg[:n] *= np.linspace(0, 1, n)
        seg[-n:] *= np.linspace(1, 0, n)
        out.append(seg)
    y = np.concatenate(out) * 0.7
    return np.stack([y, y], axis=1)


@pytest.mark.parametrize("a4", [432.0, 444.0])
def test_the_implied_reference_pitch_is_the_one_the_track_was_built_on(tmp_path, a4):
    """Regression: A=432 was reported as A=350.

    `librosa.estimate_tuning` returns fractions of a chroma bin, and a bin is
    a semitone.  Reading that as octaves put the implied reference a factor of
    twelve out in the exponent -- reported beside a `tuning_cents` computed
    from the very same number, which was right.
    """
    import librosa

    from mtx.metrics import structure as m_structure

    path = tmp_path / f"a{a4:.0f}.flac"
    sf.write(path, _detuned_progression(a4), SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    key = m_structure._key(src, librosa, Collector())

    assert key["implied_a4_hz"] == pytest.approx(a4, abs=2.0)
    # The two figures come from one estimate and must not disagree.
    assert key["implied_a4_hz"] == pytest.approx(
        440.0 * 2.0 ** (key["tuning_cents"] / 1200.0), rel=1e-9)
    expected_cents = 1200.0 * math.log2(a4 / 440.0)
    assert key["tuning_cents"] == pytest.approx(expected_cents, abs=6.0)


def test_a_detuned_master_still_lands_the_right_key(tmp_path):
    """chroma-CQT estimates the reference from the track, so the key holds."""
    import librosa

    from mtx.metrics import structure as m_structure

    keys = {}
    for a4 in (440.0, 432.0):
        path = tmp_path / f"k{a4:.0f}.flac"
        sf.write(path, _detuned_progression(a4), SR, subtype="PCM_24")
        src = AudioSource(str(path), Collector())
        keys[a4] = m_structure._key(src, librosa, Collector())["key"]
    assert keys[432.0] == keys[440.0], \
        "a 32-cent detune must not transpose the reported key"


def test_a_detuned_master_says_so(tmp_path):
    """A master off A440 is worth telling the reader about, once it is real."""
    import librosa

    from mtx.metrics import structure as m_structure

    collector = Collector()
    path = tmp_path / "flat.flac"
    sf.write(path, _detuned_progression(432.0), SR, subtype="PCM_24")
    m_structure._key(AudioSource(str(path), Collector()), librosa, collector)
    notes = [n for n in collector.notes if n["metric"] == "structure.tuning"]
    assert notes, "a 32-cent offset is past params.structure.tuning_report_cents"
    assert "cents from A440" in notes[0]["reason"]

    quiet = Collector()
    path = tmp_path / "ref.flac"
    sf.write(path, _detuned_progression(440.0), SR, subtype="PCM_24")
    m_structure._key(AudioSource(str(path), Collector()), librosa, quiet)
    assert not [n for n in quiet.notes if n["metric"] == "structure.tuning"], \
        "a track at A440 has nothing to report"
