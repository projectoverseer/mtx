"""The Tier-1 property spec: which measurements become queryable in Notion.

`analysis.json` holds ~4,261 fields and `mtx export` flattens ~2,091 of them.
Neither number can be a Notion property list: a database query returns every
property of every matched row, so a 2,000-column database makes one query
answer a megabyte.  Nothing is discarded for that reason -- the full row, the
section table and the chord track all ride along in the page body, and
`analysis.json` itself stays on disk under its sha256.  What this file decides
is narrower: **which fields you can filter, sort and benchmark on.**

The test for membership is one question: *would you ever compare this across
tracks?*  "What should I master an EDM club track to" needs LUFS-I, true peak
and PSR from every track in a cohort at once, so those are properties.  The
reverb pre-delay of the vocal stem matters only once you have already chosen a
track, so it is not.

Three groups exist here that no flat export produces, because the values live
inside JSON lists that flattening drops:

* `delivery.encode.renderings[]` -- what the master does after AAC and Opus.
  The most actionable "will this survive distribution" numbers in the dump,
  and invisible to every query until lifted out by hand.
* `stems.masking.per_section[]` -- vocal against instrumental, per section.
  The chorus figure is the mix number this whole corpus exists to compare.
* `online.genres.ranked[]` -- every genre vote, not just the winner.  "house"
  is often the third vote on a record a listener would call house.

Paths are read from the joined document: `analysis.json` at the root, with
`online.json` mounted under `online.`.
"""

from __future__ import annotations

from typing import Any, Callable

# --------------------------------------------------------------------------
# path access
# --------------------------------------------------------------------------


def dig(doc: dict, path: str, default: Any = None) -> Any:
    """`a.b.c` into nested dicts; missing or non-dict on the way returns default."""
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------
# derived readers -- values that live inside lists and so survive no flattening
# --------------------------------------------------------------------------


def _rendering(doc: dict, name: str) -> dict:
    for r in dig(doc, "delivery.encode.renderings", []) or []:
        if isinstance(r, dict) and r.get("name") == name:
            return r
    return {}


def enc(name: str, *path: str) -> Callable[[dict], Any]:
    """A field of one encode rendering, e.g. `enc("aac_256", "measured", "true_peak_dbtp_4x")`."""
    def read(doc: dict) -> Any:
        cur: Any = _rendering(doc, name)
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur
    return read


def enc_new_overs(name: str) -> Callable[[dict], Any]:
    """True when the encode pushed the signal over 0 dBTP and the source was not."""
    def read(doc: dict) -> Any:
        r = _rendering(doc, name)
        no = r.get("new_overs") if isinstance(r, dict) else None
        if not isinstance(no, dict):
            return None
        return bool(no.get("0.0"))
    return read


def worst_encode_tp_delta(doc: dict) -> Any:
    """The largest true-peak rise any tested codec inflicts on this master."""
    deltas = []
    for r in dig(doc, "delivery.encode.renderings", []) or []:
        if not isinstance(r, dict):
            continue
        d = (r.get("delta_vs_source") or {}).get("true_peak_dbtp_4x")
        if isinstance(d, (int, float)):
            deltas.append(float(d))
    return max(deltas) if deltas else None


def _sections_with_label(doc: dict, want: str) -> list[int]:
    """Indices of `form.sections` whose part label is `want`."""
    out = []
    for s in dig(doc, "form.sections", []) or []:
        if isinstance(s, dict) and str(s.get("label") or "").lower() == want:
            idx = s.get("index")
            if isinstance(idx, int):
                out.append(idx)
    return out


def vocal_minus_instr(where: str) -> Callable[[dict], Any]:
    """Vocal against instrumental in LU -- whole track, or averaged over choruses.

    `stems.masking.per_section[]` carries it per section and nothing flattens
    that list, so this is the only route to the number.
    """
    def read(doc: dict) -> Any:
        rows = dig(doc, "stems.masking.per_section", []) or []
        vals = []
        keep = set(_sections_with_label(doc, "chorus")) if where == "chorus" else None
        for r in rows:
            if not isinstance(r, dict):
                continue
            if keep is not None and r.get("index") not in keep:
                continue
            v = r.get("vocal_minus_instrumental_lu")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return round(sum(vals) / len(vals), 3) if vals else None
    return read


def masking_index(masker: str, target: str) -> Callable[[dict], Any]:
    """One cell of the whole-track masking matrix, in dB."""
    def read(doc: dict) -> Any:
        for p in dig(doc, "stems.masking.pairs", []) or []:
            if (isinstance(p, dict) and p.get("masker") == masker
                    and p.get("target") == target):
                v = p.get("masking_index_db")
                return float(v) if isinstance(v, (int, float)) else None
        return None
    return read


def genres_all(doc: dict) -> list[str]:
    """Every genre any source voted for -- not only the winner."""
    out, seen = [], set()
    for g in dig(doc, "online.genres.ranked", []) or []:
        name = str((g or {}).get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def descriptive_tags(doc: dict) -> list[str]:
    out = []
    for t in dig(doc, "online.descriptive_tags", []) or []:
        name = str((t or {}).get("name") or "").strip()
        if name:
            out.append(name)
    return out


def credit(role: str) -> Callable[[dict], Any]:
    def read(doc: dict) -> Any:
        people = (dig(doc, "online.credits", {}) or {}).get(role) or []
        names = [str((p or {}).get("name") or "").strip() for p in people]
        return ", ".join(n for n in names if n) or None
    return read


def is_single(doc: dict) -> Any:
    """Album cut or standalone release -- the corpus's natural contrast set.

    Roughly 1,100 of 1,321 tracks here are album cuts by artists whose singles
    are also in the corpus.  Same artist, same era, same producers, same
    mastering chain, very different outcomes: every confound that normally
    wrecks this comparison is held constant by construction, and what varies
    is the song and the mix.
    """
    # Asked of the whole packaging history, not of the one release this copy
    # came from.  `Scar Tissue` was a single and an album cut; reading the type
    # of the chosen release called it neither, because the release chosen was a
    # bootleg compilation.
    issued = dig(doc, "online.musicbrainz.issued_as_single")
    if isinstance(issued, bool):
        return issued
    rt = dig(doc, "online.musicbrainz.release_group.primary_type")
    if isinstance(rt, str) and rt:
        return rt.lower() == "single"
    total = dig(doc, "tags.named.totaltracks")
    try:
        return int(total) <= 3 if total is not None else None
    except (TypeError, ValueError):
        return None


def low_confidence_count(doc: dict) -> Any:
    notes = doc.get("confidence_notes") or []
    return sum(1 for n in notes
               if isinstance(n, dict) and n.get("confidence") == "low")


def coverage_pct(doc: dict) -> Any:
    by = dig(doc, "coverage.by_group", {}) or {}
    present = sum(int(v.get("present") or 0) for v in by.values() if isinstance(v, dict))
    total = sum(int(v.get("features") or 0) for v in by.values() if isinstance(v, dict))
    return round(100.0 * present / total, 2) if total else None


def catalogue_artist(doc: dict) -> str:
    """The artist whose catalogue this track belongs to.

    Deliberately the top-level scan folder, not the artist tag.  `mtx scan`
    mirrors the library, so that folder is exactly one name per artist, while
    the tag carries features, casing drift and separator damage: "Tyler, The
    Creator", "Tyler; The Creator" and "Tyler, The Creator / Daniel Caesar"
    are one catalogue and three strings.  Grouped on the tag this column held
    264 values for 55 artists, and every within-artist comparison built on it
    was comparing the wrong things.

    Falls back to the album artist, then the track artist, when a folder is
    loaded without a root.
    """
    # `tools/identity.py` resolves the folder to the name MusicBrainz uses,
    # because a folder is whatever someone typed and "Red Hot Chilli Peppers"
    # is a join key no other dataset in the world shares.  The grouping is
    # still the folder -- resolution renames the group, it never merges two.
    resolved = dig(doc, "mtx.artist.notion_name")
    if resolved:
        return str(resolved)
    name = dig(doc, "mtx.catalogue_artist")
    if name:
        return str(name)
    for path in ("tags.named.albumartist", "tags.named.artist"):
        raw_name = dig(doc, path)
        if raw_name:
            return str(raw_name).replace(";", ",").split(" / ")[0].strip()
    return ""


def artist_mbid(doc: dict) -> Any:
    """The primary artist's MusicBrainz id -- the machine-readable key.

    A name is a label for people.  An MBID is what a join should actually run
    on: it survives spelling, casing, and the fact that this library has a
    folder called "Red Hot Chilli Peppers".
    """
    for a in dig(doc, "online.musicbrainz.artists", []) or []:
        if isinstance(a, dict) and a.get("mbid"):
            return a["mbid"]
    return None


def all_artists(doc: dict) -> list[str]:
    """Every credited artist, from MusicBrainz rather than from the tag.

    MusicBrainz returns them already separated and already canonical, so this
    is a list of real names instead of one string that happens to contain
    slashes.
    """
    out, seen = [], set()
    for a in dig(doc, "online.musicbrainz.artists", []) or []:
        name = str((a or {}).get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    if not out:
        head = catalogue_artist(doc)
        if head:
            out.append(head)
    return out


def all_artist_mbids(doc: dict) -> list[str]:
    return [a["mbid"] for a in (dig(doc, "online.musicbrainz.artists", []) or [])
            if isinstance(a, dict) and a.get("mbid")]


def release_precision(doc: dict) -> Any:
    """`day` / `month` / `year` -- how much of the release date is real."""
    recorded = dig(doc, "online.cross_checks.release_date.precision")
    if recorded:
        return str(recorded)
    raw = str(dig(doc, "online.cross_checks.release_date.consensus")
              or dig(doc, "online.cross_checks.release_date.earliest") or "").strip()
    if not raw:
        return None
    return {4: "year", 7: "month"}.get(len(raw), "day")


LYRIC_MIN_WORDS = 20
LYRIC_MIN_LINES = 4


def lyric_is_real(doc: dict) -> Any:
    """Whether the lyric block holds a song's words rather than a credit line.

    `lyrics.text_available` alone is not enough on this corpus.  Until
    `lyrics.py` was fixed, any tag key containing "lyric" was accepted, so
    Apple-style `composerlyricist` credits were measured as lyrics: 956 of
    1,321 tracks report a lyric whose modal length is two words.  Analyses
    written before the fix still carry that text, and re-running `mtx scan`
    over the corpus is the only thing that clears it.

    So this applies a shape test the loader records rather than trusting the
    flag: a real lyric runs to at least LYRIC_MIN_WORDS words or
    LYRIC_MIN_LINES lines.  Credit strings are one line of two to sixteen
    words; the real lyrics in this corpus run 100-453 words.
    """
    if not dig(doc, "lyrics.text_available"):
        return False
    words = dig(doc, "lyrics.statistics.words") or 0
    lines = dig(doc, "lyrics.statistics.lines") or 0
    return bool(words >= LYRIC_MIN_WORDS or lines >= LYRIC_MIN_LINES)


# --------------------------------------------------------------------------
# the property table
# --------------------------------------------------------------------------


class Prop:
    __slots__ = ("name", "kind", "source", "unit", "group")

    def __init__(self, name: str, kind: str, source: Any,
                 unit: str | None = None, group: str = ""):
        self.name = name
        self.kind = kind
        self.source = source
        self.unit = unit
        self.group = group

    def read(self, doc: dict) -> Any:
        if callable(self.source):
            return self.source(doc)
        return dig(doc, self.source)


def P(name, kind, source, unit=None):
    return Prop(name, kind, source, unit)


PROPERTIES: list[Prop] = []


def _group(label: str, props: list[Prop]) -> None:
    for p in props:
        p.group = label
        PROPERTIES.append(p)


_group("identity", [
    P("Title", "title", "tags.named.title"),
    # `Artist` is the grouping facet.  Notion rejects commas in select and
    # multi-select option names outright, so "Tyler, The Creator" cannot be
    # stored as itself here -- the comma becomes a semicolon.  The exact name
    # is kept losslessly in `Artist canonical`, and the join key a machine
    # should actually use is `Artist MBID`.
    P("Artist", "select", catalogue_artist),
    # The exact name, commas and all, for anything that is not a Notion select.
    P("Artist canonical", "rich_text",
      lambda d: (dig(d, "mtx.artist.name")
                 or (all_artists(d) or [catalogue_artist(d)])[0] or None)),
    # The catalogue's own MBID, which is stable across the folder.  `Artist
    # MBIDs` below is per-track and holds whoever this track was credited to,
    # so a Silk Sonic cut in the Bruno Mars folder carries both.
    P("Artist MBID", "rich_text",
      lambda d: dig(d, "mtx.artist.mbid") or artist_mbid(d)),
    P("Artists all", "multi_select", all_artists),
    P("Artist MBIDs", "multi_select", all_artist_mbids),
    P("Album", "rich_text", "tags.named.album"),
    # When this *release* came out, by majority of the sources.  Not the
    # earliest any of them offers: one provider returning a wrong year used to
    # redate the record on its own -- `Efecto` was dated 2018 because iTunes
    # said so and the other five said 2022.
    P("Release date", "date", "online.cross_checks.release_date.consensus"),
    # When the *song* first appeared anywhere, which is a different question:
    # a 2015 documentary soundtrack carries recordings from 2003, and the era
    # a record belongs to is the song's, not the package's.
    P("Song first released", "date",
      "online.cross_checks.release_date.song_first_release"),
    P("Date sources agreeing", "number",
      "online.cross_checks.release_date.consensus_votes"),
    # 365 of 1,321 releases are known only to the year, and a Notion date has
    # to be a real day -- so those are stored as 1 January and would otherwise
    # be indistinguishable from a record that actually came out on 1 January.
    P("Release date precision", "select", release_precision),
    P("Year", "number",
      lambda d: _year(dig(d, "online.cross_checks.release_date.song_first_release")
                      or dig(d, "online.cross_checks.release_date.consensus")
                      or dig(d, "tags.named.date"))),
    P("ISRC", "rich_text", "online.identity.isrc"),
    P("Recording MBID", "rich_text", "online.identity.recording_mbid"),
    P("Label", "rich_text", "online.identity.label"),
    P("Duration", "number", "headline.duration_s", "s"),
    P("Track no", "number", "tags.named.tracknumber"),
    P("Is single", "checkbox", is_single),
    P("Producer", "rich_text", credit("producer")),
    P("Mixing engineer", "rich_text", credit("mixer")),
    P("Mastering engineer", "rich_text", credit("mastering")),
    P("Match confidence", "number", "online.match_confidence"),
])

_group("tags", [
    P("Genre", "select", "online.genres.primary"),
    P("Umbrella", "select", "online.genres.umbrella"),
    P("Genres all", "multi_select", genres_all),
    P("Descriptive tags", "multi_select", descriptive_tags),
    P("Genre agreement", "number", "online.genres.agreement"),
    P("Genre sources", "number", "online.genres.source_count"),
])

_group("master", [
    P("LUFS-I", "number", "headline.lufs_i", "LUFS"),
    P("LRA", "number", "headline.lra_lu", "LU"),
    P("True peak", "number", "headline.true_peak_dbtp_16x", "dBTP"),
    P("Sample peak", "number", "headline.sample_peak_dbfs", "dBFS"),
    P("PLR", "number", "headline.plr_db", "dB"),
    P("PSR min", "number", "headline.psr_min_db", "dB"),
    P("PSR min time", "rich_text", "headline.psr_min_time"),
    P("PSR median", "number", "headline.psr_median_db", "dB"),
    P("DR14", "number", "headline.dr14", "DR"),
    P("Crest 10s", "number", "headline.crest_loudest_10s_db", "dB"),
    P("Crest whole", "number", "headline.crest_whole_db", "dB"),
    P("Flat-top samples", "number", "headline.flat_top_sample_count"),
    P("Effective bit depth", "number", "headline.effective_bit_depth", "bits"),
])

_group("spectrum_stereo", [
    P("Tilt", "number", "headline.spectral_tilt_db_per_oct", "dB/oct"),
    P("Tilt R2", "number", "headline.spectral_tilt_r2"),
    P("Sub 20-60", "number", "headline.sub_band_pct", "%"),
    P("Air 12-20k", "number", "headline.air_band_pct", "%"),
    P("HF cutoff", "number", "headline.hf_cutoff_hz", "Hz"),
    P("Side/mid", "number", "headline.side_minus_mid_db", "dB"),
    P("Side/mid <120", "number", "headline.side_minus_mid_below_120hz_db", "dB"),
    P("Mono crossover", "number", "headline.mono_crossover_hz", "Hz"),
    P("Correlation mean", "number", "headline.correlation_mean"),
    P("Correlation min", "number", "headline.correlation_min"),
])

# Everything below sits inside `delivery.encode.renderings[]`, which no flat
# export reaches.  These are the disqualifier metrics: a master that gains
# true peak on encode clips on the listener's converter, whatever its LUFS.
_group("delivery", [
    P("AAC256 true peak", "number", enc("aac_256", "measured", "true_peak_dbtp_4x"), "dBTP"),
    P("AAC256 TP delta", "number", enc("aac_256", "delta_vs_source", "true_peak_dbtp_4x"), "dB"),
    P("AAC256 new overs", "checkbox", enc_new_overs("aac_256")),
    P("AAC256 HF damage", "number", enc("aac_256", "hf_damage", "band_level_delta_db"), "dB"),
    P("Opus128 true peak", "number", enc("opus_128", "measured", "true_peak_dbtp_4x"), "dBTP"),
    P("Opus128 TP delta", "number", enc("opus_128", "delta_vs_source", "true_peak_dbtp_4x"), "dB"),
    P("Opus128 new overs", "checkbox", enc_new_overs("opus_128")),
    P("Worst encode TP delta", "number", worst_encode_tp_delta, "dB"),
    P("Small-speaker energy", "number", "delivery.small_speaker.energy_share_pct", "%"),
    P("Small-speaker loss", "number", "delivery.small_speaker.loudness_delta_lu", "LU"),
    P("Mono fold loss", "number", "delivery.mono_fold.loudness_delta_lu", "LU"),
    P("Mono fold TP delta", "number", "delivery.mono_fold.true_peak_delta_db", "dB"),
])

_group("rhythm", [
    P("BPM", "number", "headline.tempo_bpm"),
    P("BPM confidence", "select", "structure.tempo.confidence"),
    P("Tempo verdict", "select", "online.cross_checks.tempo.verdict"),
    P("Published BPM", "number", "online.cross_checks.tempo.published_bpm"),
    P("Meter", "select", "rhythm.downbeats.time_signature"),
    P("Bars", "number", "headline.bar_count"),
    P("Swing ratio", "number", "headline.swing_ratio"),
    P("Grid deviation", "number", "headline.grid_deviation_std_ms", "ms"),
    P("Syncopation per bar", "number", "rhythm.syncopation.per_bar"),
    P("Kick on-off", "number", "rhythm.beat_position_profile.kick_on_minus_off_beat_db", "dB"),
    P("Snare backbeat", "number",
      "rhythm.beat_position_profile.snare_backbeat_minus_downbeat_db", "dB"),
])

_group("harmony", [
    P("Key", "select", "headline.key"),
    P("Key from chords", "select", "harmony.key_from_chords.key"),
    P("Keys agree", "checkbox", "harmony.key_cross_check.agree"),
    P("Chords", "number", "headline.chord_count"),
    P("Distinct chords", "number", "harmony.vocabulary.distinct_chords"),
    P("Chords per bar", "number", "harmony.harmonic_rhythm.changes_per_bar"),
    P("Diatonic", "number", "headline.diatonic_time_pct", "%"),
    P("Borrowed", "number", "harmony.degrees.borrowed_time_pct", "%"),
    P("Chord entropy", "number", "harmony.vocabulary.entropy_bits", "bits"),
    P("Loop bars", "number", "harmony.loop.loop"),
    P("Modulations", "number", "harmony.modulation.change_count"),
    P("Harmony confidence", "select", "harmony.confidence"),
])

_group("form", [
    P("Form letters", "rich_text", "headline.form_letters"),
    P("Parts", "number", "headline.form_part_count"),
    P("Sections", "number", "headline.section_count"),
    P("Unnamed parts", "number", "headline.form_unnamed_parts"),
    P("Chorus count", "number", "headline.chorus_count"),
    P("Chorus share", "number", "headline.chorus_share_pct", "%"),
    P("To first chorus", "number", "headline.time_to_first_chorus_s", "s"),
    P("To first chorus frac", "number", "form.time_to_first_chorus_fraction"),
    P("Intro length", "number", "form.intro_length_s", "s"),
    P("To vocal entry", "number", "headline.time_to_vocal_entry_s", "s"),
    P("Ending type", "select", "form.ending.type"),
    P("Loopability", "number", "form.loopability.spectral_cosine"),
    P("2nd chorus lift", "number", "form.second_chorus_vs_first.lufs_i", "LU"),
    P("2nd chorus crest", "number", "form.second_chorus_vs_first.crest_db", "dB"),
    P("Beat switches", "number", "form.beat_switch_count"),
])

_group("melody", [
    P("Vocal range", "number", "headline.vocal_range_p5_p95_semitones", "st"),
    P("Vocal low note", "rich_text", "headline.vocal_p5_note"),
    P("Vocal high note", "rich_text", "headline.vocal_p95_note"),
    P("Vocal median note", "rich_text", "headline.vocal_median_note"),
    P("Notes per second", "number", "headline.vocal_notes_per_second"),
    P("Stepwise share", "number", "stems.melody.vocals.intervals.stepwise_share"),
    P("Melisma index", "number", "stems.melody.vocals.melisma_index"),
    P("Vibrato rate", "number", "stems.melody.vocals.vibrato.rate_hz_median", "Hz"),
    P("Self-similarity", "number", "stems.melody.vocals.self_similarity.repeated_ngram_share"),
    P("Delivery", "select", "stems.melody.vocals.delivery.classification"),
    P("Out of scale", "number", "stems.melody.vocals.chromaticism.out_of_scale_time_pct", "%"),
])

# `stems.masking.per_section[]` is a list, so none of the vocal-against-
# instrumental figures reach a flat table.  The chorus one is the single most
# useful mix number in the document.
_group("mix", [
    P("Vocal-instr whole", "number", vocal_minus_instr("whole"), "LU"),
    P("Vocal-instr chorus", "number", vocal_minus_instr("chorus"), "LU"),
    P("Drums mask vocals", "number", masking_index("drums", "vocals"), "dB"),
    P("Other mask vocals", "number", masking_index("other", "vocals"), "dB"),
    P("Bass mask vocals", "number", masking_index("bass", "vocals"), "dB"),
    P("Bass mask drums", "number", masking_index("bass", "drums"), "dB"),
    P("Vocal HP corner", "number", "stems.masking.vocal.high_pass.corner_hz", "Hz"),
    P("Vocal reverb pre-delay", "number", "stems.masking.vocal.reverb.pre_delay_ms", "ms"),
    P("Sibilance median", "number", "stems.masking.vocal.sibilance.ratio_db.median", "dB"),
    P("Sibilance p99", "number", "stems.masking.vocal.sibilance.ratio_db.p99", "dB"),
    P("Concurrent sources", "number", "headline.concurrent_sources_mean"),
])

_group("lyrics", [
    P("Has real lyric", "checkbox", lyric_is_real),
    P("Lyric source", "select", "headline.lyric_source"),
    P("Lyric is inference", "checkbox", "lyrics.is_inference"),
    P("Lyric words", "number", "headline.lyric_word_count"),
    P("Lyric language", "select", "lyrics.language.language"),
    # The writing, which is the half of a record this corpus could say nothing
    # about until transcription worked.  All of it is null on a track with no
    # lyric, which is the correct reading of "not measured".
    P("Lyric lines", "number", "lyrics.statistics.lines"),
    P("Words per line", "number", "lyrics.statistics.words_per_line"),
    P("Syllables per line", "number", "lyrics.statistics.syllables.per_line_mean"),
    P("Unique words", "number", "lyrics.statistics.unique_words"),
    P("Type-token ratio", "number", "lyrics.statistics.type_token_ratio"),
    # How much of the lyric is the same line again.  The single most direct
    # measurement of hook-versus-verse writing in the dump.
    P("Repeated line %", "number", "lyrics.statistics.repeated_line_pct", "%"),
    P("Lyric compression", "number", "lyrics.statistics.compression_ratio"),
    P("Hook phrase", "rich_text",
      lambda d: (dig(d, "lyrics.statistics.most_repeated_ngram", {}) or {}).get("text")),
    P("Hook repeats", "number",
      lambda d: (dig(d, "lyrics.statistics.most_repeated_ngram", {}) or {}).get("occurrences")),
    P("Longest repeated phrase", "rich_text",
      lambda d: (dig(d, "lyrics.statistics.longest_repeated_ngram", {}) or {}).get("text")),
    P("Title sung", "number",
      lambda d: (dig(d, "lyrics.statistics.title_in_lyric", {}) or {}).get("occurrences")),
    P("Title first sung", "number", "lyrics.alignment.title_first_sung_s", "s"),
    P("First word at", "number", "lyrics.alignment.first_word_s", "s"),
    P("Voiced time", "number", "lyrics.alignment.voiced_time_s", "s"),
    # Syllables per second of voicing: a delivery rate, comparable across a
    # cohort in a way no text statistic is.
    P("Delivery rate", "number",
      "lyrics.alignment.delivery_rate.syllables_per_second", "syl/s"),
    P("Syllables per beat", "number",
      "lyrics.alignment.delivery_rate.syllables_per_beat"),
    P("Rhyme density", "number", "lyrics.rhyme.density_per_line", "per line"),
    P("Perfect rhymes", "number", "lyrics.rhyme.perfect"),
    P("Slant rhymes", "number", "lyrics.rhyme.slant"),
    P("Rhyme scheme", "rich_text", "lyrics.rhyme.scheme"),
    P("Readability", "number", "lyrics.statistics.readability_flesch"),
    P("First-person share", "number",
      "lyrics.statistics.pronoun_share.first_singular"),
    P("Second-person share", "number", "lyrics.statistics.pronoun_share.second"),
    P("Explicit terms", "number", "lyrics.statistics.explicit_terms.count"),
])

# Time-varying values never land here -- they go to the Observations log with
# an `observed_at`.  These two are caches of its latest row and carry the
# timestamp that says how old they are.
_group("outcome", [
    # Settled facts: immutable once a record's chart run is over, so they
    # belong on the row.  Filled from declared.json by hand.
    P("Billboard peak", "number", "declared.outcome.billboard_peak"),
    P("Weeks on chart", "number", "declared.outcome.weeks_on_chart"),
    P("Certification", "select", "declared.outcome.certification"),
    # Caches of the newest Observations row, each next to the date that says
    # how old it is.  Never treat one as current without reading that date.
    P("Latest Deezer rank", "number", "online.popularity.deezer_rank"),
    P("Latest playcount", "number", "online.popularity.lastfm_playcount"),
    P("Latest listeners", "number", "online.popularity.lastfm_listeners"),
    P("Last observed", "date", "online.queried_utc"),
    # Derived by tools/notion/outcome.py: position within the same artist's
    # catalogue, which is what holds fame and catalogue age constant.
    P("Playcount z in artist", "number", "outcome.playcount_z_within_artist"),
    P("Percentile in artist", "number", "outcome.percentile_within_artist", "%"),
    P("Percentile in corpus", "number", "outcome.percentile_in_corpus", "%"),
    P("vs artist median", "number", "outcome.playcount_vs_artist_median_db", "dB"),
    P("Outcome tercile", "select", "outcome.outcome_tercile"),
    P("Release type", "select", "outcome.release_type"),
    P("Artist catalogue size", "number", "outcome.artist_track_count"),
    P("Outcome basis", "rich_text", "outcome.reason"),
    # Eight recordings appear twice, as two different masters of one
    # performance.  Counted twice they double-vote in every percentile, so a
    # query wanting a deduplicated corpus filters `Recording primary`.
    P("Recording duplicates", "number", "outcome.recording_duplicates"),
    P("Recording primary", "checkbox", "outcome.recording_primary"),
])

def _pct(key: str):
    """Where this track sits in its cohort, on one metric, as a percentile."""
    return lambda d: dig(d, f"cohort.metrics.{key}.cohort_percentile")


def _cohort_median(key: str):
    return lambda d: dig(d, f"cohort.metrics.{key}.cohort_median")


def references(doc: dict) -> Any:
    """The nearest records in the corpus -- the A/B list, precomputed.

    "Master to -8 LUFS" is advice.  "These five released records are closest to
    yours on every measured axis, and here is where you differ from them" is
    evidence, and it is the form the question actually gets asked in.
    """
    got = dig(doc, "cohort.neighbours", {}) or {}
    rows = got.get("list") or []
    if not rows:
        return None
    # `mtx cohort` groups on the library folder deliberately -- it must not
    # depend on a file `tools/` writes -- so the names it hands back are folder
    # names.  Resolve them here, or the reference list reads "TIESTO".
    names = dig(doc, "mtx.artist_names", {}) or {}
    named = [f"{names.get(r.get('artist'), r.get('artist'))} - {r.get('title')}"
             for r in rows[:6] if r.get("title")]
    return "; ".join(named) or None


_group("cohort", [
    # Written by `mtx cohort` into its own file and mounted here.  An absolute
    # number is not actionable: -9.45 LUFS is a fact, and "the 31st percentile
    # of house records since 2022" is an answer.
    P("Cohort", "select", "cohort.primary_cohort"),
    P("Cohort size", "number", "cohort.primary_cohort_size"),
    # True when the labelled cohort was too small and a broader pool was used,
    # so every percentile on this row is against something less specific than
    # the Cohort column suggests.
    P("Cohort is fallback", "checkbox", "cohort.primary_cohort_is_fallback"),
    P("Cohort genres", "multi_select", lambda d: dig(d, "cohort.genres", []) or []),
    P("Typicality (mean |z|)", "number", "cohort.typicality.mean_abs_z"),
    P("A/B references", "rich_text", references),
    P("References basis", "rich_text", "cohort.neighbours.basis"),

    P("LUFS-I pct", "number", _pct("headline.lufs_i"), "%"),
    P("LUFS-I cohort median", "number", _cohort_median("headline.lufs_i"), "LUFS"),
    P("True peak pct", "number", _pct("headline.true_peak_dbtp_16x"), "%"),
    P("PLR pct", "number", _pct("headline.plr_db"), "%"),
    P("PSR min pct", "number", _pct("headline.psr_min_db"), "%"),
    P("PSR min cohort median", "number", _cohort_median("headline.psr_min_db"), "dB"),
    P("PSR median pct", "number", _pct("headline.psr_median_db"), "%"),
    P("DR14 pct", "number", _pct("headline.dr14"), "%"),
    P("LRA pct", "number", _pct("headline.lra_lu"), "%"),
    P("Crest pct", "number", _pct("headline.crest_loudest_10s_db"), "%"),
    P("Tilt pct", "number", _pct("headline.spectral_tilt_db_per_oct"), "%"),
    P("Tilt cohort median", "number",
      _cohort_median("headline.spectral_tilt_db_per_oct"), "dB/oct"),
    P("Air band pct", "number", _pct("headline.air_band_pct"), "%"),
    P("Sub band pct", "number", _pct("headline.sub_band_pct"), "%"),
    P("Side minus mid pct", "number", _pct("headline.side_minus_mid_db"), "%"),
    P("Correlation pct", "number", _pct("headline.correlation_mean"), "%"),
    P("HF cutoff pct", "number", _pct("headline.hf_cutoff_hz"), "%"),
    P("Tempo pct", "number", _pct("headline.tempo_bpm"), "%"),
    P("Duration pct", "number", _pct("headline.duration_s"), "%"),
    P("Section count pct", "number", _pct("headline.section_count"), "%"),
])

_group("provenance", [
    P("sha256", "rich_text", "file.sha256"),
    P("mtx run", "rich_text", lambda d: (
        f"mtx {dig(d, 'run.tool_version')} / schema {dig(d, 'run.schema_version')} "
        f"/ {dig(d, 'run.profile')}")),
    P("Coverage", "number", coverage_pct, "%"),
    P("Low-confidence", "number", low_confidence_count),
    P("Warnings", "number", lambda d: len(d.get("warnings") or [])),
    P("Analysis path", "rich_text", "mtx.analysis_path"),
    P("Traits", "multi_select", lambda d: _trait_names(d)),
])


def _year(value: Any) -> Any:
    if not value:
        return None
    text = str(value)[:4]
    return int(text) if text.isdigit() else None


# --------------------------------------------------------------------------
# traits -- tri-state, never boolean
# --------------------------------------------------------------------------
#
# A boolean silently turns "not measured" into "no".  `four_on_the_floor` is
# null on 63 of 72 electronic tracks tested, so a boolean trait would drop 87%
# of the corpus out of a club-music query with no error anywhere.  Each trait
# therefore reports `yes` / `no` / omits itself, and the threshold that decided
# it is recorded here and stamped on every row via "Trait thresholds".

TRAIT_VERSION = "1.0.0"

TRAITS: list[tuple[str, Callable[[dict], Any], str]] = [
    ("four-on-the-floor",
     lambda d: dig(d, "rhythm.beat_position_profile.inference.four_on_the_floor"),
     "rhythm.beat_position_profile.inference.four_on_the_floor; null unless a "
     "kick pattern was detected at all"),
    ("backbeat",
     lambda d: dig(d, "rhythm.beat_position_profile.inference.backbeat_on_2_and_4"),
     "rhythm.beat_position_profile.inference.backbeat_on_2_and_4"),
    ("swung",
     lambda d: _gt(dig(d, "headline.swing_ratio"), 1.25),
     "headline.swing_ratio > 1.25 (1.00 straight, 2.00 triplet shuffle)"),
    ("loop-based",
     lambda d: _notnull(dig(d, "harmony.loop.loop")),
     "harmony.loop.loop is not null -- a repeating chord period was found"),
    ("modulates",
     lambda d: _gt(dig(d, "harmony.modulation.change_count"), 0),
     "harmony.modulation.change_count > 0; low confidence by construction"),
    ("cold-stop",
     lambda d: _eq(dig(d, "form.ending.type"), "hard cut"),
     "form.ending.type == 'hard cut'"),
    ("fade-out",
     lambda d: _eq(dig(d, "form.ending.type"), "fade"),
     "form.ending.type == 'fade'"),
    ("instrumental",
     lambda d: _falsy(dig(d, "form.vocal_presence.available")) or
               _eq_count(dig(d, "form.vocal_presence.present"), True, 0),
     "no section has vocal-stem RMS within 12 dB of the track's vocal reference"),
    ("late-chorus",
     lambda d: _gt(dig(d, "form.time_to_first_chorus_fraction"), 0.35),
     "form.time_to_first_chorus_fraction > 0.35 of duration"),
    ("sub-heavy",
     lambda d: _gt(dig(d, "headline.sub_band_pct"), 40.0),
     "headline.sub_band_pct > 40% of total energy"),
    ("heavily-limited",
     lambda d: _lt(dig(d, "headline.psr_min_db"), 5.0),
     "headline.psr_min_db < 5 dB"),
    ("mono-bass",
     lambda d: _lt(dig(d, "headline.side_minus_mid_below_120hz_db"), -20.0),
     "headline.side_minus_mid_below_120hz_db < -20 dB"),
    ("encode-overs",
     lambda d: _or_none(enc_new_overs("aac_256")(d), enc_new_overs("opus_128")(d)),
     "AAC 256 or Opus 128 decode goes above 0 dBTP where the source did not"),
]


def _notnull(v):  return None if v is None else v is not None
def _gt(v, t):    return None if not isinstance(v, (int, float)) else bool(v > t)
def _lt(v, t):    return None if not isinstance(v, (int, float)) else bool(v < t)
def _eq(v, t):    return None if v is None else bool(str(v).lower() == t)
def _falsy(v):    return None if v is None else (True if v is False else None)


def _eq_count(seq, value, want):
    if not isinstance(seq, list):
        return None
    return bool(sum(1 for x in seq if x == value) == want)


def _or_none(*vals):
    if all(v is None for v in vals):
        return None
    return bool(any(v for v in vals if v is not None))


def trait_states(doc: dict) -> dict[str, str]:
    """`{trait: 'yes' | 'no' | 'not-measured'}` for every trait."""
    out = {}
    for name, fn, _rule in TRAITS:
        try:
            v = fn(doc)
        except Exception:
            v = None
        out[name] = "not-measured" if v is None else ("yes" if v else "no")
    return out


def _trait_names(doc: dict) -> list[str]:
    """Multi-select values: `trait` when yes, `no-<trait>` when measured false.

    A trait that could not be measured contributes nothing, so "absent" and
    "measured absent" stay distinguishable in a query -- which is the whole
    point of not using a checkbox.
    """
    out = []
    for name, state in trait_states(doc).items():
        if state == "yes":
            out.append(name)
        elif state == "no":
            out.append(f"no-{name}")
    return out


def trait_documentation() -> str:
    lines = [f"trait threshold set {TRAIT_VERSION}", ""]
    for name, _fn, rule in TRAITS:
        lines.append(f"{name:20} {rule}")
    return "\n".join(lines)


PROPERTY_NAMES = [p.name for p in PROPERTIES]
assert len(PROPERTY_NAMES) == len(set(PROPERTY_NAMES)), "duplicate property name"
