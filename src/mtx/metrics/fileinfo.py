"""4.1 File, container, provenance."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from typing import Any

import numpy as np
import soundfile as sf

from ..audio import AudioSource, READ_BLOCK_FRAMES
from ..util import Collector

# Tag keys pulled out by name; everything else still lands in `tags.all`.
TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "TIT2", "\xa9nam"),
    "artist": ("artist", "TPE1", "\xa9ART"),
    "album": ("album", "TALB", "\xa9alb"),
    "albumartist": ("albumartist", "album artist", "TPE2", "aART"),
    "date": ("date", "year", "originaldate", "TDRC", "TYER", "\xa9day"),
    "genre": ("genre", "TCON", "\xa9gen"),
    "label": ("label", "organization", "publisher", "TPUB"),
    "catalognumber": ("catalognumber", "catalog", "catalog #", "CATALOGNUMBER"),
    "isrc": ("isrc", "TSRC"),
    "barcode": ("barcode", "upc", "ean/upn", "BARCODE"),
    "composer": ("composer", "TCOM", "\xa9wrt"),
    "comment": ("comment", "description", "COMM", "\xa9cmt"),
    "encoder": ("encoder", "encoded-by", "encodedby", "TSSE", "\xa9too"),
    "encoder_settings": ("encoder settings", "encodersettings", "encoding"),
    "tracknumber": ("tracknumber", "TRCK", "trkn"),
    "discnumber": ("discnumber", "TPOS", "disk"),
}
MB_PREFIXES = ("musicbrainz", "acoustid")
RG_PREFIXES = ("replaygain",)
# Markers distributors write when a file went through the Apple Digital Master
# (formerly "Mastered for iTunes") pipeline.
ADM_MARKERS = ("mastered for itunes", "apple digital master", "itunnorm", "itunsmpb")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def decoded_md5(path: str, subtype: str, n_ch: int) -> tuple[str | None, str | None]:
    """MD5 of the decoded PCM, in the byte layout FLAC's STREAMINFO uses.

    FLAC hashes the interleaved little-endian samples at the file's own bit
    depth.  Returns (hex_digest, reason_if_unavailable).
    """
    width = {"PCM_S8": 1, "PCM_U8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4}.get(subtype)
    if width is None:
        return None, f"subtype {subtype} is not integer PCM; FLAC-style decoded MD5 undefined"
    h = hashlib.md5()
    with sf.SoundFile(path) as f:
        while True:
            block = f.read(frames=READ_BLOCK_FRAMES, dtype="int32", always_2d=True)
            if block.shape[0] == 0:
                break
            flat = block.reshape(-1)
            if width == 4:
                h.update(flat.astype("<i4").tobytes())
            elif width == 2:
                h.update((flat >> 16).astype("<i2").tobytes())
            elif width == 3:
                v = (flat >> 8).astype("<i4")
                h.update(v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())
            else:  # 8-bit: FLAC stores signed, WAV stores unsigned
                h.update((flat >> 24).astype("<i1").tobytes())
    return h.hexdigest(), None


def flac_streaminfo(path: str) -> dict[str, Any] | None:
    """Parse the FLAC STREAMINFO block directly, for the stored MD5."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"fLaC":
                return None
            header = f.read(4)
            if len(header) < 4:
                return None
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            if block_type != 0 or length < 34:
                return None
            data = f.read(34)
    except OSError:
        return None
    min_bs, max_bs = struct.unpack(">HH", data[0:4])
    min_fs = int.from_bytes(data[4:7], "big")
    max_fs = int.from_bytes(data[7:10], "big")
    packed = int.from_bytes(data[10:18], "big")
    sr = (packed >> 44) & 0xFFFFF
    ch = ((packed >> 41) & 0x7) + 1
    bps = ((packed >> 36) & 0x1F) + 1
    total = packed & 0xFFFFFFFFF
    return {
        "min_block_size": min_bs, "max_block_size": max_bs,
        "min_frame_size": min_fs, "max_frame_size": max_fs,
        "sample_rate_hz": sr, "channels": ch, "bits_per_sample": bps,
        "total_samples": total, "md5_stored": data[18:34].hex(),
    }


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Width/height from a PNG or JPEG byte string, without an image library."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h)
            i += 2 + seg_len
    return None, None


def read_tags(path: str, collector: Collector) -> dict[str, Any]:
    out: dict[str, Any] = {
        "all": {}, "named": {}, "musicbrainz": {}, "replaygain": {},
        "apple_digital_master_markers": [], "vendor_string": None,
        "cover_art": {"present": False, "count": 0, "width_px": None,
                      "height_px": None, "mime": None, "bytes": None},
    }
    try:
        import mutagen
    except ImportError:
        collector.warn("tags", "mutagen unavailable; no embedded tags read")
        return out
    try:
        mf = mutagen.File(path)
    except Exception as exc:  # corrupt tags must not stop the analysis
        collector.warn("tags", f"mutagen could not parse tags: {exc!r}")
        return out
    if mf is None:
        collector.warn("tags", "mutagen recognised no tag container in this file")
        return out

    flat: dict[str, str] = {}
    try:
        items = list(mf.tags.items()) if mf.tags is not None else []
    except Exception as exc:
        collector.warn("tags", f"tag enumeration failed: {exc!r}")
        items = []
    for key, val in items:
        k = str(key)
        if isinstance(val, list):
            parts = []
            for v in val:
                try:
                    parts.append(str(v))
                except Exception:
                    parts.append("<unreadable>")
            sval = " / ".join(parts)
        else:
            try:
                sval = str(val)
            except Exception:
                sval = "<unreadable>"
        if len(sval) > 2000:
            sval = sval[:2000] + "...<truncated>"
        flat[k] = sval
    out["all"] = dict(sorted(flat.items()))

    lower = {k.lower(): v for k, v in flat.items()}
    for name, aliases in TAG_ALIASES.items():
        for a in aliases:
            if a.lower() in lower:
                out["named"][name] = lower[a.lower()]
                break
        out["named"].setdefault(name, None)
    for k, v in lower.items():
        if any(k.startswith(p) for p in MB_PREFIXES):
            out["musicbrainz"][k] = v
        if any(k.startswith(p) for p in RG_PREFIXES):
            out["replaygain"][k] = v
    hay = " ".join(f"{k} {v}" for k, v in lower.items()).lower()
    out["apple_digital_master_markers"] = [m for m in ADM_MARKERS if m in hay]

    try:
        out["vendor_string"] = getattr(mf.tags, "vendor", None)
    except Exception:
        out["vendor_string"] = None

    pics = []
    if hasattr(mf, "pictures") and mf.pictures:
        pics = list(mf.pictures)
    elif mf.tags is not None and hasattr(mf.tags, "getall"):
        try:
            pics = list(mf.tags.getall("APIC"))
        except Exception:
            pics = []
    if pics:
        p = pics[0]
        data = getattr(p, "data", b"") or b""
        w = getattr(p, "width", None) or None
        h = getattr(p, "height", None) or None
        if not w or not h:
            w, h = _image_dimensions(data)
        out["cover_art"] = {
            "present": True, "count": len(pics),
            "width_px": w, "height_px": h,
            "mime": getattr(p, "mime", None),
            "bytes": len(data) or None,
        }
    return out


def run_ffprobe(path: str, collector: Collector) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        collector.warn("container", f"ffprobe unavailable or timed out ({exc.__class__.__name__}); "
                                    "container.ffprobe_raw is null")
        return None
    if proc.returncode != 0:
        collector.warn("container", f"ffprobe exited {proc.returncode}; ffprobe_raw is null")
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        collector.warn("container", f"ffprobe output was not JSON: {exc}")
        return None


def analyse(src: AudioSource, collector: Collector) -> dict[str, Any]:
    path = os.path.abspath(src.path)
    st = os.stat(path)
    md5, md5_reason = decoded_md5(path, src.subtype, src.n_ch)
    si = flac_streaminfo(path)

    md5_match: bool | None = None
    if si and md5:
        stored = si["md5_stored"]
        if stored == "0" * 32:
            collector.warn("container", "FLAC STREAMINFO MD5 is all zeros (encoder did not store one)")
        else:
            md5_match = stored == md5
            if not md5_match:
                collector.warn(
                    "container",
                    f"decoded MD5 {md5} does not match FLAC STREAMINFO MD5 {stored}: "
                    "the file may be damaged or was re-muxed without re-hashing",
                )
    if md5_reason:
        collector.warn("container", md5_reason)

    tags = read_tags(path, collector)
    ffprobe = run_ffprobe(path, collector)

    # Compression level is not stored in FLAC; only an encoder string can hint.
    enc = (tags["named"].get("encoder") or "") + " " + (tags["vendor_string"] or "")
    level = None
    for lv in range(9):
        if f"-{lv}" in enc:
            level = lv
            break

    return {
        "file": {
            "path_absolute": path,
            "filename": os.path.basename(path),
            "size_bytes": int(st.st_size),
            "sha256": sha256_file(path),
            "decoded_md5": md5,
            "decoded_md5_unavailable_reason": md5_reason,
            "flac_streaminfo_md5": si["md5_stored"] if si else None,
            "flac_md5_verified": md5_match,
        },
        "container": {
            "format": src.format,
            "subtype": src.subtype,
            "endian": src.endian,
            "sample_rate_hz": src.sr,
            "bit_depth_container": {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16,
                                    "PCM_24": 24, "PCM_32": 32,
                                    "FLOAT": 32, "DOUBLE": 64}.get(src.subtype),
            "channels": src.n_ch,
            "frames": src.n_frames,
            "duration_s": round(src.duration, 3),
            "flac_streaminfo": si,
            "flac_compression_level_inferred": level,
            "flac_compression_level_note":
                "FLAC does not store the compression level; only an encoder "
                "string can hint at it",
            "encoder_string": tags["named"].get("encoder"),
            "vendor_string": tags["vendor_string"],
            "ffprobe_raw": ffprobe,
        },
        "tags": tags,
    }
