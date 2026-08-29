# Sample output

`digest.md`, `corpus_row.json` and `predict.md` here are byte-for-byte copies of
what `mtx analyze` wrote, so you can see the output format before running the
tool on real music.

Reproduce them with:

```bash
python tools/make_test_signal.py sample.flac
mtx analyze sample.flac --out .
mtx analyze sample.flac --out . --blind      # adds predict.md
```

The input is the synthetic track from [`tools/make_test_signal.py`](../tools/make_test_signal.py):
a 75-second 44.1 kHz / 24-bit stereo piece built to contain things the tool is
supposed to notice, and nothing copyrighted. It has

- a 62 Hz bass fundamental, summed to mono below ~125 Hz;
- a hard ceiling at -1.0 dBFS with the whole file then turned down 0.7 dB,
  which is the clip-then-normalise case a fixed -0.1 dBFS threshold misses;
- a beat-synchronous duck at 120 BPM;
- a high shelf pulled down in the last section, so the sectional tilt moves;
- fades at both ends and a quarter-second of digital black around them.
