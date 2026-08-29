# mtx prediction sheet

file: sample.flac
title: (no title tag)
artist: (no artist tag)
sha256: 0227c4847d647f68...
tool: mtx 0.2.0 / schema 1.1.0 / profile full
audio: 44100 Hz, 2 ch, PCM_24, 75.5 s

The digest for this file has been written and not printed. Fill in the
fields you want to commit to -- a value, a +/- range, and how confident
you are that the true value falls inside that range -- then:

    mtx predict --check <this file> <the digest.md or analysis.json>

Leave a field as ____ to skip it. An interval, not a point guess.

## PREDICTIONS

```
LUFS-I               = ____ LUFS  +/- ____   conf ____%
LRA                  = ____ LU  +/- ____   conf ____%
True peak (16x)      = ____ dBTP  +/- ____   conf ____%
Sample peak          = ____ dBFS  +/- ____   conf ____%
PLR                  = ____ dB  +/- ____   conf ____%
PSR min              = ____ dB  +/- ____   conf ____%
PSR median           = ____ dB  +/- ____   conf ____%
DR14                 = ____  +/- ____   conf ____%
Crest (whole)        = ____ dB  +/- ____   conf ____%
Crest (loudest 10 s) = ____ dB  +/- ____   conf ____%
Spectral tilt        = ____ dB/oct  +/- ____   conf ____%
Air 12-20k           = ____ %  +/- ____   conf ____%
Sub 20-60            = ____ %  +/- ____   conf ____%
Side/mid overall     = ____ dB  +/- ____   conf ____%
Side/mid <120 Hz     = ____ dB  +/- ____   conf ____%
Mono crossover       = ____ Hz  +/- ____   conf ____%
Correlation mean     = ____  +/- ____   conf ____%
Correlation min      = ____  +/- ____   conf ____%
HF cutoff            = ____ Hz  +/- ____   conf ____%
Tempo                = ____ BPM  +/- ____   conf ____%
```

## FLAGS

- [high] forensics.hf_cutoff: spectrum runs to the Nyquist frequency; no shelf to report
- [medium] forensics.mains_hum: musical content at 50/60 Hz and its harmonics raises the same measure as mains hum
- [low] forensics.elliptical_eq: a mono bass end is also a routine modern mixing choice, not only a vinyl-era elliptical EQ
- [medium] forensics.tape_bias: narrowband energy above 15 kHz also comes from switch-mode supplies, CRT whine and synth content
- [medium] dynamics.clip_then_normalise: a flat run below full scale is also produced by a clipper set below 0 dBFS, and by any signal that genuinely holds its maximum for several samples
- [low] processing.saturation: regression R2 is 0.04
- [medium] processing.pumping: envelope cross-correlation reflects arrangement as well as gain reduction
- [medium] processing.reverb: 3130 usable decays across all bands; on dense material the decay after an onset is masked by the next one, so T20/T30 are upper bounds at best
- [unverified] DR14: not validated against a published DR rating; synthetic checks only (run mtx validate-dr once, see METHOD)

## METHOD

- LUFS-I/LRA: ITU-R BS.1770-4 K-weighting, 400 ms blocks 75% overlap, gates -70 LUFS / -10 LU; LRA on 3 s blocks, gates -70 / -20 LU, P95-P10. Cross-checked against ffmpeg ebur128 (tolerance 0.2 LU).
- True peak: scipy.signal.resample_poly (Kaiser beta 5.0) at 4x and 16x, both reported; overs counted as contiguous excursions at 16x.
- PSR: 3 s windows, 1 s hop; short-term true peak (4x) minus short-term LUFS over the same window.
- DR14: TT offline DR. 3 s blocks, block RMS sqrt(2*mean(x^2)), loudest 20%, second-highest per-block sample peak. NOT validated against a published DR rating (see loudness.dr14.validation).
- Flat-top: threshold derived per channel as max(|x|)*0.99999, never a fixed -0.1 dBFS. Run lengths, ms, and the ten longest runs reported.
- Ceiling density: fraction of samples within N dB of that channel's own ceiling; threshold-free.
- Limiter vs clipper: mean dB slope of |x| over the 2 ms before entry and after exit of each flat-top run. Inferred, not measured.
- LTAS: Welch, Hann, 50% overlap, nperseg 16384 broadband; nperseg 131072 over an auto-selected ~90 s body section for the low end.
- Tilt: least-squares dB/octave over 100 Hz-10 kHz on mid, with R2; piecewise slopes over 30-120, 120-1k, 1k-6k, 6k-20k.
- Mid/side: mid=(L+R)/2, side=(L-R)/2. Side/mid is 10*log10(P_side/P_mid). Mono crossover is the highest third-octave centre below which side/mid stays under -20 dB.
- Mono-sum damage: 10*log10(P_mid/(P_mid+P_side)) per third-octave.
- HF cutoff: highest frequency of the 1/6-octave-smoothed LTAS still within 25 dB of the 1-5 kHz median; per-5 s frames for stability.
- Effective bit depth: 32 minus the trailing zero bits of the left-justified int32 sample, maximum over non-zero samples.
- Sections: MFCC+chroma+RMS+spectral contrast, cosine SSM, Foote novelty with an 8 s kernel, peak-picked, segments under 4 s merged.
- Tempo/key: librosa.beat.beat_track; mean chroma-CQT against Krumhansl-Schmuckler profiles. Both carry a confidence.
- Saturation proxy: least-squares regression of 5-10 kHz frame level on broadband frame level, 50 ms frames, in dB/dB.
- Pumping: cross-correlation of the sub-120 Hz and 500 Hz-6 kHz dB envelopes (5 ms hop) over -200..+200 ms.
- Modulation: FFT of each band's 5 ms RMS envelope; depth at the beat rate relative to the envelope RMS.
- Reverb: Schroeder reverse integration after strong onsets, per octave band. Estimate, usually low or medium confidence.
