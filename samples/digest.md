# mtx digest

file: sample.flac
title: (no title tag)
artist: (no artist tag)
sha256: 0227c4847d647f68...
tool: mtx 0.2.0 / schema 1.1.0 / profile full
audio: 44100 Hz, 2 ch, PCM_24, 75.5 s

## HEADLINE

```
LUFS-I                -10.60 LUFS
LRA                   2.5 LU
True peak (16x)       -0.24 dBTP
True peak (4x)        -0.24 dBTP
Sample peak           -1.70 dBFS
TP-SP delta           1.46 dB
PLR                   10.4 dB
PSR min               8.0 dB @ 0:49.000
PSR median            10.3 dB
DR14                  6
Crest (whole)         10.8 dB
Crest (loudest 10 s)  9.5 dB
Spectral tilt         -3.30 dB/oct (R2 0.54)
Air 12-20k            0.08 % of energy
Sub 20-60             11.38 % of energy
Side/mid overall      -8.3 dB
Side/mid <120 Hz      -39.8 dB
Mono crossover        125.0 Hz
Correlation mean      0.74
Correlation min       0.53
Flat-top samples      43527
Flat-top longest      2.93 ms
HF cutoff             n/a Hz
Effective bit depth   24 bits
Tempo                 120.00 BPM
Key                   B minor
Sections              4
Duration              75.500 s
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

## DETAIL

### Band energy (mid / side)

```
band      Hz           mid%   mid dB  side%  side dB
--------  -----------  -----  ------  -----  -------
sub       20.0-60.0    11.32  -21.9   0.01   -60.5
bass      60.0-120.0   28.05  -18.0   0.03   -56.2
low_bass  120.0-250.0  13.98  -21.0   48.21  -23.9
low_mid   250.0-500.0  7.38   -23.8   48.45  -23.9
mid       500.0-2000   20.77  -19.3   1.87   -38.0
high_mid  2000-6000    0.05   -45.5   0.34   -45.4
presence  6000-12000   0.06   -44.4   0.45   -44.3
air       12000-20000  0.08   -43.7   0.51   -43.7
```

### Per-band crest (mono)

```
band      crest dB  rms dBFS  peak dBFS
--------  --------  --------  ---------
sub       12.7      -21.9     -9.2
bass      12.3      -18.1     -5.9
low_bass  13.7      -22.3     -8.6
low_mid   12.0      -23.8     -11.7
mid       9.9       -20.1     -10.2
high_mid  30.8      -46.0     -15.2
presence  31.9      -44.8     -12.9
air       31.8      -44.2     -12.3
spread across bands: 22.0 dB
```

### Flat-top / ceiling forensics

```
ch  thr dBFS  samples  events  runs 1/2/3-5/6-10/11-20/21+  longest ms
--  --------  -------  ------  ---------------------------  ----------
0   -1.700    20762    1406    540/214/209/104/98/241       2.83
1   -1.700    22765    1425    457/212/197/103/159/297      2.93
flat value -1.700 dBFS, below full scale: True, clip-then-normalise: True
entry/exit slope median 5.01 / 4.95 dB/ms over 2000 events -> vertical entry/exit dominates (clipper-like) (inferred)
sub-120 Hz at events vs track: 4.7 dB, correlation 0.20
```

### Source forensics

```
HF cutoff            n/a Hz, slope n/a dB/oct, collapse n/a dB, frames below floor 0.0%
Nearest codec shelf  n/a
Cutoff stability     mean n/a Hz, std n/a Hz over 15 frames
Upsampling           file is at or below 48 kHz; no upsampling check applies
Effective bit depth  24 of 24 container bits
Noise floor          -53.6 dBFS, slope above 10k -5.9 dB/oct
Silence              lead 263.4 ms (fade, fade 500 ms), tail 291.8 ms (fade, fade 590 ms)
Spectral holes       212.6 Hz -9.3 dB, 78.1 Hz -7.6 dB, 452.2 Hz -5.4 dB, 175.0 Hz -4.7 dB
Hum 50hz             max excess 9.4 dB, mean 0.8 dB
Hum 60hz             max excess 11.8 dB, mean 5.9 dB
Rumble <30 Hz        -19.1 dB rel total
Tape bias >15k       none
Wow/flutter          std 1.7 cents, detrended 1.7, drift 0.22 cents/min
```

### Sections

```
#  start     dur   LUFS   ST max  crest  tilt   side/mid  ons/s  d-track
-  --------  ----  -----  ------  -----  -----  --------  -----  -------
0  0:00.000  25.2  -11.8  -11.7   11.8   -2.74  -7.3      4.33   -1.2
1  0:25.170  30.1  -9.3   -9.3    9.5    -3.75  -9.3      4.22   1.3
2  0:55.263  16.2  -11.8  -11.8   11.5   -3.07  -7.4      4.76   -1.2
3  1:11.425  4.1   -14.5  -15.1   16.3   -2.86  -6.8      3.44   -3.9
loudest section 1, quietest 3, widest 3 (widest band there: air at 0.7 dB)
biggest jump 2.7 dB at 1:11.425
```

### Short-term loudness, 5 s grid (LUFS)

```
short-term percentiles LUFS  P10 -11.8  P25 -11.8  P50 -11.7  P75 -9.3  P90 -9.3  P95 -9.3  max -9.3
momentary  percentiles LUFS  P10 -12.6  P25 -12.2  P50 -11.3  P75 -9.4  P90 -9.1  P95 -9.1  max -9.0

 -14.5  -11.7  -11.7  -11.7  -11.7   -9.4   -9.3   -9.3   -9.3   -9.3   -9.3  -11.5
 -11.8  -11.8  -12.2
```

### PSR, 5 s grid (dB)

```
min 8.0 @ 0:49.000   P10 8.2   median 10.3   max 15.8

  13.5   11.0   10.7   10.7   10.5    8.6    8.2    8.1    8.8    9.0    8.2   10.1
  10.3   10.5   10.7
```

### Stereo detail

```
Correlation               mean 0.74, min 0.53, P5 0.58, median 0.74, <0 0.0%, <0.3 0.0%
Most negative windows     0:00.000 0.53, 1:07.000 0.56, 1:12.000 0.56
Channel balance           L-R rms -0.34 dB, L-R LUFS -0.46
Inter-channel offset      0 samples (0.0 us), corr at lag 0.784
Energy outside +/-45 deg  9.1 %
```

### Side/mid per third-octave (Hz:dB)

```
20.0:-28.9  25.0:-30.3  31.5:-31.1  40.0:-31.7  50.0:-36.5
63.0:-48.7  80.0:-29.2  100.0:-32.3  125.0:-33.7  160.0:-17.2
200.0:-0.1  250.0:-0.1  315.0:-0.1  400.0:-0.0  500.0:-8.0
630.0:-19.1  800.0:-5.4  1000:-0.9  1250:-1.2  1600:-0.5
2000:-0.4  2500:0.2  3150:0.1  4000:-0.0  5000:0.1
6300:0.2  8000:0.1  10000:0.2  12500:0.1  16000:-0.0
20000:0.0

mono-sum loss, broadband: -0.60 dB
mono-sum loss, worst bands: 10000 Hz -3.1 dB, 2500 Hz -3.1 dB, 6300 Hz -3.1 dB, 3150 Hz -3.1 dB, 5000 Hz -3.1 dB, 8000 Hz -3.1 dB
```

### Band-envelope correlation (10 ms envelopes, dB domain)

```
Off-diagonal r    min 0.11  median 0.35  max 1.00  mean 0.49  over 28 band pairs
Least correlated  low_mid/presence 0.11; low_mid/air 0.11
Most correlated   presence/air 1.00
```

### Processing forensics (all inferred)

```
Saturation slope                0.826 dB/dB (R2 0.04, 1073 frames) [low]
Pumping                         most negative corr 0.29 at -200 ms, dip -4.1 dB, release 55 ms [medium]
Perc/harm                       -8.6 dB, percussive fraction 0.121
T20 by octave (Hz:s)            63.0:0.84, 125.0:0.90, 250.0:0.75, 500.0:0.68, 1000:0.80, 2000:0.07, 4000:0.06, 8000:0.06
Tail L/R corr                   -0.00 [medium]
Mod depth beat/half/quarter dB  sub:-8.7/-14.3/-18.9  bass:-8.6/-15.5/-22.2  low_bass:-19.0/-22.9/-26.2  low_mid:-26.1/-30.0/-34.7  mid:-23.6/-28.1/-33.5  high_mid:-22.4/-11.4/-12.1  presence:-20.9/-11.1/-11.7  air:-20.6/-11.0/-11.5
```

### Per-band level over time (dB, 10 s grid; 100 ms series is in analysis.json)

```
sub       -22  -22  -22  -22  -22  -22  -22  -24
bass      -19  -18  -18  -18  -18  -18  -18  -21
low_bass  -23  -22  -22  -22  -22  -22  -22  -25
low_mid   -24  -24  -23  -23  -24  -24  -24  -26
mid       -48  -47  -19  -16  -16  -19  -47  -49
high_mid  -47  -46  -46  -46  -46  -45  -45  -47
presence  -45  -44  -44  -44  -44  -45  -46  -49
air       -44  -43  -43  -43  -43  -45  -51  -53
```

### Streaming normalisation preview

```
platform                      target  gain dB  TP after dBTP  turned up
----------------------------  ------  -------  -------------  ---------
spotify_youtube_tidal_amazon  -14     -3.40    -3.64          no
apple                         -16     -5.40    -5.64          no
```

### Bass fundamentals (131072-point Welch)

```
#  Hz     dB rel  note  cents  Q
-  -----  ------  ----  -----  -----
1  61.9   0.0     B1    4.8    61.3
2  59.9   -5.6    A#1   47.4   59.3
3  63.9   -6.4    C2    -39.6  95.0
4  124.2  -9.8    B2    9.5    123.0
5  196.2  -10.6   G3    1.4    145.8
6  57.9   -11.2   A#1   -11.9  57.3
resolution 0.336 Hz, section 0:00.000-1:15.500, single-note low end: False
```

### Ceiling density (% of samples within N dB of the ceiling)

```
ch  0.1    0.5    1      3      6
--  -----  -----  -----  -----  ------
0   0.656  0.801  1.071  3.241  10.042
1   0.717  0.892  1.183  3.525  11.471
```

_Dropped to stay under the 12 KB digest budget (full detail is in analysis.json; --sections or --digest-budget keeps a block that a session needs): arrangement gaps, resonances, spectral descriptors._

## CORPUS ROW

Empty fields are left empty on purpose: nothing here is guessed.

```
Title: 
Artist: 
Year: 
Genre: 
Engineers: 
LUFS-I: -10.60
True peak: -0.24 dBTP
LRA: 2.5 LU
PLR: 10.4 dB
PSR min: 8.0 dB
PSR median: 10.3 dB
DR14: 6
Crest (loudest 10s): 9.5 dB
Tonal tilt notes: tilt -3.30 dB/oct (R2 0.54); 30.0-120.0 -1.83 dB/oct; 120.0-1000 -10.51 dB/oct; 1000-6000 -0.40 dB/oct; 6000-20000 -0.73 dB/oct; air 0.08%, sub 11.38%
Width/mono notes: side/mid -8.3 dB overall, -39.8 dB below 120 Hz; mono crossover 125.0 Hz; correlation mean 0.74, min 0.53
mtx run: mtx 0.2.0 / schema 1.1.0 / profile full / sha256 0227c4847d647f68
```

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
