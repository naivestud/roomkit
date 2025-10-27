# Usage

Install from the repository root:

```bash
pip install .              # core (numpy/scipy/soundfile)
pip install ".[neural]"    # adds torch and roomkit.neural
pip install -e ".[test]"   # development install
```

## Denoise a recording

The estimator assumes the recording starts with a short noise-only lead-in
(a few tens of milliseconds is enough):

```python
import roomkit

noisy, rate = roomkit.load_audio("noisy.wav")
enhanced = roomkit.denoise(noisy)            # method="wiener" (default)
roomkit.save_audio("enhanced.wav", enhanced, rate)
```

`method="spectral"` switches to spectral subtraction; both accept
`noise_frames` to control how many leading frames seed the noise estimate.

## Dereverberate

```python
from roomkit.dereverb import dereverb

dry = dereverb(reverberant, taps=10, delay=3, iterations=5)
```

`delay` guards the direct sound (in frames, at 10 ms per frame with the
default hop); `taps` controls how much late tail is cancelled.

## Voice activity detection

```python
from roomkit.vad import detect_voice

segments = detect_voice(samples, sample_rate=16000)
for s in segments:
    print(f"{s.start:6.2f}s - {s.end:6.2f}s")
```

## Speaker diarization

```python
segments = roomkit.diarize(samples, num_speakers=2)  # omit num_speakers
                                                     # to estimate it
for s in segments:
    print(s.label, f"{s.start:.2f}-{s.end:.2f}")
```

## Room simulation and RT60

```python
from roomkit.acoustics.rir import ShoeboxRoom, simulate_rir
from roomkit.acoustics.rt60 import estimate_rt60
from roomkit.acoustics.sabine import sabine_rt60

room = ShoeboxRoom(dimensions=(5.0, 4.0, 3.0), absorption=0.2)
rir = simulate_rir(room, source=(1.0, 1.0, 1.5), microphone=(3.5, 2.5, 1.0))
print(estimate_rt60(rir, 16000), sabine_rt60(room.volume, room.surface_area, 0.2))
```

## Codec tokens

```python
from roomkit.codec.frontend import CodecFrontend

frontend = CodecFrontend(n_codebooks=4, codebook_size=32).fit(samples)
tokens = frontend.encode(samples)        # (num_frames, n_codebooks)
waveform = frontend.synthesize(tokens)   # Griffin-Lim resynthesis
```

## Evaluation

```python
from roomkit.metrics.sdr import si_sdr
from roomkit.metrics.spectral import log_spectral_distortion, segmental_snr
from roomkit.metrics.der import diarization_error_rate

print(si_sdr(clean, enhanced), log_spectral_distortion(clean, enhanced))
```

## Command line

```bash
roomkit denoise noisy.wav clean.wav --method wiener
roomkit dereverb roomy.wav dry.wav --taps 10 --iterations 5
roomkit vad meeting.wav --output vad.json
roomkit diarize meeting.wav --num-speakers 2
roomkit rt60 impulse_response.wav
roomkit codec encode in.wav tokens.npz
roomkit codec decode tokens.npz out.wav
```
