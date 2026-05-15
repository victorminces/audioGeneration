# Generative Sound

A Streamlit app for training waveform autoencoders (AE and VAE) and exploring latent space interpolation between audio clips.

## What it does

- **Train** an autoencoder or VAE on your own audio files (WAV/FLAC)
- **Reconstruct** audio through the bottleneck to hear compression quality
- **Interpolate** between two sounds by walking through the latent space
- **Compare** multiple trained models side by side
- Download public datasets (LibriSpeech, Speech Commands, NSynth) directly from the app

## Architecture

1D convolutional encoder/decoder operating on raw waveforms at 16 kHz. Clips are 50 ms (800 samples). The encoder downsamples 16× through 4 conv layers; the decoder mirrors it with transposed convolutions. The VAE variant uses a dual-head encoder (mu + logvar) with reparameterization, free-bits KL flooring, and configurable β.

## Known limitations

### Interpolation sounds like addition of both sounds

The core problem with waveform-domain interpolation. When two audio clips have different phases, linearly mixing their waveforms (or their latent representations) causes phase interference — the result sounds like both sounds playing simultaneously rather than a smooth morph between them.

Spherical interpolation (slerp) is available as a toggle but does not fully solve this, because the issue is not the interpolation geometry but the fact that the decoder must produce a single coherent-phase waveform from a latent point that sits between two phase-misaligned encodings.

**The real fix:** move to the spectrogram domain. Encoding magnitude spectrograms (e.g. mel spectrograms) avoids the phase problem — interpolating magnitudes produces smooth timbral blends, and audio is reconstructed with Griffin-Lim or a learned vocoder. This is how production audio VAEs (RAVE, EnCodec, etc.) approach the problem.

### KL loss and latent space regularity

With β=0.001 the VAE behaves almost like a plain autoencoder — the latent space is not regularized enough for smooth interpolation. Increasing β (tried up to 0.01) tightens the space but does not eliminate the phase problem described above.

### Short clip length

50 ms clips are too short to capture harmonic and timbral structure reliably. Longer clips (200–500 ms) would give the encoder more context but require more memory and longer training.

## Running

```bash
python -m streamlit run app.py
```

Requires: `torch`, `streamlit`, `soundfile`, `numpy`, `audio-recorder-streamlit`, `requests`
