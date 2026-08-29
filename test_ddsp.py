"""Structural guarantees of the DDSP model. Run: python test_ddsp.py

These are the properties the instrument's design rests on, so they are checked
rather than assumed. They hold at *any* weights — that is what makes them
structural — so the tests use an untrained model and need no checkpoint.

The one they exist to protect is gain invariance: analysing a performance
recorded at two microphone distances must give the same numbers. It has been
broken twice already, both times by something that looked unrelated — an
absolute epsilon added before a log, and a normalisation applied over the wrong
axis — and neither was audible until much later.
"""
import sys

import numpy as np
import torch

import ddsp

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def a_performance():
    """Two phrases at different levels, with silence around and between them —
    the silence matters, since it is what an absolute floor would clamp."""
    t = np.arange(int(ddsp.SR * 2.0)) / ddsp.SR
    w = np.zeros_like(t)
    for lo, hi, f, a in ((0.15, 0.75, 220.0, 0.6), (1.10, 1.80, 330.0, 0.15)):
        seg = (t > lo) & (t < hi)
        w[seg] = a * np.sin(2 * np.pi * f * t[seg])
    w += np.random.default_rng(0).normal(0, 1e-4, w.shape)
    return torch.tensor(w, dtype=torch.float32).unsqueeze(0)


def test_analysis_is_gain_invariant(model, base):
    print("\nanalysis is invariant to recording gain")
    p0 = ddsp.analyze(model, base)
    d_rel = d_z = d_f0 = d_ref = 0.0
    for g in (0.03125, 0.125, 0.5, 2.0, 8.0, 32.0):
        p = ddsp.analyze(model, base * g)
        d_rel = max(d_rel, float((p["rel"] - p0["rel"]).abs().max()))
        d_z = max(d_z, float((p["z"] - p0["z"]).abs().max()))
        d_f0 = max(d_f0, float((p["f0"] - p0["f0"]).abs().max()))
        d_ref = max(d_ref, abs(p["ref"] - np.log(g) - p0["ref"]))
    check("relative level unchanged", d_rel < 1e-4, f"max drift {d_rel:.2e}")
    check("timbre code unchanged", d_z < 1e-4, f"max drift {d_z:.2e}")
    check("pitch unchanged", d_f0 < 1e-4, f"max drift {d_f0:.2e}")
    check("performed level tracks log(gain)", d_ref < 1e-3, f"max error {d_ref:.2e}")


def test_mel_reference_absorbs_gain(base):
    print("\nmel reference absorbs exactly the gain, nothing else")
    n = base.shape[1] // ddsp.HOP
    m1, m2 = ddsp.mel_frames(base, n), ddsp.mel_frames(base * 8.0, n)
    shift = float(ddsp.mel_reference(m2) - ddsp.mel_reference(m1))
    resid = float(((m2 - ddsp.mel_reference(m2))
                   - (m1 - ddsp.mel_reference(m1))).abs().max())
    check("reference shifts by 2*log(gain)", abs(shift - 2 * np.log(8.0)) < 1e-3,
          f"{shift:+.4f} vs {2 * np.log(8.0):+.4f}")
    check("nothing else survives the subtraction", resid < 1e-4,
          f"max residual {resid:.2e}")


def test_decoder_cannot_see_level(model):
    print("\nthe decoder sees neither level nor balance")
    d = model.decoder
    check("input width is 2 + Z_DIM", d.mlp_in[0].in_features == 2 + ddsp.Z_DIM,
          f"got {d.mlp_in[0].in_features}")
    check("no amplitude head", not hasattr(d, "head_amp"))
    Tf = 40
    f0 = torch.full((1, Tf), 300.0)
    ap = torch.rand(1, Tf)
    with torch.no_grad():
        harm, noise = d(ddsp.scale_f0(f0), ap, torch.randn(1, Tf, ddsp.Z_DIM), f0)
    check("harmonics sum to 1",
          bool((harm.sum(-1) - 1).abs().max() < 1e-4))
    check("noise bands are unit RMS",
          bool((noise.pow(2).mean(-1).sqrt() - 1).abs().max() < 1e-4))
    check("no balance head either", not hasattr(d, "head_mix"))


def test_envelope_sampling():
    """env_to_harm must read the envelope at k*f0 and nowhere else, or the
    source-filter model is quietly fitting the wrong frequencies."""
    print("\nthe envelope is sampled at k*f0")
    env = torch.linspace(0.1, 3.0, ddsp.N_ENV).view(1, 1, -1)
    for f0_hz in (80.0, 220.0, 1100.0):
        f0 = torch.full((1, 1), f0_hz)
        got = ddsp.env_to_harm(env, f0)[0, 0].numpy()
        want = np.array([np.interp(ddsp.env_band(f0_hz * k),
                                   np.arange(ddsp.N_ENV), env[0, 0].numpy())
                         for k in range(1, ddsp.N_HARMONICS + 1)])
        want = want / want.sum()
        check(f"f0 = {f0_hz:.0f} Hz", float(np.abs(got - want).max()) < 1e-5,
              f"max error {float(np.abs(got - want).max()):.2e}")
    # bands are log spaced, so an octave up is a fixed number of bands along
    step = ddsp.env_band(440.0) - ddsp.env_band(220.0)
    check("an octave is a constant number of bands",
          abs((ddsp.env_band(1760.0) - ddsp.env_band(880.0)) - step) < 1e-4,
          f"{step:.3f} bands per octave")


def test_output_level_is_what_was_asked(model):
    print("\noutput RMS equals the requested amplitude, at every mix")
    Tf = 100
    f0 = torch.full((1, Tf), 300.0)
    harm = torch.zeros(1, Tf, ddsp.N_HARMONICS)
    harm[..., 0], harm[..., 1], harm[..., 2] = 0.5, 0.3, 0.2
    noise = torch.from_numpy(
        np.exp(-np.arange(ddsp.N_NOISE) / 9.0).astype("float32")).view(1, 1, -1)
    noise = (noise / noise.pow(2).mean().sqrt()).expand(1, Tf, -1).contiguous()
    worst = 0.0
    for mv in (0.0, 0.25, 0.5, 0.75, 1.0):
        for amp in (0.02, 0.2):
            y = model.synthesize(f0, torch.full((1, Tf, 1), amp), harm, noise,
                                 torch.full((1, Tf, 1), mv))
            got = float(y[0, 4 * ddsp.HOP:].pow(2).mean().sqrt())
            worst = max(worst, abs(got - amp) / amp)
    check("within 5% at every mix and level", worst < 0.05,
          f"worst {100 * worst:.1f}%")


def test_streaming_matches_batch(model):
    print("\nstreaming synthesis agrees with the batch synthesizer")
    z = torch.randn(ddsp.Z_DIM).numpy()
    s = ddsp.StreamingSynth(model)
    out = np.concatenate([s.render(300.0, 0.0, 0.2, z) for _ in range(80)])
    got = float(np.sqrt((out[20 * ddsp.HOP:] ** 2).mean()))
    want = float(ddsp.level_to_amp(0.0))
    check("held frame renders at the reference level",
          abs(got - want) / want < 0.08,
          f"{got:.5f} vs {want:.5f} ({100 * (got - want) / want:+.1f}%)")


def test_aperiodicity_is_calibrated():
    """The noise fraction is measured, not learned, so the measurement itself
    is the thing that has to be right. Checked against mixtures whose noise
    fraction is known by construction, in the same quadrature convention the
    synthesizer mixes with.

    The *known* pitch is passed in, not a tracked one. This measures how much
    energy sits off a harmonic comb, so it can only be as good as the comb's
    placement — feeding it a tracked pitch tests the tracker and the measure at
    once, and a tracker failure then reads as a measurement failure. That is
    exactly what happened: swapping pitch trackers made this test fail while
    the measure was untouched. The tracker has its own test below.
    """
    print("\naperiodicity tracks a known noise fraction")
    rng = np.random.default_rng(0)
    n = int(ddsp.SR * 2.0)
    tt = np.arange(n) / ddsp.SR
    unit = lambda x: x / (np.sqrt((x ** 2).mean()) + 1e-12)
    fade = np.minimum(1.0, np.minimum(tt / 0.05, (2.0 - tt) / 0.05))
    noise = unit(rng.normal(0, 1, n))
    nf = n // ddsp.HOP
    for f0_hz in (220.0, 1800.0):
        tone = unit(np.sin(2 * np.pi * f0_hz * tt))
        f0 = torch.full((nf,), f0_hz)                  # known, not tracked
        worst = 0.0
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = ((np.sqrt(1 - a) * tone + np.sqrt(a) * noise) * fade * 0.4).astype("float32")
            wt = torch.tensor(x).unsqueeze(0)
            got = float(ddsp.extract_aperiodicity(wt, f0, nf).mean())
            worst = max(worst, abs(got - a))
        check(f"within 0.1 at {f0_hz:.0f} Hz", worst < 0.1, f"worst error {worst:.3f}")


def test_pitch_tracker_is_supported_by_the_spectrum():
    """The tracker's own test: does the pitch it claims actually have energy?

    Judged against the spectrum rather than against another tracker, because
    the two disagree and taking one as ground truth is circular — an error that
    nearly sent this project an octave in the wrong direction. For a claimed
    f0, `support` is how far a narrow band at that frequency stands above the
    frame's median spectral level, and a claim is suspect if far more energy
    sits an octave above it.
    """
    print("\nthe pitch tracker's answer has energy where it says")
    rng = np.random.default_rng(0)
    n = int(ddsp.SR * 2.0)
    tt = np.arange(n) / ddsp.SR
    fade = np.minimum(1.0, np.minimum(tt / 0.05, (2.0 - tt) / 0.05))
    n_fft = 4096
    hz = np.fft.rfftfreq(n_fft, 1 / ddsp.SR)
    for f0_hz, label in ((110.0, "low voice"), (932.0, "whistle"),
                         (2041.0, "high whistle")):
        # a few harmonics, so this is not only a bare sinusoid
        x = sum(np.sin(2 * np.pi * f0_hz * k * tt) / k
                for k in range(1, min(6, int(ddsp.SR / 2 / f0_hz)) + 1))
        x = ((x / np.abs(x).max()) * fade * 0.4).astype("float32")
        wt = torch.tensor(x).unsqueeze(0)
        nf = n // ddsp.HOP
        got = float(ddsp.extract_f0(wt, nf).median())
        S = np.abs(np.fft.rfft(x[:n_fft] * np.hanning(n_fft)))
        near = lambda f: S[(hz > f * 0.97) & (hz < f * 1.03)].max() if f < ddsp.SR / 2 else 0.0
        sup = 20 * np.log10(max(near(got), 1e-12) / (np.median(S) + 1e-12))
        cents = 1200 * np.log2(max(got, 1e-6) / f0_hz)
        check(f"{label} {f0_hz:.0f} Hz", abs(cents) < 60 and sup > 20,
              f"found {got:.0f} Hz ({cents:+.0f} cents), support {sup:.0f} dB")


def test_aperiodicity_agrees_with_world():
    """Cross-check against WORLD's D4C, on the material D4C is authoritative for.

    Optional: skipped unless pyworld is installed. It is a development
    dependency on purpose and is deliberately not in requirements.txt — the app
    never calls it, and it will not build everywhere.

    This exists because the calibration above uses signals this repository
    generated, which makes it a regression guard rather than independent
    validation. D4C is an entirely separate algorithm from a different decade,
    so agreement with it is real corroboration. But only where it applies:
    WORLD assumes a glottal source and refuses to voice pure tones, so the
    check is restricted to speech-like input, and the tolerance is loose
    because the two define aperiodicity differently — D4C reports an amplitude
    ratio against its own CheapTrick envelope, ours an energy fraction against
    the raw spectrum.
    """
    print("\nagrees with WORLD's D4C on speech-like input")
    try:
        import pyworld
    except ImportError:
        print("  SKIP  pyworld not installed (pip install pyworld)")
        return

    sr, dur = ddsp.SR, 2.0
    tt = np.arange(int(sr * dur)) / sr
    fade = np.minimum(1.0, np.minimum(tt / 0.05, (dur - tt) / 0.05))
    rng = np.random.default_rng(0)
    unit = lambda x: x / (np.sqrt((x ** 2).mean()) + 1e-12)

    def formants(f):
        return (np.exp(-((f - 700) ** 2) / (2 * 180 ** 2))
                + 0.7 * np.exp(-((f - 1200) ** 2) / (2 * 250 ** 2))
                + 0.15 * np.exp(-((f - 2600) ** 2) / (2 * 400 ** 2)) + 0.02)

    ph = 2 * np.pi * 200.0 * tt
    vow = sum(formants(200.0 * k) * np.sin(k * ph) / np.sqrt(k)
              for k in range(1, int(sr / 2 / 200.0)))
    n = len(tt)
    breath = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, n))
                          * formants(np.fft.rfftfreq(n, 1 / sr)), n)

    for name, x in (("voiced vowel", unit(vow)),
                    ("breathy vowel", unit(vow) + 0.7 * unit(breath))):
        w = unit(x) * fade * 0.4
        wt = torch.tensor(w.astype("float32")).unsqueeze(0)
        nf = wt.shape[1] // ddsp.HOP
        x64 = np.asarray(w, dtype=np.float64)
        fp = ddsp.HOP / sr * 1000.0
        f0w, tp = pyworld.harvest(x64, sr, f0_floor=ddsp.F0_DETECT_MIN,
                                  f0_ceil=min(ddsp.F0_DETECT_MAX, sr / 2 - 1),
                                  frame_period=fp)
        f0w = pyworld.stonemask(x64, f0w, tp, sr)
        apw = pyworld.d4c(x64, f0w, tp, sr)
        voiced_frac = float((f0w > 0).mean())
        if voiced_frac < 0.5:
            print(f"  SKIP  {name}: WORLD voiced only {100*voiced_frac:.0f}% of it")
            continue
        # Both estimators get WORLD's pitch. Feeding each its own conflates
        # two things: pitch trackers disagree on some material — WORLD's own
        # declines to voice a whistle at all — and a pitch disagreement would
        # then surface here as an aperiodicity disagreement. This test is
        # about the measurement, not the tracker.
        f0t = torch.tensor(np.interp(np.arange(nf) * ddsp.HOP / sr, tp,
                                     np.where(f0w > 0, f0w, np.nan)),
                           dtype=torch.float32)
        f0t = torch.nan_to_num(f0t, nan=float(np.median(f0w[f0w > 0])))
        ours = float(ddsp.extract_aperiodicity(wt, f0t, nf).mean())
        theirs = float((apw[f0w > 0] ** 2).mean())
        check(f"{name} within 0.2", abs(ours - theirs) < 0.2,
              f"ours {ours:.3f} vs D4C {theirs:.3f} (both on WORLD's f0)")


def main():
    torch.manual_seed(0)
    base = a_performance()
    model = ddsp.DDSP().eval()

    test_analysis_is_gain_invariant(model, base)
    test_mel_reference_absorbs_gain(base)
    test_output_level_is_what_was_asked(model)
    test_envelope_sampling()
    test_aperiodicity_is_calibrated()
    test_pitch_tracker_is_supported_by_the_spectrum()
    test_aperiodicity_agrees_with_world()
    test_decoder_cannot_see_level(model)
    test_streaming_matches_batch(model)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
