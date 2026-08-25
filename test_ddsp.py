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
    print(f"\nthe decoder has no level input and no level output "
          f"[{model.harm_mode}]")
    d = model.decoder
    check("input width is 1 + Z_DIM", d.mlp_in[0].in_features == 1 + ddsp.Z_DIM,
          f"got {d.mlp_in[0].in_features}")
    check("no amplitude head", not hasattr(d, "head_amp"))
    Tf = 40
    f0 = torch.full((1, Tf), 300.0)
    with torch.no_grad():
        harm, noise, mix = d(ddsp.scale_f0(f0), torch.randn(1, Tf, ddsp.Z_DIM), f0)
    check("harmonics sum to 1",
          bool((harm.sum(-1) - 1).abs().max() < 1e-4))
    check("noise bands are unit RMS",
          bool((noise.pow(2).mean(-1).sqrt() - 1).abs().max() < 1e-4))
    check("mix is a bounded fraction",
          bool(mix.min() >= 0 and mix.max() <= 1),
          f"range {float(mix.min()):.3f}..{float(mix.max()):.3f}")


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
    print(f"\nstreaming synthesis agrees with the batch synthesizer "
          f"[{model.harm_mode}]")
    z = torch.randn(ddsp.Z_DIM).numpy()
    s = ddsp.StreamingSynth(model)
    out = np.concatenate([s.render(300.0, 0.0, z) for _ in range(80)])
    got = float(np.sqrt((out[20 * ddsp.HOP:] ** 2).mean()))
    want = float(ddsp.level_to_amp(0.0))
    check("held frame renders at the reference level",
          abs(got - want) / want < 0.08,
          f"{got:.5f} vs {want:.5f} ({100 * (got - want) / want:+.1f}%)")


def main():
    torch.manual_seed(0)
    base = a_performance()
    models = {m: ddsp.DDSP(harm_mode=m).eval() for m in ddsp.HARM_MODES}

    # gain invariance and level accuracy live in the encoder and synthesizer,
    # which both modes share — checking one is checking both
    test_analysis_is_gain_invariant(models["amp"], base)
    test_mel_reference_absorbs_gain(base)
    test_output_level_is_what_was_asked(models["amp"])
    test_envelope_sampling()
    # the decoder heads and the streaming path do differ by mode
    for m in ddsp.HARM_MODES:
        test_decoder_cannot_see_level(models[m])
        test_streaming_matches_batch(models[m])

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
