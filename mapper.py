"""Gesture->voice mapping shared by the desktop instrument and the web server.

KernelMapper: kernel smoothing (Nadaraya-Watson) from a gesture point in
[0,1]^D to voice parameters [log_f0, loudness, z_1..z_16]. The prediction at a
point is the Gaussian-weighted average of the demonstrated samples, so it can
never leave the range of what was actually sung. A confidence — the kernel
value at the nearest sample — fades the instrument to silence away from the
data.

D is whatever the demonstrations carry: 2 for the original (x, y) pad, 4 for
the full-body features (per-arm extension + per-hand height). Every step below
is dimension-agnostic; nothing assumes a plane.

Two kernel modes:
  "iso"    round Gaussian, one width (sigma).
  "aniso"  Gaussian oriented by the local direction of travel: narrow along
           the stroke (sigma_par, keeps consecutive sounds from smearing),
           wide across it (sigma, the reach before fading to silence).

Samples are thinned into small spatial bins (mean per bin) so prediction cost
stays bounded no matter how long the demonstrations run. Tangents that
disagree within a bin (path crossings) average toward zero, which gracefully
falls back to round behavior there.
"""
import glob
import json
import os

import numpy as np
import torch

import ddsp

# Thinning grid, bins of 1/res on a side. Coarser as D grows: a fixed 1/48 grid
# has 48^D cells, so in 4D almost every sample lands alone and thinning stops
# doing anything. These keep the cell count (and so the worklet's per-frame
# loop) in the same ballpark across dimensions.
BIN_RES_BY_DIM = {2: 48, 3: 24, 4: 16}
MODES = ("iso", "aniso")


def _bin_res(dim):
    return BIN_RES_BY_DIM.get(dim, 12)


class KernelMapper:
    N_OUT = 2 + ddsp.Z_DIM

    def __init__(self, dim=2):
        self.mode = "aniso"
        self.sigma = 0.15                       # across-stroke width (iso: only width)
        self.sigma_par = 0.04                   # along-stroke width (aniso only)
        # How far the *gain* reaches, as a multiple of sigma. Widening sigma by
        # reach divides the exponent by reach^2, so this only softens the fade
        # to silence — the averaging stays exactly as sharp as sigma says. It
        # exists because the same demonstrations sit much further apart in 4-D
        # than on the 2-D pad, so the nearest sample is rarely close and the
        # instrument reads as too quiet.
        #
        # 2.0 rather than something larger: reach is really conf^(1/reach^2),
        # and by 3.0 the fade is all but gone (measured: level only 1.6x down
        # right across the space, versus 36x at reach 1), which throws away the
        # "silent where you never taught it" behaviour. At 2.0 the level in the
        # region you actually play is ~2x up and the fade still fades.
        self.reach = 1.0 if dim <= 2 else 2.0
        self.dim = dim                          # D of the gesture space
        self.data_p = []                        # raw per-take points, kept for
        self.data_t = []                        # save/load and re-thinning
        self.S = None                           # (M, D) thinned sample positions
        self.U = None                           # (M, D) unit tangents (0 where unclear)
        self.T = None                           # (M, N_OUT) thinned targets
        self.seconds = 0.0

    @property
    def ready(self):
        return self.S is not None

    def add(self, p, targets, seconds):
        p = np.asarray(p, dtype=np.float64)
        # first take defines the space; a take of a different width would make
        # the stored samples inconsistent, so start over rather than mix them
        if self.data_p and p.shape[1] != self.dim:
            self.data_p, self.data_t, self.seconds = [], [], 0.0
        self.dim = p.shape[1]
        self.data_p.append(p)
        self.data_t.append(targets)
        self.seconds += seconds
        self.refit()

    def refit(self):
        """Rebuild the thinned sample set (one mean sample per occupied bin)."""
        if not self.data_p:
            self.S = self.U = self.T = None
            return
        p = np.concatenate(self.data_p)
        t = np.concatenate(self.data_t)
        # local direction of travel, per take so takes don't bleed into each other
        u = np.concatenate([self._tangents(a) for a in self.data_p])

        res = _bin_res(self.dim)
        idx = np.minimum((np.clip(p, 0.0, 1.0) * res).astype(np.int64), res - 1)
        # group by whole bin coordinate: no flat index, so no dimension limit
        _, inv = np.unique(idx, axis=0, return_inverse=True)
        m = int(inv.max()) + 1
        counts = np.bincount(inv, minlength=m).astype(np.float64)

        def bin_mean(a):
            out = np.empty((m, a.shape[1]))
            for j in range(a.shape[1]):
                out[:, j] = np.bincount(inv, weights=a[:, j], minlength=m)
            return out / counts[:, None]

        self.S = bin_mean(p)
        self.T = bin_mean(t)
        um = bin_mean(u)
        n = np.linalg.norm(um, axis=1, keepdims=True)
        # crossings/stalls average out -> keep zero (falls back to round kernel)
        self.U = np.where(n > 0.25, um / (n + 1e-12), 0.0)

    @staticmethod
    def _tangents(p):
        """(N, D) path -> (N, D) unit tangents; zero where the gesture stood
        still. Tangent sign doesn't matter (the kernel uses its square)."""
        if len(p) < 2:
            return np.zeros_like(p)
        g = np.gradient(p, axis=0)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return np.where(n > 1e-4, g / (n + 1e-12), 0.0)

    def predict(self, p):
        """(N, D) -> (pred (N, N_OUT), conf (N,)) or None if no data.

        pred is the kernel-weighted average of the samples; conf is the kernel
        value at the nearest sample (1 on the data, ->0 away from it), meant to
        be applied as an output gain."""
        if self.S is None:
            return None
        d = p[:, None, :] - self.S[None, :, :]              # (N, M, D)
        d2 = (d ** 2).sum(axis=2)
        if self.mode == "aniso":
            along = (d * self.U[None, :, :]).sum(axis=2)    # offset along tangent
            across2 = np.maximum(d2 - along ** 2, 0.0)
            e = (along ** 2 / (2 * self.sigma_par ** 2)
                 + across2 / (2 * self.sigma ** 2))
        else:
            e = d2 / (2 * self.sigma ** 2)
        k = np.exp(-e)
        sw = k.sum(axis=1)
        # gain uses the same nearest-sample distance but a wider sigma, which
        # scales the exponent by 1/reach^2 — so the averaging stays as sharp as
        # sigma says while the fade to silence is gentler
        conf = np.exp(-e.min(axis=1) / (self.reach ** 2))
        pred = (k @ self.T) / (sw[:, None] + 1e-12)
        return pred, conf

    def save(self, path, ddsp_name):
        np.savez(path,
                 p=np.concatenate(self.data_p) if self.data_p
                   else np.zeros((0, self.dim)),
                 t=np.concatenate(self.data_t) if self.data_t
                   else np.zeros((0, self.N_OUT)),
                 dim=self.dim, reach=self.reach,
                 sigma=self.sigma, sigma_par=self.sigma_par, mode=self.mode,
                 seconds=self.seconds, ddsp_name=ddsp_name)

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        p = z["p"] if "p" in z else z["xy"]      # "xy": pre-N-D saved maps
        self.data_p = [p] if len(p) else []
        self.data_t = [z["t"]] if len(z["t"]) else []
        self.dim = int(z["dim"]) if "dim" in z else (p.shape[1] if len(p) else 2)
        self.reach = float(z["reach"]) if "reach" in z else (1.0 if self.dim <= 2 else 3.0)
        self.sigma = float(z["sigma"])
        self.sigma_par = float(z["sigma_par"]) if "sigma_par" in z else 0.04
        self.mode = str(z["mode"]) if "mode" in z else "aniso"
        self.seconds = float(z["seconds"])
        self.refit()
        return str(z["ddsp_name"])


def load_latest_ddsp(proj=None):
    proj = proj or os.path.dirname(os.path.abspath(__file__))
    cfgs = []
    for p in glob.glob(os.path.join(proj, "checkpoints", "*.json")):
        with open(p) as f:
            cfg = json.load(f)
        if cfg.get("domain") == "ddsp":
            pt = os.path.join(proj, "checkpoints", cfg["filename"])
            if os.path.exists(pt):
                cfgs.append((os.path.getmtime(pt), cfg, pt))
    if not cfgs:
        raise RuntimeError("No DDSP checkpoint found — train one first.")
    _, cfg, pt = max(cfgs)
    # A checkpoint's weights interpret f0 on the scale it was trained with, so
    # the range comes from its config rather than today's default. Checkpoints
    # written before the range was widened have no entry and were all trained
    # on 60-600; loading one with the current default would transpose
    # everything it plays by well over an octave.
    m = ddsp.DDSP(f0_min=float(cfg.get("f0_min", 60.0)),
                  f0_max=float(cfg.get("f0_max", 600.0)))
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    m.eval()
    return m, cfg["name"]
