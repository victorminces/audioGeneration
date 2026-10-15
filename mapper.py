"""Gesture->voice mapping shared by the desktop instrument and the web server.

KernelMapper: kernel smoothing (Nadaraya-Watson) from (x, y) in [0,1]^2 to
voice parameters [log_f0, loudness, z_1..z_16]. The prediction at a point is
the Gaussian-weighted average of the demonstrated samples, so it can never
leave the range of what was actually sung. A confidence — the kernel value at
the nearest sample — fades the instrument to silence away from the data.

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

BIN_RES = 48                   # thinning grid: bins of 1/BIN_RES on a side
MODES = ("iso", "aniso")


class KernelMapper:
    N_OUT = 2 + ddsp.Z_DIM

    def __init__(self):
        self.mode = "aniso"
        self.sigma = 0.15                       # across-stroke width (iso: only width)
        self.sigma_par = 0.04                   # along-stroke width (aniso only)
        self.data_xy = []                       # raw per-take pairs, kept for
        self.data_t = []                        # save/load and re-thinning
        self.S = None                           # (M, 2) thinned sample positions
        self.U = None                           # (M, 2) unit tangents (0 where unclear)
        self.T = None                           # (M, N_OUT) thinned targets
        self.seconds = 0.0

    @property
    def ready(self):
        return self.S is not None

    def add(self, xy, targets, seconds):
        self.data_xy.append(xy)
        self.data_t.append(targets)
        self.seconds += seconds
        self.refit()

    def refit(self):
        """Rebuild the thinned sample set (one mean sample per occupied bin)."""
        if not self.data_xy:
            self.S = self.U = self.T = None
            return
        xy = np.concatenate(self.data_xy)
        t = np.concatenate(self.data_t)
        # local direction of travel, per take so takes don't bleed into each other
        u = np.concatenate([self._tangents(a) for a in self.data_xy])
        idx = np.minimum((xy * BIN_RES).astype(np.int64), BIN_RES - 1)
        flat = idx[:, 0] * BIN_RES + idx[:, 1]
        order = np.argsort(flat, kind="stable")
        flat, xy, t, u = flat[order], xy[order], t[order], u[order]
        _, starts = np.unique(flat, return_index=True)
        self.S = np.stack([g.mean(axis=0) for g in np.split(xy, starts[1:])])
        self.T = np.stack([g.mean(axis=0) for g in np.split(t, starts[1:])])
        um = np.stack([g.mean(axis=0) for g in np.split(u, starts[1:])])
        n = np.linalg.norm(um, axis=1, keepdims=True)
        # crossings/stalls average out -> keep zero (falls back to round kernel)
        self.U = np.where(n > 0.25, um / (n + 1e-12), 0.0)

    @staticmethod
    def _tangents(xy):
        """(N, 2) path -> (N, 2) unit tangents; zero where the hand stood still.
        Tangent sign doesn't matter (the kernel uses its square)."""
        if len(xy) < 2:
            return np.zeros_like(xy)
        g = np.gradient(xy, axis=0)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return np.where(n > 1e-4, g / (n + 1e-12), 0.0)

    def predict(self, xy):
        """(N, 2) -> (pred (N, N_OUT), conf (N,)) or None if no data.

        pred is the kernel-weighted average of the samples; conf is the kernel
        value at the nearest sample (1 on the data, ->0 away from it), meant to
        be applied as an output gain."""
        if self.S is None:
            return None
        d = xy[:, None, :] - self.S[None, :, :]             # (N, M, 2)
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
        conf = k.max(axis=1)
        pred = (k @ self.T) / (sw[:, None] + 1e-12)
        return pred, conf

    def save(self, path, ddsp_name):
        np.savez(path,
                 xy=np.concatenate(self.data_xy) if self.data_xy else np.zeros((0, 2)),
                 t=np.concatenate(self.data_t) if self.data_t else np.zeros((0, self.N_OUT)),
                 sigma=self.sigma, sigma_par=self.sigma_par, mode=self.mode,
                 seconds=self.seconds, ddsp_name=ddsp_name)

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        self.data_xy = [z["xy"]] if len(z["xy"]) else []
        self.data_t = [z["t"]] if len(z["t"]) else []
        self.sigma = float(z["sigma"])
        self.sigma_par = float(z["sigma_par"]) if "sigma_par" in z else 0.04
        self.mode = str(z["mode"]) if "mode" in z else "aniso"
        self.seconds = float(z["seconds"])
        self.refit()
        return str(z["ddsp_name"])


class FeatureMapper:
    """Kernel smoothing from a face-shape vector (D mouth/jaw measurements)
    to the same voice parameters as KernelMapper — any sound, not just a 2D
    chart: whatever was sung with a given mouth shape comes back at that shape.

    Each feature is divided by its spread over the taught data (floored, so a
    barely-moving feature can't turn jitter into distance), then a round
    Gaussian of width sigma works in that normalized space. Samples are kept
    per frame (no spatial bins in D dimensions), uniformly thinned past MAX_N.
    """
    MAX_N = 4000
    SCALE_FLOOR = 0.03

    def __init__(self):
        self.sigma = 1.0
        self.seconds = 0.0
        self.data_f = []
        self.data_t = []
        self.F = None                           # (M, D) normalized features
        self.T = None                           # (M, N_OUT) targets
        self.scale = None                       # (D,) per-feature divisor

    @property
    def ready(self):
        return self.F is not None

    def add(self, feats, targets, seconds=0.0):
        if self.data_f and feats.shape[1] != self.data_f[0].shape[1]:
            self.data_f, self.data_t = [], []   # feature set changed: start over
            self.seconds = 0.0
        self.seconds += seconds
        self.data_f.append(feats)
        self.data_t.append(targets)
        self.refit()

    def refit(self):
        if not self.data_f:
            self.F = self.T = self.scale = None
            return
        f = np.concatenate(self.data_f)
        t = np.concatenate(self.data_t)
        self.scale = np.maximum(f.std(axis=0), self.SCALE_FLOOR)
        if len(f) > self.MAX_N:
            keep = np.linspace(0, len(f) - 1, self.MAX_N).astype(np.int64)
            f, t = f[keep], t[keep]
        self.F = f / self.scale
        self.T = t


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
    m = ddsp.DDSP()
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    m.eval()
    return m, cfg["name"]
