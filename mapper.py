"""Gesture->voice mapping shared by the desktop instrument and the web server.

RBFMapper: ridge regression from (x, y) in [0,1]^2 to voice parameters
[log_f0, loudness, z_1..z_16] over a fixed grid of Gaussian bumps.
Incremental: accumulates raw pairs; solves in closed form on demand.
"""
import glob
import json
import os

import numpy as np
import torch

import ddsp

GRID      = 8                  # RBF grid (GRID x GRID bumps)
RIDGE_LAM = 1e-2


class RBFMapper:
    N_OUT = 2 + ddsp.Z_DIM

    def __init__(self):
        cx, cy = np.meshgrid(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID))
        self.centers = np.stack([cx.ravel(), cy.ravel()], axis=1)   # (G^2, 2)
        self.sigma = 1.2 / GRID
        self.lam = RIDGE_LAM
        self.n_feat = GRID * GRID + 1
        self.data_xy = []                       # raw pairs, kept so sigma/lam
        self.data_t = []                        # can be changed after the fact
        self.W = None
        self.seconds = 0.0

    def features(self, xy):
        """(N, 2) -> (N, n_feat)"""
        d2 = ((xy[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        phi = np.exp(-d2 / (2 * self.sigma ** 2))
        return np.concatenate([phi, np.ones((len(xy), 1))], axis=1)

    def add(self, xy, targets, seconds):
        self.data_xy.append(xy)
        self.data_t.append(targets)
        self.seconds += seconds
        self.refit()

    def refit(self):
        if not self.data_xy:
            self.W = None
            return
        X = self.features(np.concatenate(self.data_xy))
        Y = np.concatenate(self.data_t)
        reg = self.lam * len(X) * np.eye(self.n_feat)
        self.W = np.linalg.solve(X.T @ X + reg, X.T @ Y)

    def predict(self, xy):
        if self.W is None:
            return None
        return self.features(xy) @ self.W       # (N, N_OUT)

    def save(self, path, ddsp_name):
        np.savez(path,
                 xy=np.concatenate(self.data_xy) if self.data_xy else np.zeros((0, 2)),
                 t=np.concatenate(self.data_t) if self.data_t else np.zeros((0, self.N_OUT)),
                 sigma=self.sigma, lam=self.lam,
                 seconds=self.seconds, ddsp_name=ddsp_name)

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        self.data_xy = [z["xy"]] if len(z["xy"]) else []
        self.data_t = [z["t"]] if len(z["t"]) else []
        self.sigma = float(z["sigma"]); self.lam = float(z["lam"])
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
    m = ddsp.DDSP()
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    m.eval()
    return m, cfg["name"]
