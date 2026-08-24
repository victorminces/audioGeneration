"""Web gesture instrument — backend.

The browser does everything real-time (RBF map eval, DDSP decoder, synthesis
in an AudioWorklet). This server only:
  GET  /api/decoder/manifest   decoder tensor names/shapes/offsets + constants
  GET  /api/decoder/weights    all decoder tensors as one float32 blob
  GET  /api/state              current mapper state (W, sigma, seconds)
  POST /api/take               demonstration audio + cursor -> refit -> new W
  POST /api/clear              reset the mapper

Run:  uvicorn server:app --port 8321      (from web/, or python server.py)
"""
import io
import json
import os
import sys
import threading

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import ddsp                                     # noqa: E402
from mapper import KernelMapper, load_latest_ddsp  # noqa: E402

torch.set_num_threads(2)

print("loading DDSP model…", flush=True)
MODEL, MODEL_NAME = load_latest_ddsp(PROJ)
print(f"model: {MODEL_NAME}", flush=True)

# One mapper per modality, kept side by side: a pad point and a body pose are
# different kinds of thing, so their demonstrations can't be pooled — but they
# also shouldn't destroy each other, so flipping the toggle in the front end
# leaves the other space's work intact.
SPACES = {"xy": 2, "body": 4}
# Minimum gap in log-RMS (~17 dB) between the quiet and loud halves of a take
# before we believe they are two different things rather than one performance.
MIN_SEPARATION = 2.0


def voiced_mask(loud):
    """Which frames of a take were actually sung.

    A take's loudness is two clusters — silence and voice — with a wide gap
    between them, so the split is found from the take's own distribution
    rather than any fixed level: no assumption about microphone gain, room
    noise, or how loudly you sing.

    This replaced "90th percentile minus a margin", which quietly stops
    working once singing is under a tenth of the take (the reference lands in
    the silence and everything is kept). Measured on a take that was 11%
    singing, the percentile rule kept 100% of the silence; this keeps 0%.
    """
    n = len(loud)
    if n < 8:
        return np.ones(n, bool)
    lo, hi = float(loud.min()), float(loud.max())
    if hi - lo < MIN_SEPARATION:                # uniformly loud: nothing to drop
        return np.ones(n, bool)

    best_v, best_t = -1.0, None                 # Otsu: maximum between-class variance
    for th in np.linspace(lo, hi, 64)[1:-1]:
        a, b = loud[loud <= th], loud[loud > th]
        if len(a) < 2 or len(b) < 2:
            continue
        v = len(a) * len(b) * (b.mean() - a.mean()) ** 2
        if v > best_v:
            best_v, best_t = v, th
    if best_t is None:
        return np.ones(n, bool)

    sung = loud > best_t
    quiet = ~sung
    # A take with no silence in it still has a loudest and a quietest half;
    # without this guard we would throw away the softer half of a performance.
    if not quiet.any() or (loud[sung].mean() - loud[quiet].mean()) < MIN_SEPARATION:
        return np.ones(n, bool)
    return sung
MAPPERS = {name: KernelMapper(dim=d) for name, d in SPACES.items()}
LOCK = threading.Lock()


def mapper_for(space):
    return MAPPERS.get(space if space in MAPPERS else "xy")

# decoder weights, flattened once in a fixed order the worklet knows
_sd = MODEL.decoder.state_dict()
DECODER_ORDER = [
    "mlp_in.0.weight", "mlp_in.0.bias", "mlp_in.1.weight", "mlp_in.1.bias",
    "mlp_in.3.weight", "mlp_in.3.bias", "mlp_in.4.weight", "mlp_in.4.bias",
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "mlp_out.0.weight", "mlp_out.0.bias", "mlp_out.1.weight", "mlp_out.1.bias",
    "head_amp.weight", "head_amp.bias",
    "head_harm.weight", "head_harm.bias",
    "head_noise.weight", "head_noise.bias",
]
_manifest, _chunks, _off = [], [], 0
for name in DECODER_ORDER:
    t = _sd[name].detach().cpu().numpy().astype(np.float32)
    _manifest.append({"name": name, "shape": list(t.shape), "offset": _off})
    _chunks.append(t.ravel())
    _off += t.size
WEIGHTS_BLOB = np.concatenate(_chunks).tobytes()

app = FastAPI()


def state_json(space="xy"):
    m = mapper_for(space)
    return {
        # one row per thinned sample: [p_1..p_D, u_1..u_D, log_f0, loud, z_1..z_16]
        # u = unit tangent of the stroke there; all-zero = no clear direction
        "points": (np.concatenate([m.S, m.U, m.T], axis=1).tolist()
                   if m.ready else None),
        "space": space,
        "dim": m.dim,               # D of the gesture space; the worklet strides by it
        "mode": m.mode,
        "sigma": m.sigma,
        "sigma_par": m.sigma_par,
        "reach": m.reach,
        "seconds": m.seconds,
        "model": MODEL_NAME,
    }


@app.get("/api/decoder/manifest")
def decoder_manifest():
    return {
        "tensors": _manifest,
        "total": _off,
        # F0_MIN/MAX come from the loaded checkpoint, not the module defaults:
        # the worklet must scale f0 exactly the way these weights were trained
        "constants": {"SR": ddsp.SR, "HOP": ddsp.HOP,
                      "N_HARMONICS": ddsp.N_HARMONICS,
                      "N_NOISE": ddsp.N_NOISE, "Z_DIM": ddsp.Z_DIM,
                      "F0_MIN": MODEL.f0_min, "F0_MAX": MODEL.f0_max},
    }


@app.get("/api/decoder/weights")
def decoder_weights():
    return Response(WEIGHTS_BLOB, media_type="application/octet-stream")


@app.get("/api/state")
def get_state(space: str = "xy"):
    with LOCK:
        return state_json(space)


@app.post("/api/take")
async def post_take(audio: UploadFile, cursor: str = Form(...), sr: int = Form(...),
                    space: str = Form("xy"), loud_threshold: float = Form(None)):
    """audio: raw little-endian float32 mono; cursor: JSON [[t, p_1..p_D],...]
    (D=2 for the mouse pad, D=4 for the full-body features)."""
    raw = await audio.read()
    wav = np.frombuffer(raw, dtype=np.float32).copy()
    cur = np.array(json.loads(cursor), dtype=np.float64)
    if sr != ddsp.SR:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav).unsqueeze(0), sr, ddsp.SR)[0].numpy()
    if len(cur) < 6 or len(wav) < ddsp.SR // 2:
        with LOCK:
            return {"ok": False, "reason": "take too short", **state_json(space)}

    wt = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    n_frames = max(1, wt.shape[1] // ddsp.HOP)
    p = ddsp.analyze(MODEL, wt)
    logf0 = p["f0"].clamp(min=1).log().numpy()
    targets = np.concatenate([logf0[:, None], p["loud"].numpy()[:, None],
                              p["z"].numpy()], axis=1)
    t = (np.arange(n_frames) + 0.5) * ddsp.HOP / ddsp.SR
    t = t[t <= cur[-1, 0]]                      # audio may outlast cursor
    # resample every gesture dimension onto the analysis frame times
    dim = cur.shape[1] - 1                      # column 0 is the timestamp
    pts = np.stack([np.interp(t, cur[:, 0], cur[:, j + 1]) for j in range(dim)],
                   axis=1)
    tg = targets[:len(t)]

    # Drop the frames you weren't actually singing on.
    #
    # A take is bracketed by getting into position and punctuated by breathing
    # between phrases, and those frames are near-silent. The decoder renders
    # them faithfully — as almost pure noise: measured on this model, frames in
    # the quietest 40% come out ~170x breathier than sung ones, at a thousandth
    # of the amplitude. Storing them means the instrument is quiet and breathy
    # wherever you happened to be standing while not singing, which is most of
    # the gesture space.
    # Prefer the threshold the front end measured from the room during the hold
    # gesture: it is what the live indicator was showing while you sang, so what
    # gets dropped is exactly what you were told would be dropped. Fall back to
    # splitting the take's own distribution when there was no calibration (the
    # x,y modality has no hold gesture) or when trusting it would leave almost
    # nothing — a cough during the hold shouldn't cost you the take.
    loud_col = tg[:, 1]
    keep = None
    if loud_threshold is not None:
        cal = loud_col > float(loud_threshold)
        if cal.mean() >= 0.15:
            keep = cal
    if keep is None:
        keep = voiced_mask(loud_col)
    dropped = int(len(keep) - keep.sum())
    if keep.sum() >= 6:                         # unless that leaves nothing to fit
        pts, tg = pts[keep], tg[keep]
    else:
        dropped = 0

    with LOCK:
        mapper_for(space).add(pts, tg, seconds=len(tg) * ddsp.HOP / ddsp.SR)
        return {"ok": True, "path": pts.tolist(),
                "dropped_seconds": round(dropped * ddsp.HOP / ddsp.SR, 2),
                **state_json(space)}


@app.post("/api/config")
def post_config(mode: str = Form(None), sigma: float = Form(None),
                sigma_par: float = Form(None), reach: float = Form(None),
                space: str = Form("xy")):
    with LOCK:
        m = mapper_for(space)
        if mode in ("iso", "aniso"):
            m.mode = mode
        if sigma is not None:
            # ceiling is generous: in 4D points sit further apart than on the
            # 2D pad, so a usable across-width there runs well past the old 0.6
            m.sigma = float(np.clip(sigma, 0.03, 1.2))
        if sigma_par is not None:
            m.sigma_par = float(np.clip(sigma_par, 0.01, 0.3))
        if reach is not None:
            m.reach = float(np.clip(reach, 1.0, 8.0))
        return state_json(space)


@app.post("/api/clear")
def post_clear(space: str = Form("xy")):
    with LOCK:
        # only the named space is reset; the other modality keeps its work
        name = space if space in MAPPERS else "xy"
        MAPPERS[name] = KernelMapper(dim=SPACES[name])
        return state_json(name)


STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/api/hand_model")
def hand_model():
    return FileResponse(os.path.join(PROJ, "models", "hand_landmarker.task"),
                        media_type="application/octet-stream")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_cache(request, call_next):
    """Never let the browser hold on to the front end or the map state.

    Two failures this prevents, both seen in practice: a cached index.html
    paired with a fresh main.js, where the script looks for controls the stale
    markup doesn't have; and a cached GET /api/state, where the front end is
    handed an empty map that the server filled in long ago. Everything here is
    tiny and served from localhost, so caching buys nothing.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8321)
