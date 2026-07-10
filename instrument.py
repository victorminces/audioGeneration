"""Gesture instrument — mapping by demonstration (mouse or camera).

Run with --camera to control with one hand via the webcam:
  PINCH (thumb+index touching) = demonstrate: sing while moving; the pinch
                                 point is the tracked coordinate.
  TWO FINGERS (index+middle extended) = play live; the index fingertip is
                                 the tracked coordinate.
  Any other hand pose = idle. The mouse keeps working alongside.

DEMONSTRATE: hold the LEFT mouse button, move the cursor and sing.
             Each take updates the gesture->voice map (ridge over RBF grid).
PLAY:        hold the RIGHT mouse button and trace a path in silence.
             On release, the mapped voice parameters are synthesized (DDSP)
             and played back.
Keys:        S save session   L load session   C clear map   Q/Esc quit

The mapper is linear-in-features (Gaussian RBF grid + bias), fitted in closed
form and updatable incrementally: meaningful after ~5 s of demonstration,
refining smoothly without limit as takes accumulate.
"""
import os
import sys
import glob
import json
import time

import threading
from collections import deque

import numpy as np
import torch
import pygame
import sounddevice as sd

torch.set_num_threads(1)   # tiny per-frame ops: avoid thread-pool overhead

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
import ddsp  # noqa: E402

SIZE        = 560                  # square size in px
MARGIN      = 40
CURSOR_HZ   = 60
GRID        = 8                    # RBF grid (GRID x GRID bumps)
RIDGE_LAM   = 1e-2
SESSION_DIR = os.path.join(PROJ, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


# ── mapper ────────────────────────────────────────────────────────────────────

class RBFMapper:
    """Ridge regression from (x, y) in [0,1]^2 to voice parameters
    [log_f0, loudness, z_1..z_16] over a fixed grid of Gaussian bumps.
    Incremental: accumulates X'X and X'y; solves on demand."""

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


# ── ddsp model ────────────────────────────────────────────────────────────────

def load_latest_ddsp():
    cfgs = []
    for p in glob.glob(os.path.join(PROJ, "checkpoints", "*.json")):
        with open(p) as f:
            cfg = json.load(f)
        if cfg.get("domain") == "ddsp":
            pt = os.path.join(PROJ, "checkpoints", cfg["filename"])
            if os.path.exists(pt):
                cfgs.append((os.path.getmtime(pt), cfg, pt))
    if not cfgs:
        raise RuntimeError("No DDSP checkpoint found — train one first.")
    _, cfg, pt = max(cfgs)
    m = ddsp.DDSP()
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    m.eval()
    return m, cfg["name"]


def analyze_take(audio):
    """(samples,) float32 -> (frame_times, targets (T, 18))"""
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    n_frames = max(1, wav.shape[1] // ddsp.HOP)
    p = ddsp.analyze(MODEL, wav)
    logf0 = p["f0"].clamp(min=1).log().numpy()
    targets = np.concatenate([logf0[:, None], p["loud"].numpy()[:, None],
                              p["z"].numpy()], axis=1)
    t = (np.arange(n_frames) + 0.5) * ddsp.HOP / ddsp.SR
    return t, targets


def synthesize_targets(targets):
    """(T, 18) -> (samples,) float32 via the DDSP decoder."""
    f0   = torch.tensor(np.exp(targets[:, 0]), dtype=torch.float32).clamp(60, 600)
    loud = torch.tensor(targets[:, 1], dtype=torch.float32)
    z    = torch.tensor(targets[:, 2:], dtype=torch.float32)
    wav = ddsp.synthesize_params(MODEL, {"f0": f0, "loud": loud, "z": z})
    return wav[0].numpy()


# ── capture ───────────────────────────────────────────────────────────────────

class Take:
    """Synchronized cursor + microphone capture."""

    def __init__(self, record_audio):
        self.record_audio = record_audio
        self.blocks = []
        self.cursor = []                        # (t, x, y)
        self.t0 = time.perf_counter()
        self.stream = None
        if record_audio:
            self.stream = sd.InputStream(samplerate=ddsp.SR, channels=1,
                                         callback=self._cb)
            self.stream.start()

    def _cb(self, indata, frames, t, status):
        self.blocks.append(indata[:, 0].copy())

    def tick(self, x, y):
        self.cursor.append((time.perf_counter() - self.t0, x, y))

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()
        audio = np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)
        cur = np.array(self.cursor) if self.cursor else np.zeros((0, 3))
        return audio, cur


def cursor_at(cur, times):
    """Interpolate cursor path (N,3) at given times -> (T, 2) in [0,1]^2."""
    x = np.interp(times, cur[:, 0], cur[:, 1])
    y = np.interp(times, cur[:, 0], cur[:, 2])
    return np.stack([x, y], axis=1)


# ── live engine ─────────────────────────────────────────────────────────────────

class LiveEngine:
    """Continuous audio: a producer thread renders 16 ms frames slightly ahead
    (reading the latest cursor + gate), the sound-card callback consumes them.
    Gate on = sound follows the cursor live; gate off = fade to silence."""

    AHEAD = 6          # frames of cushion (~96 ms) between producer and card

    def __init__(self, mapper):
        self.mapper = mapper
        self.synth = ddsp.StreamingSynth(MODEL)
        self.buf = deque()
        self.lock = threading.Lock()
        self.target = (0.5, 0.5, 0.0)          # x, y, gate
        self.gain = 0.0
        self.underruns = 0
        self.running = True
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.stream = sd.OutputStream(samplerate=ddsp.SR, channels=1,
                                      blocksize=ddsp.HOP, latency="low",
                                      callback=self._cb)
        self.thread.start()
        self.stream.start()

    def set(self, x, y, gate):
        self.target = (x, y, 1.0 if gate else 0.0)

    def set_mapper(self, mapper):
        self.mapper = mapper

    def _cb(self, outdata, frames, t, status):
        with self.lock:
            block = self.buf.popleft() if self.buf else None
        if block is None:
            outdata[:, 0] = 0.0
            if self.gain > 1e-3:
                self.underruns += 1
        else:
            outdata[:, 0] = block

    def _produce(self):
        while self.running:
            with self.lock:
                n = len(self.buf)
            if n >= self.AHEAD:
                time.sleep(0.002)
                continue
            x, y, gate = self.target
            new_gain = self.gain + 0.35 * (gate - self.gain)   # ~50 ms ramp
            if new_gain < 1e-3 and gate == 0.0:
                new_gain = 0.0
            if new_gain <= 0.0 or self.mapper.W is None:
                frame = np.zeros(ddsp.HOP, dtype=np.float32)
                self.synth.reset()                # fresh state on next note
            else:
                p = self.mapper.predict(np.array([[x, y]]))[0]
                f0 = float(np.clip(np.exp(p[0]), 60, 600))
                frame = self.synth.render(f0, float(p[1]), p[2:])
                ramp = np.linspace(self.gain, new_gain, ddsp.HOP, dtype=np.float32)
                frame = frame * ramp
            self.gain = new_gain
            with self.lock:
                self.buf.append(frame)

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.stream.stop(); self.stream.close()


# ── camera hand tracking (separate process) ───────────────────────────────────

class CameraProc:
    """Camera + MediaPipe run in a child process (camtrack.py) with their own
    interpreter and CPU core — no GIL contention with audio or UI. This process
    only drains small queues."""

    def __init__(self):
        import multiprocessing as mp_
        self.state_q = mp_.Queue()
        self.frame_q = mp_.Queue()
        self.stop_ev = mp_.Event()
        import camtrack
        self.proc = mp_.Process(target=camtrack.run,
                                args=(self.state_q, self.frame_q, self.stop_ev),
                                daemon=True)
        self.proc.start()
        self.state = (0.5, 0.5, None)          # x, y, gesture
        self.fps = 0.0
        self.frame_bgr = None
        self.frame_id = 0
        self.error = None

    def poll(self):
        """Drain queues; keep only the freshest state/frame."""
        while not self.state_q.empty():
            msg = self.state_q.get_nowait()
            if msg[0] == "state":
                _, x, y, g, fps = msg
                self.state = (x, y, g)
                self.fps = fps
            elif msg[0] == "error":
                self.error = msg[1]
        while not self.frame_q.empty():
            _, frame = self.frame_q.get_nowait()
            self.frame_bgr = frame
            self.frame_id += 1

    def close(self):
        self.stop_ev.set()
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()


# ── app ───────────────────────────────────────────────────────────────────────

print("loading DDSP model…", flush=True)
MODEL, MODEL_NAME = load_latest_ddsp()
print(f"model: {MODEL_NAME}", flush=True)

MAP_RES = 56

def render_map_surface(mapper):
    """Predicted pitch (hue) x loudness (brightness) over the square, or None."""
    if mapper.W is None:
        return None
    g = (np.arange(MAP_RES) + 0.5) / MAP_RES
    gx, gy = np.meshgrid(g, g)
    pred = mapper.predict(np.stack([gx.ravel(), gy.ravel()], axis=1))
    logf0 = pred[:, 0].reshape(MAP_RES, MAP_RES)
    loud  = pred[:, 1].reshape(MAP_RES, MAP_RES)
    hue = np.clip((logf0 - np.log(80)) / (np.log(500) - np.log(80)), 0, 1)
    val = np.clip((loud + 7) / 6, 0.15, 1.0)
    # simple HSV->RGB (S=0.85), vectorized
    h6 = hue * 5.0                              # blue(low) -> red(high) span
    c = val * 0.85
    xcomp = c * (1 - np.abs(h6 % 2 - 1))
    m = val - c
    r = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [c, xcomp, 0, 0, xcomp], c)
    gch = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [xcomp, c, c, xcomp, 0], 0)
    b = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [0, 0, xcomp, c, c], xcomp)
    rgb = (np.stack([b, gch, r], axis=-1) + m[..., None]) * 255   # low=blue, high=red
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).astype(np.uint8))
    return pygame.transform.smoothscale(surf, (SIZE, SIZE))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SIZE + 2 * MARGIN, SIZE + 2 * MARGIN + 60))
    pygame.display.set_caption("Gesture Instrument — demonstrate (L-drag + sing) / play (R-drag)")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    camera = None
    if "--no-camera" not in sys.argv:
        print("starting camera…", flush=True)
        camera = CameraProc()
        print("camera on", flush=True)

    mapper = RBFMapper()
    take = None
    cam_gesture = None                          # last raw gesture from camera
    cam_gesture_t = time.perf_counter()         # when it last changed
    cam_pos = (0.5, 0.5)
    mode = "idle"                               # idle | demo | trace
    status = "hold LEFT + sing to demonstrate; hold RIGHT to play"
    demo_paths, trace_path = [], []
    map_surface, show_map = None, False
    cam_driving = False                         # current take started by camera?
    playback = None
    last_live = None
    analysis_done = []                          # finished takes -> UI applies
    cam_bg, cam_bg_id = None, -1
    engine = LiveEngine(mapper)                             # {"cur": (t,x,y) array, "start": t0, "dur": s}

    def to_unit(pos):
        return (np.clip((pos[0] - MARGIN) / SIZE, 0, 1),
                np.clip((pos[1] - MARGIN) / SIZE, 0, 1))

    session_path = os.path.join(SESSION_DIR, "session_latest.npz")

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_c:
                    mapper = RBFMapper(); demo_paths = []
                    engine.set_mapper(mapper)
                    map_surface = None
                    status = "map cleared"
                elif ev.key == pygame.K_m:
                    show_map = not show_map
                elif ev.key == pygame.K_v:
                    if camera is None:
                        status = "starting camera…"
                        camera = CameraProc()
                    else:
                        camera.close()
                        camera, cam_bg, cam_gesture = None, None, None
                        status = "camera off"
                elif ev.key in (pygame.K_UP, pygame.K_DOWN):
                    factor = 0.85 if ev.key == pygame.K_UP else 1 / 0.85
                    mapper.sigma = float(np.clip(mapper.sigma * factor, 0.03, 0.6))
                    mapper.refit()
                    map_surface = render_map_surface(mapper)
                    status = f"kernel width: {mapper.sigma:.3f} (UP=sharper DOWN=smoother)"
                elif ev.key == pygame.K_s:
                    mapper.save(session_path, MODEL_NAME)
                    status = f"saved {os.path.basename(session_path)}"
                elif ev.key == pygame.K_p and last_live is not None and mapper.W is not None:
                    cur = last_live
                    dur = cur[-1, 0]
                    t = np.arange(int(dur * ddsp.SR / ddsp.HOP)) * ddsp.HOP / ddsp.SR
                    if len(t) > 2:
                        status = "synthesizing replay…"; pygame.display.flip()
                        out = synthesize_targets(mapper.predict(cursor_at(cur, t)))
                        sd.play(out, ddsp.SR)
                        playback = {"cur": cur, "start": time.perf_counter(),
                                    "dur": len(out) / ddsp.SR}
                        status = f"replaying {dur:.1f}s"
                elif ev.key == pygame.K_l and os.path.exists(session_path):
                    name = mapper.load(session_path)
                    engine.set_mapper(mapper)
                    map_surface = render_map_surface(mapper)
                    status = f"loaded session ({mapper.seconds:.0f}s, model {name})"
            elif ev.type == pygame.MOUSEBUTTONDOWN and mode == "idle":
                cam_driving = getattr(ev, "cam", False)
                if ev.button == 1:
                    mode, take = "demo", Take(record_audio=True)
                    status = "RECORDING — sing and move…"
                elif ev.button == 3:
                    mode, take = "trace", Take(record_audio=False)
                    trace_path = []
                    status = "LIVE — sound follows the cursor"
            elif ev.type == pygame.MOUSEBUTTONUP and mode != "idle":
                audio, cur = take.stop()
                if mode == "demo" and len(cur) > 5 and len(audio) > ddsp.SR // 2:
                    status = "analyzing take (background)…"

                    def _analyze(audio=audio, cur=cur):
                        t, targets = analyze_take(audio)
                        t = t[t <= cur[-1, 0]]  # audio may outlast cursor
                        xy = cursor_at(cur, t)
                        mapper.add(xy, targets[:len(t)], seconds=len(audio) / ddsp.SR)
                        analysis_done.append(cur[:, 1:])

                    threading.Thread(target=_analyze, daemon=True).start()
                elif mode == "trace" and len(cur) > 5:
                    trace_path = cur[:, 1:]
                    last_live = cur
                    status = ("no map yet — demonstrate first"
                              if mapper.W is None else
                              "released (P = clean offline replay of that path)")
                mode, take = "idle", None

        # camera gestures drive the same modes as the mouse buttons
        if camera is not None:
            camera.poll()
            if camera.error:
                status = f"camera process error: {camera.error.splitlines()[0]}"
                camera = None
            if camera is None:
                cx_, cy_, g_raw = 0.5, 0.5, None
            else:
                cx_, cy_, g_raw = camera.state
            cam_pos = (cx_, cy_)
            now = time.perf_counter()
            if g_raw != cam_gesture:
                cam_gesture, cam_gesture_t = g_raw, now
            want = {"pinch": "demo", "two": "trace"}.get(cam_gesture, "idle")
            # quick to engage, slow to release: brief flickers don't end a take
            hold = 0.12 if want != "idle" else 0.35
            # camera may start takes when idle, but only ends takes it started
            # (a mouse-held take is never interrupted by the hand going idle)
            if (want != mode and now - cam_gesture_t >= hold
                    and (mode == "idle" or cam_driving)):
                if mode != "idle" and take is not None:
                    # close the current take exactly like a button release
                    pygame.event.post(pygame.event.Event(
                        pygame.MOUSEBUTTONUP, button=1 if mode == "demo" else 3))
                if want == "demo":
                    pygame.event.post(pygame.event.Event(
                        pygame.MOUSEBUTTONDOWN, button=1, cam=True))
                elif want == "trace":
                    pygame.event.post(pygame.event.Event(
                        pygame.MOUSEBUTTONDOWN, button=3, cam=True))

        while analysis_done:                    # apply finished analyses (UI thread)
            demo_paths.append(analysis_done.pop(0))
            map_surface = render_map_surface(mapper)
            status = (f"map updated: {mapper.seconds:.1f}s total "
                      f"({len(demo_paths)} takes)")

        pos = pygame.mouse.get_pos()
        ux, uy = to_unit(pos)
        # hand position drives camera-started takes; the mouse drives its own
        if camera is not None and (cam_driving if mode != "idle"
                                   else cam_gesture is not None):
            ux, uy = cam_pos
        engine.set(ux, uy, gate=(mode == "trace"))
        if mode != "idle" and take is not None:
            take.tick(ux, uy)

        # ── draw ──
        def decimate(pts, cap=300):
            """Cap polyline cost: at most `cap` points regardless of length."""
            step = max(1, len(pts) // cap)
            return pts[::step]

        screen.fill((16, 16, 30))
        if camera is not None and camera.frame_bgr is not None                 and camera.frame_id != cam_bg_id:
            f = camera.frame_bgr[:, :, ::-1]                # BGR -> RGB, small
            small = pygame.surfarray.make_surface(f.swapaxes(0, 1).copy())
            cam_bg = pygame.transform.scale(small, (SIZE, SIZE))
            cam_bg.fill((80, 80, 90), special_flags=pygame.BLEND_MULT)  # dim
            cam_bg_id = camera.frame_id
        if cam_bg is not None:
            screen.blit(cam_bg, (MARGIN, MARGIN))
        if show_map and map_surface is not None:
            map_surface.set_alpha(150 if cam_bg is not None else 255)
            screen.blit(map_surface, (MARGIN, MARGIN))
        pygame.draw.rect(screen, (60, 60, 100),
                         (MARGIN, MARGIN, SIZE, SIZE), width=1)
        for path in demo_paths:                 # past demonstrations
            pts = [(MARGIN + p[0] * SIZE, MARGIN + p[1] * SIZE)
                   for p in decimate(path)]
            if len(pts) > 1:
                pygame.draw.lines(screen, (130, 190, 130), False, pts, 2)
        if mode == "demo" and take and len(take.cursor) > 1:
            pts = [(MARGIN + c[1] * SIZE, MARGIN + c[2] * SIZE)
                   for c in decimate(take.cursor)]
            pygame.draw.lines(screen, (220, 90, 90), False, pts, 2)
        # playing (trace mode) intentionally leaves no trace on screen

        if playback is not None:
            el = time.perf_counter() - playback["start"]
            if el > playback["dur"]:
                playback = None
            else:
                cur_p = playback["cur"]
                px = np.interp(el, cur_p[:, 0], cur_p[:, 1])
                py = np.interp(el, cur_p[:, 0], cur_p[:, 2])
                pygame.draw.circle(screen, (255, 235, 120),
                                   (MARGIN + px * SIZE, MARGIN + py * SIZE), 9)

        color = {"idle": (150, 150, 170), "demo": (230, 90, 90),
                 "trace": (90, 200, 130)}[mode]
        if camera is not None:
            hp = (MARGIN + cam_pos[0] * SIZE, MARGIN + cam_pos[1] * SIZE)
            pygame.draw.circle(screen, color, hp, 10, width=2)
            pygame.draw.circle(screen, color, hp, 2)
        else:
            pygame.draw.circle(screen, color, pos, 6, width=0 if mode != "idle" else 1)
        screen.blit(font.render(status, True, (200, 200, 210)),
                    (MARGIN, SIZE + MARGIN + 12))
        screen.blit(font.render(
            f"map: {mapper.seconds:.1f}s | kernel {mapper.sigma:.2f} | "
            f"underruns {engine.underruns} | "
            + (f"cam {camera.fps:.0f}fps [{cam_gesture or 'idle'}] | " if camera else "") +
            f"S save L load C clear M map V camera P replay UP/DOWN detail Q quit",
            True, (120, 120, 140)), (MARGIN, SIZE + MARGIN + 34))
        pygame.display.flip()
        clock.tick(CURSOR_HZ)

    engine.close()
    if camera is not None:
        camera.close()
    pygame.quit()


if __name__ == "__main__":
    main()
