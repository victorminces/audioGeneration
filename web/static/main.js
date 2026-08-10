/* Gesture instrument — browser front end.
 * LEFT-drag / either hand closed  = demonstrate (audio + path go to the server)
 * RIGHT-drag / both hands open    = play (kernel map -> decoder -> synth, local)
 *
 * The gesture space is D-dimensional. With the camera off it is the 2D pad
 * (mouse x, y). With the camera on it is the four full-body features below,
 * each scale- and translation-invariant so it doesn't matter how far from the
 * camera you stand or where in frame you are.
 */

const canvas = document.getElementById("pad");
const ctx2d = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const hudEl = document.getElementById("hud");

// ── gesture space ────────────────────────────────────────────────────────────
// Body features, in the order they are sent as the gesture point. Each is
// normalized to [0,1] by lo/hi so one kernel width is meaningful across all of
// them (the kernel is isotropic in feature units — it has no idea these are
// different quantities).
const BODY_FEATURES = [
  { key: "extL", label: "L arm",  lo: 0.20, hi: 1.40, color: "#e0854a" },
  { key: "extR", label: "R arm",  lo: 0.20, hi: 1.40, color: "#4ab0e0" },
  { key: "hgtL", label: "L hand", lo: -0.60, hi: 1.90, color: "#e0c04a" },
  { key: "hgtR", label: "R hand", lo: -0.60, hi: 1.90, color: "#7ad46a" },
];
const PAD_FEATURES = [
  { key: "x", label: "pad x", lo: 0, hi: 1, color: "#9aa0c0" },
  { key: "y", label: "pad y", lo: 0, hi: 1, color: "#9aa0c0" },
];
let features = PAD_FEATURES;    // whichever space is currently driving the map
const GDIM = () => features.length;

let node = null;                // AudioWorkletNode
let audioCtx = null;
let state = { seconds: 0, model: "…", mode: "aniso", sigma: 0.15, sigma_par: 0.04,
              points: null, dim: 2 };
let demoPaths = [];             // [[p_1..p_D],...] per finished take
let mode = "idle";              // idle | demo | trace
let cursor = [];                // [t, p_1..p_D] during a take
let takeT0 = 0;
let livePath = [];              // current demo trace for drawing
let pos = [0.5, 0.5];           // current gesture point, length D
let pendingCursor = null;       // cursor of the take whose audio we await

function setStatus(s) { statusEl.textContent = s; }

function hud() {
  hudEl.textContent =
    `model ${state.model}  ·  map ${state.seconds.toFixed(1)}s  ·  ` +
    (audioCtx ? `${audioCtx.sampleRate} Hz` : "audio off");
}

// ── kernel controls ─────────────────────────────────────────────────────────
const modeEl = document.getElementById("mode");
const sigxEl = document.getElementById("sigx"), sigxV = document.getElementById("sigxv");
const sigpEl = document.getElementById("sigp"), sigpV = document.getElementById("sigpv");

function syncControls() {
  modeEl.value = state.mode;
  sigxEl.value = state.sigma;
  sigpEl.value = state.sigma_par;
  sigxV.textContent = state.sigma.toFixed(3);
  sigpV.textContent = state.sigma_par.toFixed(3);
  const aniso = state.mode === "aniso";
  sigpEl.disabled = !aniso;
  document.getElementById("sigpRow").style.opacity = aniso ? 1 : 0.35;
}

async function clearMap(dim) {
  const fd = new FormData();
  fd.append("dim", String(dim === undefined ? GDIM() : dim));
  state = await fetch("/api/clear", { method: "POST", body: fd }).then(r => r.json());
  demoPaths = [];
  if (node) sendMap();
  hud();
}

/* Switch which set of features is the gesture space. The demonstrations
 * already in the map were recorded as points in the old space and have no
 * meaning in the new one (a 2-D pad point is not a 4-D body pose), so the map
 * is cleared rather than reinterpreted. */
async function setSpace(next) {
  if (features === next) return;
  features = next;
  const D = GDIM();
  const p = new Array(D).fill(0.5);             // new dims start centred
  for (let i = 0; i < Math.min(pos.length, D); i++) p[i] = pos[i];
  pos = p;
  if (state.dim !== D) {
    await clearMap(D);
    setStatus(`gesture space is now ${features.map(f => f.label).join(" · ")}`
              + " — previous demonstrations cleared");
  }
}

let configBusy = false;
async function postConfig(fields) {
  if (configBusy) return;
  configBusy = true;
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) fd.append(k, String(v));
  try {
    state = await fetch("/api/config", { method: "POST", body: fd }).then(r => r.json());
    sendMap();
  } catch (err) {
    setStatus("server error: " + err.message);
  }
  configBusy = false;
  syncControls();
  hud();
}

modeEl.addEventListener("change", () => postConfig({ mode: modeEl.value }));
sigxEl.addEventListener("input", () => { sigxV.textContent = (+sigxEl.value).toFixed(3); });
sigpEl.addEventListener("input", () => { sigpV.textContent = (+sigpEl.value).toFixed(3); });
sigxEl.addEventListener("change", () => postConfig({ sigma: sigxEl.value }));
sigpEl.addEventListener("change", () => postConfig({ sigma_par: sigpEl.value }));

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  setStatus("loading decoder…");
  const [manifest, weightsBuf, st] = await Promise.all([
    fetch("/api/decoder/manifest").then(r => r.json()),
    fetch("/api/decoder/weights").then(r => r.arrayBuffer()),
    fetch("/api/state").then(r => r.json()),
  ]);
  state = st;
  // Adopt whatever space the server's map is already in. Without this a reload
  // (or a fresh tab against a running server) would sit in 2-D pad mode while
  // the map holds 4-D body samples, and the points sent would not line up.
  features = state.dim === BODY_FEATURES.length ? BODY_FEATURES : PAD_FEATURES;
  pos = new Array(GDIM()).fill(0.5);

  audioCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
  // cache-bust: addModule ignores hard-refresh, so pin the worklet version
  await audioCtx.audioWorklet.addModule("/static/worklet.js?v=4");
  node = new AudioWorkletNode(audioCtx, "instrument", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { manifest, weights: weightsBuf, constants: manifest.constants },
  });
  node.port.onmessage = (e) => onWorklet(e.data);
  node.connect(audioCtx.destination);

  sendMap();
  syncControls();
  hud();
  setStatus("allow the microphone to demonstrate…");

  try {
    const mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    audioCtx.createMediaStreamSource(mic).connect(node);
    setStatus("hold LEFT + sing to demonstrate · hold RIGHT to play");
  } catch (err) {
    setStatus("mic denied — playing works, demonstrating won't. " + err.name);
  }
}

function sendMap() {
  node.port.postMessage({ type: "map", points: state.points, mode: state.mode,
                          dim: state.dim || GDIM(),
                          sigma: state.sigma, sigmaPar: state.sigma_par });
}

function onWorklet(m) {
  if (m.type === "take") uploadTake(new Float32Array(m.audio));
}

// ── takes ────────────────────────────────────────────────────────────────────
async function uploadTake(audio) {
  const cur = pendingCursor || [];
  pendingCursor = null;
  if (audio.length < audioCtx.sampleRate / 2 || cur.length < 6) {
    setStatus("take too short — hold longer and sing");
    return;
  }
  setStatus("analyzing take…");
  const fd = new FormData();
  fd.append("audio", new Blob([audio.buffer], { type: "application/octet-stream" }));
  fd.append("cursor", JSON.stringify(cur));
  fd.append("sr", String(audioCtx.sampleRate));
  try {
    const r = await fetch("/api/take", { method: "POST", body: fd }).then(r => r.json());
    if (r.ok) {
      state = r;
      demoPaths.push(r.path);
      sendMap();
      setStatus(`map updated: ${r.seconds.toFixed(1)}s total (${demoPaths.length} takes)`);
    } else {
      setStatus(r.reason || "take rejected");
    }
  } catch (err) {
    setStatus("server error: " + err.message);
  }
  hud();
}

// ── input ────────────────────────────────────────────────────────────────────
// The mouse only ever supplies two numbers. In the 2D pad space that is the
// whole point; in body space it steers the first two features (the arm
// extensions) and leaves the rest where the body last put them, which keeps
// the pad usable for testing without standing in front of the camera.
function toUnit(ev) {
  const r = canvas.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
  const y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
  const p = pos.slice();
  p[0] = x;
  if (p.length > 1) p[1] = y;
  return p;
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

let camDriving = false;        // current take started by the camera?

function startTake(kind, byCam) {
  if (!node || mode !== "idle") return;
  if (audioCtx.state !== "running") audioCtx.resume();
  camDriving = byCam;
  takeT0 = performance.now();
  cursor = [[0, ...pos]];
  mode = kind;
  if (kind === "demo") {
    livePath = [pos.slice()];
    node.port.postMessage({ type: "record", on: true });
    setStatus("RECORDING — sing and move…");
  } else {
    node.port.postMessage({ type: "gate", on: true });
    setStatus(state.points ? "LIVE — sound follows the cursor"
                      : "no map yet — demonstrate first (left-drag + sing)");
  }
}

function endTake() {
  if (mode === "demo") {
    pendingCursor = cursor;
    node.port.postMessage({ type: "record", on: false });   // worklet posts audio back
    livePath = [];
  } else if (mode === "trace") {
    node.port.postMessage({ type: "gate", on: false });
    setStatus("released");
  }
  mode = "idle";
  camDriving = false;
}

function moveTo(p) {
  pos = p;
  if (!node) return;
  node.port.postMessage({ type: "pos", p: pos });
  if (mode !== "idle") {
    cursor.push([(performance.now() - takeT0) / 1000, ...pos]);
    if (mode === "demo") livePath.push(pos.slice());
  }
}

canvas.addEventListener("pointerdown", (ev) => {
  if (mode !== "idle") return;
  canvas.setPointerCapture(ev.pointerId);
  pos = toUnit(ev);
  if (node) node.port.postMessage({ type: "pos", p: pos });
  if (ev.button === 0) startTake("demo", false);
  else if (ev.button === 2) startTake("trace", false);
});

canvas.addEventListener("pointermove", (ev) => {
  // the mouse never steers a camera-started take
  if (mode !== "idle" && camDriving) return;
  moveTo(toUnit(ev));
});

canvas.addEventListener("pointerup", () => { if (mode !== "idle" && !camDriving) endTake(); });
canvas.addEventListener("pointercancel", () => { if (mode !== "idle" && !camDriving) endTake(); });

// ── camera tracking (MediaPipe Tasks, fully in-browser) ─────────────────────
// Two models run per frame: hands for the pinch/two-finger gestures that start
// and stop takes, pose for the body features that drive the map.
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
// Served from Google's CDN like the tasks-vision bundle above; the hand model
// is local (/api/hand_model) only because it was already vendored in the repo.
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const video = document.getElementById("cam");
const camBtn = document.getElementById("camera");
let landmarker = null;
let poseLandmarker = null;
let camOn = false;
let camStream = null;
let camHands = [];              // [{x, y, closed, openness}] — for the overlay
let camGesture = null;          // "record" | "play" | null (from the hand states)
let camGestureT = 0;            // when it last changed
let lastSeen = 0;               // last time a hand was detected
let lastBodySeen = 0;           // last time the body was reliably measured
let lastVideoTime = -1;

// Inference throttling (same approach as mflow's useHandTracking): never run
// detectForVideo faster than a ~30fps ceiling — pinch/two gestures don't
// benefit from more — and back off proportionally on slow devices so a weak
// CPU degrades to a lower tracking rate instead of pinning the main thread
// (which would otherwise compete with drawing and network calls here).
const CAM_MIN_INTERVAL_MS = 33;
const CAM_ADAPTIVE_HEADROOM = 1.5;
let camInferenceInterval = CAM_MIN_INTERVAL_MS;
let lastInferenceRun = 0;

// ── body features ───────────────────────────────────────────────────────────
// MediaPipe pose landmarks are named anatomically (LEFT_* is the person's own
// left), so these stay correct even though the preview is mirrored.
const PL = { LSH: 11, RSH: 12, LEL: 13, REL: 14, LWR: 15, RWR: 16, LHIP: 23, RHIP: 24 };
const MIN_VIS = 0.5;            // below this a landmark is a guess, not a measurement

let bodyVec = null;             // [extL, extR, hgtL, hgtR] in [0,1], or null
let bodyMarks = null;           // landmark geometry kept for the overlay

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const norm01 = (v, lo, hi) => Math.min(1, Math.max(0, (v - lo) / (hi - lo)));

/* Landmarks -> the four features, or null when the body isn't reliably visible.
 *
 * Everything is divided by the shoulder-to-hip distance, which is what makes
 * the features invariant to how far away you stand, and measured relative to
 * body points rather than the frame, which makes them invariant to where in
 * frame you are. Raw units are therefore "torso lengths".
 *   extension = shoulder-to-wrist distance   (folded ~0.2, fully reaching ~1.4)
 *   height    = how far the wrist is above the hips (hips 0, shoulders 1) */
function bodyFeatures(lm) {
  const need = [PL.LSH, PL.RSH, PL.LHIP, PL.RHIP, PL.LWR, PL.RWR];
  for (const i of need)
    if (!lm[i] || (lm[i].visibility !== undefined && lm[i].visibility < MIN_VIS))
      return null;

  const shC = mid(lm[PL.LSH], lm[PL.RSH]);
  const hipC = mid(lm[PL.LHIP], lm[PL.RHIP]);
  const torso = dist(shC, hipC);
  if (!(torso > 1e-4)) return null;             // degenerate (facing straight down?)

  const extLraw = dist(lm[PL.LWR], lm[PL.LSH]) / torso;
  const extRraw = dist(lm[PL.RWR], lm[PL.RSH]) / torso;
  // screen y grows downward, so hip minus wrist is "how far above the hips"
  const hgtLraw = (hipC.y - lm[PL.LWR].y) / torso;
  const hgtRraw = (hipC.y - lm[PL.RWR].y) / torso;
  const raw = [extLraw, extRraw, hgtLraw, hgtRraw];

  bodyMarks = { shC, hipC, torso, raw,
                lsh: lm[PL.LSH], rsh: lm[PL.RSH], lwr: lm[PL.LWR], rwr: lm[PL.RWR],
                lel: lm[PL.LEL], rel: lm[PL.REL] };
  return BODY_FEATURES.map((f, i) => norm01(raw[i], f.lo, f.hi));
}

/* Open vs closed hand.
 *
 * Pinch needed the thumb and index tips resolved separately, which they are
 * not once the whole body is in frame and the hand is a few dozen pixels
 * across. This measures the grossest thing a hand does instead: how far the
 * fingertips sit from the wrist, in palm lengths. A flat hand puts them about
 * 2.2 palms out, a fist about 1.1, and that gap survives landmark noise at
 * body distance.
 *
 * Hysteresis is keyed by handedness rather than detection order, so the two
 * hands swapping places in the result list can't smear their states together. */
const CLOSE_ENTER = 1.50;       // below this an open hand becomes closed
const OPEN_EXIT   = 1.75;       // above this a closed hand opens again
const closedState = { Left: false, Right: false };

function handOpenness(lm) {
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const wrist = lm[0];
  const palm = d(wrist, lm[9]) + 1e-6;              // wrist -> middle knuckle
  const tips = [8, 12, 16, 20];                     // index, middle, ring, pinky
  let s = 0;
  for (const t of tips) s += d(lm[t], wrist);
  return s / tips.length / palm;
}

function classifyHand(lm, handed) {
  const openness = handOpenness(lm);
  const wasClosed = closedState[handed] || false;
  const closed = wasClosed ? openness < OPEN_EXIT : openness < CLOSE_ENTER;
  closedState[handed] = closed;
  return { closed, openness, x: lm[9].x, y: lm[9].y };   // knuckle = steadier than a tip
}

function camFrame(now) {
  if (!camOn) return;
  if (now - lastInferenceRun >= camInferenceInterval && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    lastInferenceRun = now;
    const started = performance.now();
    const res = landmarker.detectForVideo(video, now);
    const pres = poseLandmarker ? poseLandmarker.detectForVideo(video, now) : null;
    const elapsed = performance.now() - started;
    // both models are inside the measurement, so the backoff accounts for the
    // real per-frame cost of running them together
    camInferenceInterval = Math.max(CAM_MIN_INTERVAL_MS, elapsed * CAM_ADAPTIVE_HEADROOM);
    const t = now / 1000;

    // body -> gesture point. Held at its last value while the body is out of
    // view, so a momentary tracking dropout doesn't jump the sound.
    if (pres && pres.landmarks && pres.landmarks.length) {
      const f = bodyFeatures(pres.landmarks[0]);
      if (f) { bodyVec = f; lastBodySeen = t; }
    }
    if (t - lastBodySeen > 0.5) { bodyVec = null; bodyMarks = null; }

    if (res.landmarks && res.landmarks.length) {
      const vw = video.videoWidth, vh = video.videoHeight;
      const side = Math.min(vw, vh);
      camHands = res.landmarks.map((lm, i) => {
        const handed = res.handedness?.[i]?.[0]?.categoryName === "Left" ? "Right" : "Left";
        const c = classifyHand(lm, handed);
        // map through the same center square crop the preview is drawn with
        const cx = (c.x * vw - (vw - side) / 2) / side;
        const cy = (c.y * vh - (vh - side) / 2) / side;
        return { ...c, handed,
                 x: Math.min(1, Math.max(0, 1 - cx)),     // mirror
                 y: Math.min(1, Math.max(0, cy)) };
      });
      // either hand closed -> record; hands up and all open -> play
      const g = camHands.some((h) => h.closed) ? "record" : "play";
      if (g !== camGesture) { camGesture = g; camGestureT = t; }
      lastSeen = t;
    } else if (t - lastSeen > 0.3 && camGesture !== null) {
      camGesture = null; camGestureT = t;                // ride through dropouts
      camHands = [];
    }

    // camera may start takes when idle, but only ends takes it started
    const want = camGesture === "record" ? "demo"
               : camGesture === "play" ? "trace" : "idle";
    const hold = want !== "idle" ? 0.12 : 0.35;          // quick engage, slow release
    // no body reading means no gesture point to record or play at
    if (want !== mode && t - camGestureT >= hold && (mode === "idle" || camDriving)
        && (want === "idle" || bodyVec)) {
      if (mode !== "idle") endTake();
      if (want !== "idle") { pos = bodyVec.slice(); startTake(want, true); }
    }
    if (camGesture !== null && bodyVec && (mode === "idle" || camDriving))
      moveTo(bodyVec.slice());
  }
  requestAnimationFrame(camFrame);
}

async function toggleCamera() {
  if (camOn) {
    camOn = false;
    if (mode !== "idle" && camDriving) endTake();
    camStream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    camGesture = null;
    bodyVec = bodyMarks = null;
    await setSpace(PAD_FEATURES);
    camBtn.textContent = "camera: off";
    return;
  }
  try {
    camBtn.textContent = "camera: …";
    if (!landmarker) {
      const mp = await import(`${MP_CDN}/vision_bundle.mjs`);
      const files = await mp.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
      const opts = (delegate) => ({
        baseOptions: { modelAssetPath: "/api/hand_model", delegate },
        numHands: 2, runningMode: "VIDEO",      // open/closed is read per hand
        minHandDetectionConfidence: 0.6, minTrackingConfidence: 0.5,
      });
      // Prefer the GPU delegate (runs off the main thread, frees it for
      // drawing/network work); fall back to CPU when the device/driver can't
      // provide it, so tracking still works rather than failing outright.
      try {
        landmarker = await mp.HandLandmarker.createFromOptions(files, opts("GPU"));
      } catch (gpuErr) {
        landmarker = await mp.HandLandmarker.createFromOptions(files, opts("CPU"));
      }
    }
    if (!poseLandmarker) {
      const mp = await import(`${MP_CDN}/vision_bundle.mjs`);   // cached after the first load
      const files = await mp.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
      const popts = (delegate) => ({
        baseOptions: { modelAssetPath: POSE_MODEL, delegate },
        numPoses: 1, runningMode: "VIDEO",
        minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      try {
        poseLandmarker = await mp.PoseLandmarker.createFromOptions(files, popts("GPU"));
      } catch (gpuErr) {
        poseLandmarker = await mp.PoseLandmarker.createFromOptions(files, popts("CPU"));
      }
    }
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    video.srcObject = camStream;
    await video.play();
    camOn = true;
    await setSpace(BODY_FEATURES);
    camBtn.textContent = "camera: on";
    requestAnimationFrame(camFrame);
  } catch (err) {
    camBtn.textContent = "camera: off";
    setStatus("camera failed: " + err.message);
  }
}

camBtn.addEventListener("click", toggleCamera);

// UP = sharper (smaller across-width), DOWN = smoother; the sliders do the rest
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") return;
  ev.preventDefault();
  if (!node) return;
  const factor = ev.key === "ArrowUp" ? 0.85 : 1 / 0.85;
  postConfig({ sigma: state.sigma * factor });
});

document.getElementById("clear").addEventListener("click", async () => {
  await clearMap();
  setStatus("map cleared");
});

// ── draw ─────────────────────────────────────────────────────────────────────
function decimate(pts, cap = 300) {
  const step = Math.max(1, Math.floor(pts.length / cap));
  return pts.filter((_, i) => i % step === 0);
}

function drawPath(pts, color, width) {
  if (pts.length < 2) return;
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth = width;
  ctx2d.lineJoin = ctx2d.lineCap = "round";
  ctx2d.beginPath();
  const w = canvas.width, h = canvas.height;
  ctx2d.moveTo(pts[0][0] * w, pts[0][1] * h);
  for (const p of pts) ctx2d.lineTo(p[0] * w, p[1] * h);
  ctx2d.stroke();
}

// normalized landmark -> canvas pixels, through the same centre-square crop and
// mirror the preview is drawn with, so overlays land on the actual body
function toCanvas(lm) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh);
  const cx = (lm.x * vw - (vw - side) / 2) / side;
  const cy = (lm.y * vh - (vh - side) / 2) / side;
  return [(1 - cx) * canvas.width, cy * canvas.height];
}

function line(a, b, color, width, dash) {
  ctx2d.save();
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth = width;
  ctx2d.setLineDash(dash || []);
  ctx2d.lineCap = "round";
  ctx2d.beginPath();
  ctx2d.moveTo(a[0], a[1]);
  ctx2d.lineTo(b[0], b[1]);
  ctx2d.stroke();
  ctx2d.restore();
}

function dot(p, color, r) {
  ctx2d.fillStyle = color;
  ctx2d.beginPath();
  ctx2d.arc(p[0], p[1], r, 0, 2 * Math.PI);
  ctx2d.fill();
}

/* Draw what each feature actually measures, on the body itself:
 *   - the torso line every measurement is divided by (the scale reference)
 *   - a shoulder-to-wrist line per arm: that length IS the extension feature
 *   - the hip-level line heights are measured from, and a drop from each wrist
 *     down to it: that vertical gap IS the height feature                    */
function drawBodyMarks() {
  const m = bodyMarks;
  if (!m) return;
  const shC = toCanvas(m.shC), hipC = toCanvas(m.hipC);
  const lsh = toCanvas(m.lsh), rsh = toCanvas(m.rsh);
  const lwr = toCanvas(m.lwr), rwr = toCanvas(m.rwr);
  const lel = toCanvas(m.lel), rel = toCanvas(m.rel);
  const [cL, cR, cHL, cHR] = BODY_FEATURES.map((f) => f.color);

  // hip-level reference: where "height 0" is
  line([0, hipC[1]], [canvas.width, hipC[1]], "rgba(255,255,255,0.18)", 1, [4, 6]);
  // scale reference
  line(shC, hipC, "rgba(255,255,255,0.5)", 2, [2, 4]);
  dot(shC, "rgba(255,255,255,0.7)", 3);
  dot(hipC, "rgba(255,255,255,0.7)", 3);

  // arms: faint anatomical path, then the measured chord on top
  line(lsh, lel, "rgba(255,255,255,0.22)", 2);
  line(lel, lwr, "rgba(255,255,255,0.22)", 2);
  line(rsh, rel, "rgba(255,255,255,0.22)", 2);
  line(rel, rwr, "rgba(255,255,255,0.22)", 2);
  line(lsh, lwr, cL, 3);
  line(rsh, rwr, cR, 3);

  // heights: drop each wrist to the hip line
  line(lwr, [lwr[0], hipC[1]], cHL, 2, [3, 4]);
  line(rwr, [rwr[0], hipC[1]], cHR, 2, [3, 4]);

  for (const [p, c] of [[lsh, cL], [rsh, cR], [lwr, cHL], [rwr, cHR]]) dot(p, c, 5);
}

// min/max of each dimension over the demonstrated samples — "where the map
// actually has data", recomputed only when the sample set changes
let coverCache = { pts: null, ranges: null };
function demoRanges() {
  if (coverCache.pts === state.points) return coverCache.ranges;
  let r = null;
  if (state.points && state.points.length) {
    const D = state.dim || GDIM();
    r = [];
    for (let j = 0; j < D; j++) {
      let lo = Infinity, hi = -Infinity;
      for (const row of state.points) {
        if (row[j] < lo) lo = row[j];
        if (row[j] > hi) hi = row[j];
      }
      r.push([lo, hi]);
    }
  }
  coverCache = { pts: state.points, ranges: r };
  return r;
}

/* One bar per feature: the shaded band is the range the demonstrations cover,
 * the bright tick is where you are now. Outside the band the kernel has
 * nothing to average and the instrument fades toward silence. */
function drawFeatureBars() {
  const pad = 12, bw = Math.min(190, canvas.width * 0.42), bh = 9, gap = 20;
  const ranges = demoRanges();
  ctx2d.save();
  ctx2d.font = "11px ui-monospace, monospace";
  ctx2d.textBaseline = "middle";
  features.forEach((f, j) => {
    const y = pad + 6 + j * gap;
    ctx2d.fillStyle = "rgba(200,200,220,0.75)";
    ctx2d.textAlign = "left";
    ctx2d.fillText(f.label, pad, y);
    const x0 = pad + 46;
    ctx2d.fillStyle = "rgba(255,255,255,0.10)";          // track
    ctx2d.fillRect(x0, y - bh / 2, bw, bh);
    if (ranges && ranges[j]) {                            // demonstrated coverage
      const [lo, hi] = ranges[j];
      ctx2d.fillStyle = "rgba(255,255,255,0.20)";
      ctx2d.fillRect(x0 + lo * bw, y - bh / 2, Math.max(1, (hi - lo) * bw), bh);
    }
    const v = pos[j];
    if (v !== undefined) {                                // current value
      ctx2d.fillStyle = f.color;
      ctx2d.fillRect(x0 + Math.min(1, Math.max(0, v)) * bw - 1.5, y - bh / 2 - 2, 3, bh + 4);
      ctx2d.fillStyle = "rgba(200,200,220,0.6)";
      ctx2d.textAlign = "left";
      ctx2d.fillText(v.toFixed(2), x0 + bw + 8, y);
    }
  });
  if (camOn && !bodyVec) {
    ctx2d.fillStyle = "rgba(230,120,120,0.9)";
    ctx2d.textAlign = "left";
    ctx2d.fillText("body not visible — step back so hips and shoulders are in frame",
                   pad, pad + 6 + features.length * gap + 4);
  }
  ctx2d.restore();
}

/* Each hand as a filled disc when closed, a ring when open — the same
 * distinction the gesture is made of, so you can see what the tracker thinks
 * your hands are doing rather than guessing from the sound. */
function drawHands() {
  const w = canvas.width, h = canvas.height;
  for (const hnd of camHands) {
    const x = hnd.x * w, y = hnd.y * h;
    ctx2d.lineWidth = 3;
    ctx2d.strokeStyle = hnd.closed ? "#e65c5c" : "#5ac882";
    ctx2d.beginPath();
    ctx2d.arc(x, y, 16, 0, 2 * Math.PI);
    ctx2d.stroke();
    if (hnd.closed) {
      ctx2d.fillStyle = "rgba(230,92,92,0.55)";
      ctx2d.fill();
    }
    ctx2d.font = "10px ui-monospace, monospace";
    ctx2d.textAlign = "center";
    ctx2d.fillStyle = "rgba(220,220,235,0.85)";
    ctx2d.fillText(hnd.closed ? "closed" : "open", x, y + 30);
  }
}

/* Recording vs playing, unmissable: a full-canvas border in the mode colour
 * plus a label. Recording also pulses, because that is the state you must not
 * be in by accident. */
function drawModeBanner() {
  const w = canvas.width, h = canvas.height;
  const rec = mode === "demo", play = mode === "trace";
  if (!rec && !play) {
    if (camHands.length === 0) {
      ctx2d.save();
      ctx2d.font = "13px ui-monospace, monospace";
      ctx2d.textAlign = "center";
      ctx2d.fillStyle = "rgba(190,190,210,0.7)";
      ctx2d.fillText("show your hands — close one to record, open both to play",
                     w / 2, h - 18);
      ctx2d.restore();
    }
    return;
  }
  const pulse = rec ? 0.55 + 0.45 * Math.abs(Math.sin(performance.now() / 380)) : 1;
  const color = rec ? "230,92,92" : "90,200,130";

  ctx2d.save();
  ctx2d.strokeStyle = `rgba(${color},${pulse})`;
  ctx2d.lineWidth = 8;
  ctx2d.strokeRect(4, 4, w - 8, h - 8);

  const label = rec ? "● RECORDING — keep singing" : "▶ PLAYING";
  ctx2d.font = "bold 20px ui-monospace, monospace";
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "middle";
  const tw = ctx2d.measureText(label).width;
  ctx2d.fillStyle = `rgba(${color},${0.85 * pulse})`;
  ctx2d.fillRect(w / 2 - tw / 2 - 14, h - 54, tw + 28, 32);
  ctx2d.fillStyle = "#0b0b14";
  ctx2d.fillText(label, w / 2, h - 38);
  ctx2d.restore();
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx2d.fillStyle = "#0b0b14";
  ctx2d.fillRect(0, 0, w, h);
  if (camOn && video.readyState >= 2) {        // dimmed, mirrored preview
    const vw = video.videoWidth, vh = video.videoHeight;
    const side = Math.min(vw, vh);             // center square crop: no stretch
    ctx2d.save();
    ctx2d.globalAlpha = 0.25;
    ctx2d.scale(-1, 1);
    ctx2d.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side,
                    -w, 0, w, h);
    ctx2d.restore();
  }
  const colors = { idle: "#9696aa", demo: "#e65c5c", trace: "#5ac882" };

  // a path through 4-D body space isn't a curve on this canvas, so the drawn
  // trace only means something in the 2-D pad space; the bars carry the rest
  if (GDIM() === 2) {
    for (const p of demoPaths) drawPath(decimate(p), "rgba(130,190,130,0.9)", 2);
    if (mode === "demo") drawPath(decimate(livePath), "#dc5a5a", 2);
    ctx2d.fillStyle = colors[mode];
    ctx2d.beginPath();
    ctx2d.arc(pos[0] * w, pos[1] * h, mode === "idle" ? 4 : 7, 0, 2 * Math.PI);
    ctx2d.fill();
  } else if (camOn) {
    drawBodyMarks();
  }

  if (camOn) drawHands();
  drawFeatureBars();
  if (camOn) drawModeBanner();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

document.getElementById("start").addEventListener("click", async function () {
  this.remove();
  await boot();
}, { once: true });
