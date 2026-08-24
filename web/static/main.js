/* Gesture instrument — browser front end.
 *
 * Two modalities, chosen by the mode toggle and kept side by side:
 *   x,y    mouse on the pad, or a tracked hand; pinch records, two fingers play
 *   body   four full-body features; hands together for 3s toggles record/play
 *
 * The body features are scale- and translation-invariant, so it doesn't matter
 * how far from the camera you stand or where in frame you are.
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
// The two modalities. Each keeps its own map on the server and its own drawn
// paths here, so switching back and forth never destroys the other's work.
const SPACE_FEATURES = { xy: PAD_FEATURES, body: BODY_FEATURES };
let space = "xy";               // "xy" | "body"
let features = PAD_FEATURES;    // whichever space is currently driving the map
const GDIM = () => features.length;
const demoPathsBy = { xy: [], body: [] };

let node = null;                // AudioWorkletNode
let audioCtx = null;
let state = { seconds: 0, model: "…", mode: "aniso", sigma: 0.15, sigma_par: 0.04,
              points: null, dim: 2 };
let demoPaths = demoPathsBy.xy; // [[p_1..p_D],...] per finished take, current space
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

// ── modality toggle ─────────────────────────────────────────────────────────
const spaceBtn = document.getElementById("space");
const legendEl = document.getElementById("legend");
const LEGENDS = {
  xy: '<i class="r">left-drag / pinch</i> sing to teach &nbsp; '
    + '<i class="g">right-drag / two fingers</i> play',
  body: '<i class="r">hands together 3s</i> record &nbsp; '
      + '<i class="r">again</i> play &nbsp; <i class="g">hold 6s</i> stop',
};

// ── kernel controls ─────────────────────────────────────────────────────────
const modeEl = document.getElementById("mode");
const sigxEl = document.getElementById("sigx"), sigxV = document.getElementById("sigxv");
const sigpEl = document.getElementById("sigp"), sigpV = document.getElementById("sigpv");
const reachEl = document.getElementById("reach"), reachV = document.getElementById("reachv");
const breathEl = document.getElementById("breath"), breathV = document.getElementById("breathv");

function syncControls() {
  modeEl.value = state.mode;
  sigxEl.value = state.sigma;
  sigpEl.value = state.sigma_par;
  sigxV.textContent = state.sigma.toFixed(3);
  sigpV.textContent = state.sigma_par.toFixed(3);
  if (reachEl) {
    reachEl.value = state.reach ?? 1;
    reachV.textContent = (state.reach ?? 1).toFixed(1) + "×";
  }
  const aniso = state.mode === "aniso";
  sigpEl.disabled = !aniso;
  document.getElementById("sigpRow").style.opacity = aniso ? 1 : 0.35;
}

async function clearMap() {
  const fd = new FormData();
  fd.append("space", space);
  state = await fetch("/api/clear", { method: "POST", body: fd }).then(r => r.json());
  demoPaths.length = 0;
  if (node) sendMap();
  hud();
}

/* Switch modality. Each space has its own mapper on the server, so this only
 * has to fetch the other one's state and swap the drawn paths — nothing is
 * cleared, and whatever you demonstrated in the space you're leaving is still
 * there when you come back. */
async function setSpace(next) {
  if (space === next) return;
  if (mode !== "idle") endTake();
  space = next;
  features = SPACE_FEATURES[space];
  demoPaths = demoPathsBy[space];
  pos = new Array(GDIM()).fill(0.5);
  holdSince = 0; holdArmed = null;
  state = await fetch(`/api/state?space=${space}`).then(r => r.json());
  if (node) sendMap();
  syncControls();
  hud();
  setStatus(space === "body"
    ? "whole body — bring your hands together for 3s to start recording"
    : "x,y — hold LEFT + sing to demonstrate · hold RIGHT to play");
  spaceBtn.textContent = space === "body" ? "mode: whole body" : "mode: x,y";
  legendEl.innerHTML = LEGENDS[space];
}

spaceBtn.addEventListener("click", async () => {
  await setSpace(space === "xy" ? "body" : "xy");
  if (space === "body" && !camOn) setStatus("whole body needs the camera — turn it on");
});

let configBusy = false;
async function postConfig(fields) {
  if (configBusy) return;
  configBusy = true;
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) fd.append(k, String(v));
  fd.append("space", space);
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
// Guarded: a browser holding a cached index.html without these controls used
// to throw here, and a throw at top level kills every declaration below it —
// the whole instrument died over two optional sliders.
if (reachEl) {
  reachEl.addEventListener("input", () => { reachV.textContent = (+reachEl.value).toFixed(1) + "×"; });
  reachEl.addEventListener("change", () => postConfig({ reach: reachEl.value }));
}
// breath never reaches the server: it trims the synth, not the map
if (breathEl) {
  breathEl.addEventListener("input", () => {
    breathV.textContent = (+breathEl.value).toFixed(2);
    if (node) node.port.postMessage({ type: "breath", value: +breathEl.value });
  });
}
sigpEl.addEventListener("change", () => postConfig({ sigma_par: sigpEl.value }));

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  setStatus("loading decoder…");
  const [manifest, weightsBuf, st] = await Promise.all([
    fetch("/api/decoder/manifest").then(r => r.json()),
    fetch("/api/decoder/weights").then(r => r.arrayBuffer()),
    fetch(`/api/state?space=${space}`).then(r => r.json()),
  ]);
  state = st;
  pos = new Array(GDIM()).fill(0.5);

  audioCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
  // cache-bust: addModule ignores hard-refresh, so pin the worklet version
  await audioCtx.audioWorklet.addModule("/static/worklet.js?v=8");
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
                          dim: state.dim || GDIM(), reach: state.reach ?? 1,
                          sigma: state.sigma, sigmaPar: state.sigma_par });
}

/* Mic level, and the threshold above which singing counts as material.
 *
 * The hold gesture gives us a calibration window for free: you cannot be
 * singing into the microphone while holding your hands together for three
 * seconds to start a take, so whatever the mic hears then is the room — its
 * fans, its traffic, your breathing. Measuring that and putting the threshold
 * a fixed margin above it means the threshold is known *before* recording
 * begins, which is what makes live feedback possible at all: the take-level
 * analysis can only ever tell you afterwards.
 *
 * Levels are in log-RMS, the same units the server analyses takes in, so the
 * number sent with the take is directly comparable to what it measures.       */
const CAL_MIN_SAMPLES = 12;     // ~0.8 s of hold before the floor is trusted
const DB_PER_LN = 8.686;        // 20/ln(10): log-RMS units -> dB, for the slider
let micLoud = -12;              // current mic level
let roomFloor = null;           // measured during a hold (authoritative)
let floorTracked = -9.0;        // continuous follower, used when there is no hold
let floorMargin = 2.5;          // how far above the floor counts, set by the slider
let calSamples = [];            // level samples collected during this hold

// Falls instantly to a new quiet level and climbs back very slowly (~1 log unit
// per two minutes), so it settles on the room rather than chasing your voice.
// It exists so the meter means something before any hold has happened at all,
// and in x,y mode, which has no hold gesture to calibrate from.
function trackFloor(v) {
  if (v < floorTracked) floorTracked = v;
  else floorTracked = Math.min(-2.0, floorTracked + 0.0008);
}

const micFloor = () => (roomFloor !== null ? roomFloor : floorTracked);
const micThreshold = () => micFloor() + floorMargin;

function onWorklet(m) {
  if (m.type === "level") {
    micLoud = Math.log(m.rms + 1e-5);          // matches ddsp.extract_loudness
    trackFloor(micLoud);
    if (holdSince) calSamples.push(micLoud);   // hands together = room only
  } else if (m.type === "take") {
    uploadTake(new Float32Array(m.audio));
  }
}

/* Turn a completed hold into a threshold. Uses a low percentile rather than
 * the mean so one cough during the hold can't lift the floor and start
 * discarding real singing. */
function calibrateFloor() {
  if (calSamples.length < CAL_MIN_SAMPLES) { calSamples = []; return; }
  const s = calSamples.slice().sort((a, b) => a - b);
  roomFloor = s[Math.floor(s.length * 0.25)];
  floorTracked = roomFloor;                    // the measured one is better
  calSamples = [];
}

// is what the mic hears right now loud enough to be learned?
const capturing = () => micLoud > micThreshold();

// Placed here deliberately: this reads floorMargin and DB_PER_LN, which are
// declared just above. Wired up higher with the other sliders it would throw
// at load — and a top-level throw takes every declaration below it with it.
const gateEl = document.getElementById("gate"), gateV = document.getElementById("gatev");
if (gateEl) {
  floorMargin = (+gateEl.value) / DB_PER_LN;
  gateEl.addEventListener("input", () => {
    gateV.textContent = `${gateEl.value} dB`;
    floorMargin = (+gateEl.value) / DB_PER_LN;   // takes effect on the meter at once
  });
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
  fd.append("space", space);
  // the threshold measured from the room during the hold, so the server drops
  // exactly what the live indicator said it would
  fd.append("loud_threshold", String(micThreshold()));
  try {
    const r = await fetch("/api/take", { method: "POST", body: fd }).then(r => r.json());
    if (r.ok) {
      state = r;
      demoPaths.push(r.path);
      sendMap();
      const drop = r.dropped_seconds
        ? `, ${r.dropped_seconds.toFixed(1)}s of silence dropped` : "";
      setStatus(`map updated: ${r.seconds.toFixed(1)}s total `
              + `(${demoPaths.length} takes${drop})`);
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
    // A take is stopped by the hold gesture, which takes 3s — so the tail of
    // the cursor is the gesture itself, not playing. Drop everything from the
    // moment the hold began; the server clips the audio to the cursor's last
    // timestamp, so the sung tail goes with it.
    if (holdSince) {
      const cutoff = holdSince - takeT0 / 1000;
      const kept = cursor.filter((row) => row[0] <= cutoff);
      if (kept.length >= 6) cursor = kept;
    }
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
let camPos = { x: 0.5, y: 0.5 };  // tracked hand position (x,y mode)
let camGesture = null;          // "pinch" | "two" | null, with hysteresis
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

// ── x,y mode gestures: pinch / two fingers ──────────────────────────────────
// pinch enters at 0.35 / exits at 0.50; "two" relaxes once active (hysteresis,
// same constants as camtrack.py) so landmark jitter doesn't flicker takes
function classify(lm, prev) {
  const p = (i) => lm[i];
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const wrist = p(0);
  const ext = (tip, pip, slack) => d(p(tip), wrist) > slack * d(p(pip), wrist);
  const scale = d(p(9), wrist) + 1e-6;
  if (d(p(4), p(8)) / scale < (prev === "pinch" ? 0.50 : 0.35)) {
    return { g: "pinch", x: (p(4).x + p(8).x) / 2, y: (p(4).y + p(8).y) / 2 };
  }
  const two = prev === "two"
    ? ext(8, 6, 1.02) && ext(12, 10, 1.02)
    : ext(8, 6, 1.12) && ext(12, 10, 1.12) && !ext(16, 14, 1.12) && !ext(20, 18, 1.12);
  if (two) return { g: "two", x: p(8).x, y: p(8).y };
  return { g: null, x: p(8).x, y: p(8).y };
}

// ── body mode gesture: hands brought together ───────────────────────────────
/* Just the distance between the two wrists, in torso lengths.
 *
 * This replaced crossed forearms, which the pose model could not read
 * reliably: crossing puts one forearm in front of the other, and occlusion is
 * exactly where landmark confidence collapses — the detector was least sure
 * precisely while the gesture was being made. Here the arms stay separate and
 * plainly visible, and wrist-to-wrist distance is also the one relationship
 * the four features don't measure, so it can't be confused with playing.
 *
 * Timing (measured on a continuous hold, acted on when you separate your
 * hands, so the gesture pose itself never lands in a recording):
 *   held >= 3s, then release -> toggle record <-> play
 *   held >= 6s, then release -> stop everything
 *   released before 3s       -> nothing happened                            */
const HOLD_TOGGLE_S = 3.0;
const HOLD_STOP_S   = 6.0;
const HOLD_GAP_S    = 0.25;     // tracking dropout tolerated without resetting
const TOGETHER_ENTER = 0.25;    // wrists within this (x torso) starts the hold
const TOGETHER_EXIT  = 0.40;    // and must pass this to end it (hysteresis)
let holdSince = 0;              // when the current continuous hold began
let holdLastSeen = 0;
let holdArmed = null;           // "toggle" | "stop" — what release will do
let handsGap = null;            // latest wrist separation, for the overlay

function handsTogether(lm) {
  const need = [PL.LWR, PL.RWR, PL.LSH, PL.RSH, PL.LHIP, PL.RHIP];
  for (const i of need)
    if (!lm[i] || (lm[i].visibility !== undefined && lm[i].visibility < MIN_VIS))
      return false;
  const torso = dist(mid(lm[PL.LSH], lm[PL.RSH]), mid(lm[PL.LHIP], lm[PL.RHIP]));
  if (!(torso > 1e-4)) return false;
  handsGap = dist(lm[PL.LWR], lm[PL.RWR]) / torso;
  // once holding, tolerate drifting a bit further apart before letting go
  return holdSince ? handsGap < TOGETHER_EXIT : handsGap < TOGETHER_ENTER;
}

function camFrame(now) {
  if (!camOn) return;
  if (now - lastInferenceRun >= camInferenceInterval && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    lastInferenceRun = now;
    // only the model this modality needs runs — halves the per-frame cost
    const started = performance.now();
    const res = space === "xy" ? landmarker.detectForVideo(video, now) : null;
    const pres = space === "body" && poseLandmarker
               ? poseLandmarker.detectForVideo(video, now) : null;
    const elapsed = performance.now() - started;
    camInferenceInterval = Math.max(CAM_MIN_INTERVAL_MS, elapsed * CAM_ADAPTIVE_HEADROOM);
    const t = now / 1000;

    if (space === "xy") frameXY(res, t);
    else frameBody(pres, t);
  }
  requestAnimationFrame(camFrame);
}

// x,y: hand position drives the point, pinch/two-fingers drive the takes
function frameXY(res, t) {
  if (res && res.landmarks && res.landmarks.length) {
    const c = classify(res.landmarks[0], camGesture);
    // map through the same center square crop the preview is drawn with
    const vw = video.videoWidth, vh = video.videoHeight;
    const side = Math.min(vw, vh);
    const cx = (c.x * vw - (vw - side) / 2) / side;
    const cy = (c.y * vh - (vh - side) / 2) / side;
    camPos = { x: Math.min(1, Math.max(0, 1 - cx)),      // mirror
               y: Math.min(1, Math.max(0, cy)) };
    if (c.g !== camGesture) { camGesture = c.g; camGestureT = t; }
    lastSeen = t;
  } else if (t - lastSeen > 0.3 && camGesture !== null) {
    camGesture = null; camGestureT = t;                  // ride through dropouts
  }

  // camera may start takes when idle, but only ends takes it started
  const want = camGesture === "pinch" ? "demo"
             : camGesture === "two" ? "trace" : "idle";
  const hold = want !== "idle" ? 0.12 : 0.35;            // quick engage, slow release
  if (want !== mode && t - camGestureT >= hold && (mode === "idle" || camDriving)) {
    if (mode !== "idle") endTake();
    if (want !== "idle") { pos = [camPos.x, camPos.y]; startTake(want, true); }
  }
  if (camGesture !== null && (mode === "idle" || camDriving))
    moveTo([camPos.x, camPos.y]);
}

// body: the four features drive the point, the hold gesture drives the takes
function frameBody(pres, t) {
  const lm = pres && pres.landmarks && pres.landmarks.length ? pres.landmarks[0] : null;

  // gesture point, held at its last value through a brief tracking dropout so
  // the sound doesn't jump
  if (lm) {
    const f = bodyFeatures(lm);
    if (f) { bodyVec = f; lastBodySeen = t; }
  }
  if (t - lastBodySeen > 0.5) { bodyVec = null; bodyMarks = null; }

  if (!lm) handsGap = null;
  const together = lm ? handsTogether(lm) : false;
  if (together) {
    if (!holdSince) holdSince = t;
    holdLastSeen = t;
    const held = t - holdSince;
    holdArmed = held >= HOLD_STOP_S ? "stop"
              : held >= HOLD_TOGGLE_S ? "toggle" : null;
  } else if (holdSince && t - holdLastSeen > HOLD_GAP_S) {
    // the hold just ended — everything the mic heard during it was the room,
    // so turn it into the threshold before the take it is about to start
    calibrateFloor();
    applyHold(holdArmed);                      // acts on release, not on the hold
    holdSince = 0; holdArmed = null;
  }

  // while holding, the arms are making a gesture rather than playing, so the
  // point is frozen — otherwise the gesture would smear across the whole map
  if (bodyVec && !holdSince && (mode === "idle" || camDriving))
    moveTo(bodyVec.slice());
}

function applyHold(action) {
  if (!action) return;
  if (action === "stop") {
    if (mode !== "idle") endTake();
    setStatus("stopped — hands together 3s to record again");
    return;
  }
  // toggle: idle -> record, record -> play, play -> record
  const next = mode === "demo" ? "trace" : "demo";
  if (mode !== "idle") endTake();
  if (!bodyVec) { setStatus("step into frame — body not visible"); return; }
  pos = bodyVec.slice();
  startTake(next, true);
}

async function toggleCamera() {
  if (camOn) {
    camOn = false;
    if (mode !== "idle" && camDriving) endTake();
    camStream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    camGesture = null;
    bodyVec = bodyMarks = null;
    holdSince = 0; holdArmed = null;
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
  // the preview underneath is at full brightness now, so the panel needs its
  // own backing or the bars vanish against a light shirt
  ctx2d.fillStyle = "rgba(8,8,16,0.62)";
  ctx2d.fillRect(pad - 8, pad - 6, bw + 100, features.length * gap + 10);
  features.forEach((f, j) => {
    const y = pad + 6 + j * gap;
    ctx2d.fillStyle = "rgba(225,225,240,0.9)";
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

/* One ring, filling once over the whole 6s — the arc never restarts, so there
 * is only ever a single timer to read. A notch on the ring marks the 3s point:
 * arc short of the notch means nothing happens yet, past it means releasing
 * switches, and all the way round means releasing stops. */
function drawHoldProgress() {
  if (!holdSince) return;
  const held = performance.now() / 1000 - holdSince;
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2, r = 54;
  const TAU = 2 * Math.PI, top = -Math.PI / 2;
  const frac = Math.min(1, held / HOLD_STOP_S);
  const notch = HOLD_TOGGLE_S / HOLD_STOP_S;       // where "switch" becomes armed
  const past = held >= HOLD_TOGGLE_S;

  ctx2d.save();
  ctx2d.lineWidth = 9;
  ctx2d.strokeStyle = "rgba(0,0,0,0.5)";           // track
  ctx2d.beginPath(); ctx2d.arc(cx, cy, r, 0, TAU); ctx2d.stroke();

  ctx2d.strokeStyle = past ? "#e0a54a" : "#5ac8e0";
  ctx2d.lineCap = "round";
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, r, top, top + frac * TAU);
  ctx2d.stroke();

  // the 3s threshold, drawn on the ring itself
  const a = top + notch * TAU;
  ctx2d.lineWidth = 3;
  ctx2d.strokeStyle = "rgba(255,255,255,0.85)";
  ctx2d.beginPath();
  ctx2d.moveTo(cx + Math.cos(a) * (r - 9), cy + Math.sin(a) * (r - 9));
  ctx2d.lineTo(cx + Math.cos(a) * (r + 9), cy + Math.sin(a) * (r + 9));
  ctx2d.stroke();

  const next = mode === "demo" ? "play" : "record";
  const outlined = (txt, y, font) => {
    ctx2d.font = font;
    ctx2d.lineWidth = 4;
    ctx2d.strokeStyle = "rgba(0,0,0,0.8)";
    ctx2d.strokeText(txt, cx, y);
    ctx2d.fillStyle = "#fff";
    ctx2d.fillText(txt, cx, y);
  };
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "middle";
  outlined(held.toFixed(1) + "s", cy, "bold 17px ui-monospace, monospace");
  outlined(past ? `release → stop` : `release → ${next}`, cy + r + 22,
           "bold 13px ui-monospace, monospace");
  ctx2d.restore();
}

/* Recording vs playing, unmissable: a full-canvas border in the mode colour
 * plus a label. Recording also pulses, because that is the state you must not
 * be in by accident. */
function drawModeBanner() {
  const w = canvas.width, h = canvas.height;
  const rec = mode === "demo", play = mode === "trace";
  if (!rec && !play) {
    if (space === "body" && !holdSince) {
      ctx2d.save();
      ctx2d.font = "bold 13px ui-monospace, monospace";
      ctx2d.textAlign = "center";
      ctx2d.lineWidth = 4;
      ctx2d.strokeStyle = "rgba(0,0,0,0.75)";
      const msg = "bring your hands together for 3s to start recording";
      ctx2d.strokeText(msg, w / 2, h - 18);
      ctx2d.fillStyle = "rgba(225,225,240,0.95)";
      ctx2d.fillText(msg, w / 2, h - 18);
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

/* Transport state in the corner, in the usual symbols: filled circle for
 * recording, triangle for playing, square for stopped. Always drawn, in both
 * modalities, so there is one fixed place to look for "what is it doing".
 * Shapes rather than glyphs — no font dependency, and crisp at any size. */
function drawTransport() {
  const pw = 132, ph = 34, x = canvas.width - pw - 12, y = 12;
  const rec = mode === "demo", play = mode === "trace";
  // While recording, red only means "this is going into the map". Below the
  // threshold the take is still running but nothing is being learned, so it
  // goes hollow and amber rather than lying to you with a solid red dot.
  const live = rec && capturing();
  const color = rec ? (live ? "#e65c5c" : "#c8912f") : play ? "#5ac882" : "#9696aa";
  const label = rec ? (live ? "REC" : "too soft") : play ? "PLAY" : "STOP";
  const sx = x + 22, sy = y + ph / 2, r = 9;

  ctx2d.save();
  ctx2d.fillStyle = "rgba(8,8,16,0.66)";
  ctx2d.fillRect(x, y, pw, ph);
  ctx2d.globalAlpha = rec && live
    ? 0.6 + 0.4 * Math.abs(Math.sin(performance.now() / 380)) : 1;
  ctx2d.strokeStyle = ctx2d.fillStyle = color;
  ctx2d.lineWidth = 2.5;
  ctx2d.beginPath();
  if (rec) {
    ctx2d.arc(sx, sy, r, 0, 2 * Math.PI);
  } else if (play) {
    ctx2d.moveTo(sx - r * 0.75, sy - r);
    ctx2d.lineTo(sx + r, sy);
    ctx2d.lineTo(sx - r * 0.75, sy + r);
    ctx2d.closePath();
  } else {
    ctx2d.rect(sx - r * 0.82, sy - r * 0.82, r * 1.64, r * 1.64);
  }
  if (rec && !live) ctx2d.stroke(); else ctx2d.fill();
  ctx2d.font = "bold 12px ui-monospace, monospace";
  ctx2d.textAlign = "left";
  ctx2d.textBaseline = "middle";
  ctx2d.fillText(label, sx + r + 11, sy);
  ctx2d.restore();

  drawMicMeter(x, y + ph + 6, pw);
}

/* Mic level with the threshold marked, so "too soft" is actionable — you can
 * see how far under you are rather than just being told no. */
function drawMicMeter(x, y, w) {
  if (!audioCtx) return;
  const h = 8, LO = -9.5, HI = -1.5;            // useful log-RMS range
  const at = (v) => Math.min(1, Math.max(0, (v - LO) / (HI - LO)));
  ctx2d.save();
  ctx2d.fillStyle = "rgba(8,8,16,0.66)";
  ctx2d.fillRect(x, y, w, h);
  ctx2d.fillStyle = capturing() ? "#e65c5c" : "rgba(200,200,220,0.45)";
  ctx2d.fillRect(x, y, at(micLoud) * w, h);
  ctx2d.fillStyle = "rgba(120,200,255,0.55)";   // the measured room floor
  ctx2d.fillRect(x + at(micFloor()) * w - 1, y, 2, h);
  ctx2d.fillStyle = "#ffffff";                  // the gate, moved by the slider
  ctx2d.fillRect(x + at(micThreshold()) * w - 1, y - 2, 2, h + 4);
  ctx2d.restore();
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx2d.fillStyle = "#0b0b14";
  ctx2d.fillRect(0, 0, w, h);
  if (camOn && video.readyState >= 2) {        // full-brightness mirrored preview
    const vw = video.videoWidth, vh = video.videoHeight;
    const side = Math.min(vw, vh);             // center square crop: no stretch
    ctx2d.save();
    ctx2d.scale(-1, 1);
    ctx2d.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side,
                    -w, 0, w, h);
    ctx2d.restore();
  }
  const colors = { idle: "#9696aa", demo: "#e65c5c", trace: "#5ac882" };

  // a path through 4-D body space isn't a curve on this canvas, so the drawn
  // trace only means something in the 2-D pad space; the bars carry the rest
  if (space === "xy") {
    for (const p of demoPaths) drawPath(decimate(p), "rgba(130,190,130,0.9)", 2);
    if (mode === "demo") drawPath(decimate(livePath), "#dc5a5a", 2);
    if (camOn) {                               // tracked hand, and which gesture
      ctx2d.strokeStyle = colors[camGesture === "pinch" ? "demo"
                               : camGesture === "two" ? "trace" : "idle"];
      ctx2d.lineWidth = 2;
      ctx2d.beginPath();
      ctx2d.arc(camPos.x * w, camPos.y * h, 12, 0, 2 * Math.PI);
      ctx2d.stroke();
    }
    ctx2d.fillStyle = colors[mode];
    ctx2d.beginPath();
    ctx2d.arc(pos[0] * w, pos[1] * h, mode === "idle" ? 4 : 7, 0, 2 * Math.PI);
    ctx2d.fill();
  } else if (camOn) {
    drawBodyMarks();
  }

  drawFeatureBars();
  drawTransport();
  if (camOn) {
    drawModeBanner();
    if (space === "body") drawHoldProgress();
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

document.getElementById("start").addEventListener("click", async function () {
  this.remove();
  await boot();
}, { once: true });
