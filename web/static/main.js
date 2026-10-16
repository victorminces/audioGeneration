/* Gesture instrument — browser front end.
 * LEFT-drag + sing = demonstrate (audio + path go to the server, W comes back)
 * RIGHT-drag       = play (all local: RBF -> decoder -> synth in the worklet)
 */

const canvas = document.getElementById("pad");
const ctx2d = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const hudEl = document.getElementById("hud");

let node = null;                // AudioWorkletNode
let audioCtx = null;
let state = { seconds: 0, model: "…", mode: "aniso", sigma: 0.15, sigma_par: 0.04, points: null,
              face_points: null, face_scale: null, face_sigma: 1.0 };
let demoPaths = [];             // [[x,y],...] per finished take
let mode = "idle";              // idle | demo | trace
let cursor = [];                // [t, x, y] during a take
let takeT0 = 0;
let livePath = [];              // current demo trace for drawing
let pos = { x: 0.5, y: 0.5 };
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
const fsigEl = document.getElementById("fsig"), fsigV = document.getElementById("fsigv");

function syncControls() {
  modeEl.value = state.mode;
  sigxEl.value = state.sigma;
  sigpEl.value = state.sigma_par;
  sigxV.textContent = state.sigma.toFixed(3);
  sigpV.textContent = state.sigma_par.toFixed(3);
  fsigEl.value = state.face_sigma;
  fsigV.textContent = state.face_sigma.toFixed(2);
  const aniso = state.mode === "aniso";
  sigpEl.disabled = !aniso;
  document.getElementById("sigpRow").style.opacity = aniso ? 1 : 0.35;
}

let configBusy = false;
async function postConfig(fields) {
  if (configBusy) return;
  configBusy = true;
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) fd.append(k, String(v));
  try {
    const r = await fetch("/api/config", { method: "POST", body: fd }).then(r => r.json());
    state = { ...state, ...r };                // reply leaves out the face map
    sendMap(false);
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
fsigEl.addEventListener("input", () => { fsigV.textContent = (+fsigEl.value).toFixed(2); });
fsigEl.addEventListener("change", () => postConfig({ face_sigma: fsigEl.value }));

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  setStatus("loading decoder…");
  const [manifest, weightsBuf, st] = await Promise.all([
    fetch("/api/decoder/manifest").then(r => r.json()),
    fetch("/api/decoder/weights").then(r => r.arrayBuffer()),
    fetch("/api/state").then(r => r.json()),
  ]);
  state = st;

  audioCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
  // cache-bust: addModule ignores hard-refresh, so pin the worklet version
  await audioCtx.audioWorklet.addModule("/static/worklet.js?v=6");
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

// withFace=false skips the (large) face map when it hasn't changed
function sendMap(withFace = true) {
  const m = { type: "map", points: state.points, mode: state.mode,
              sigma: state.sigma, sigmaPar: state.sigma_par, faceSigma: state.face_sigma };
  if (withFace) { m.facePoints = state.face_points; m.faceScale = state.face_scale; }
  node.port.postMessage(m);
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
      sendMap();
      if (r.face_take) {
        setStatus(`face sound learned: ${r.seconds.toFixed(1)}s taught in total`);
      } else {
        demoPaths.push(r.path);
        setStatus(`map updated: ${r.seconds.toFixed(1)}s total (${demoPaths.length} takes)`);
      }
    } else {
      setStatus(r.reason || "take rejected");
    }
  } catch (err) {
    setStatus("server error: " + err.message);
  }
  hud();
}

// ── input ────────────────────────────────────────────────────────────────────
function toUnit(ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
    y: Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height)),
  };
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

let camDriving = false;        // current take started by the camera?

function startTake(kind, byCam) {
  if (!node || mode !== "idle") return;
  if (audioCtx.state !== "running") audioCtx.resume();
  camDriving = byCam;
  takeT0 = performance.now();
  cursor = [[0, pos.x, pos.y, ...faceRow()]];
  mode = kind;
  const byFace = camKind === "face" && byCam;
  node.port.postMessage({ type: "face", on: byFace });   // mouse play uses the pad map
  if (kind === "demo") {
    livePath = [[pos.x, pos.y]];
    node.port.postMessage({ type: "record", on: true });
    setStatus(camKind === "face" ? "RECORDING — make your sound; your face is the key…"
                                 : "RECORDING — sing and move…");
  } else {
    node.port.postMessage({ type: "gate", on: true });
    if (byFace) return;                      // face play is always on: stay quiet
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
    if (!(camDriving && camKind === "face")) setStatus("released");
  }
  mode = "idle";
  camDriving = false;
}

// face takes carry the face vector after x, y (the server learns from both)
function faceRow() {
  return camKind === "face" && camDriving && faceVec ? faceVec : [];
}

function moveTo(p) {
  pos = p;
  if (!node) return;
  node.port.postMessage({ type: "pos", ...pos });
  if (mode !== "idle") {
    cursor.push([(performance.now() - takeT0) / 1000, pos.x, pos.y, ...faceRow()]);
    if (mode === "demo") livePath.push([pos.x, pos.y]);
  }
}

canvas.addEventListener("pointerdown", (ev) => {
  if (mode === "trace" && camDriving && camKind === "face") endTake();   // mouse wins
  if (mode !== "idle") return;
  canvas.setPointerCapture(ev.pointerId);
  pos = toUnit(ev);
  if (node) node.port.postMessage({ type: "pos", ...pos });
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
// Two camera modes share one video stream:
//  hand  pinch = sing to teach, two fingers = play (index fingertip steers)
//  face  the whole mouth shape (a vector of ~19 measurements) picks the
//        sound: hold SPACE + make any sound to teach it; otherwise it plays
//        whatever was taught with the most similar face. The pad shows a 2D
//        view of that shape (a vowel chart) so you can see where you are.
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const video = document.getElementById("cam");
const camSel = document.getElementById("camera");
const legendEl = document.getElementById("legend");
const LEGEND_POINTER = legendEl.innerHTML;
const LEGEND_FACE = '<i class="r">hold SPACE + sing</i> teach &nbsp; ' +
                    '<i class="g">make the face</i> play';
let mp = null, visionFiles = null;
let handLandmarker = null;
let faceLandmarker = null;
let camKind = "off";            // off | hand | face
let camOn = false;
let camLoop = false;            // a camFrame rAF loop is running
let camStream = null;
let camPos = { x: 0.5, y: 0.5 };
let camGesture = null;          // raw (with hysteresis) gesture
let camGestureT = 0;            // when it last changed
let lastSeen = 0;               // last time a hand/face was detected
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

const clamp01 = (v) => Math.min(1, Math.max(0, v));

// normalized video coords -> unit pad coords, through the same mirrored
// center square crop the preview is drawn with
function camToUnit(p) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh);
  return { x: 1 - (p.x * vw - (vw - side) / 2) / side,
           y: (p.y * vh - (vh - side) / 2) / side };
}

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

function handStep(res, t) {
  if (res.landmarks && res.landmarks.length) {
    const c = classify(res.landmarks[0], camGesture);
    const u = camToUnit(c);
    camPos = { x: clamp01(u.x), y: clamp01(u.y) };
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
    if (want !== "idle") { pos = camPos; startTake(want, true); }
  }
  if (camGesture !== null && (mode === "idle" || camDriving)) moveTo(camPos);
}

// ── face: mouth shape -> sound ──────────────────────────────────────────────
// What picks the sound is FACE_VECTOR: every mouth/jaw blendshape score
// (0..1, left/right pairs averaged) plus two mesh measurements — the inner-lip
// gap and the mouth width. Sounds that look different (hum vs. "ah" vs. "sss"
// vs. a lip buzz) can be told apart; sounds that differ only inside the mouth
// (tongue, voicing: "t"/"k", "p"/"b") can't — the camera can't see them.
//
// For display only, the pad shows a 2D vowel-chart view of the same face:
//   x = rounding minus spreading (pucker/funnel vs. smile/stretch)
//   y = jawOpen
// The optional calibration (C) records a relaxed face, "ee", "oo" and "ah"
// to pin that chart to this particular face; it doesn't change the sound.
const CAL_STEPS = [
  { key: "neutral", say: "relax your face, mouth closed" },
  { key: "ee", say: 'say "eeee" (big smile)' },
  { key: "oo", say: 'say "oooo" (round lips)' },
  { key: "ah", say: 'say "aaah" (open wide)' },
];
const CAL_SEC = 2.0, CAL_SETTLE = 0.7;       // per step; ignore the reaction time
const FACE_SMOOTH = 0.5;                     // one-pole smoothing per tracked frame
const DEFAULT_CAL = {                        // rough typical values until calibrated
  neutral: { jaw: 0.02, shape: 0.0, open: 0.01 },
  ee: { jaw: 0.08, shape: -0.35, open: 0.10 },
  oo: { jaw: 0.10, shape: 0.45, open: 0.25 },
  ah: { jaw: 0.55, shape: 0.0, open: 0.45 },
};
let faceCal = DEFAULT_CAL;
let faceCalibrated = false;
let calib = null;               // { step, t0, acc, result } while calibrating
let face = null;                // latest features + landmarks, null when lost
let faceVec = null;             // smoothed FACE_VECTOR, sent to the worklet

const FACE_VECTOR = [
  ["jawOpen"], ["jawForward"], ["mouthClose"], ["mouthFunnel"], ["mouthPucker"],
  ["mouthSmileLeft", "mouthSmileRight"], ["mouthFrownLeft", "mouthFrownRight"],
  ["mouthDimpleLeft", "mouthDimpleRight"], ["mouthStretchLeft", "mouthStretchRight"],
  ["mouthRollLower"], ["mouthRollUpper"], ["mouthShrugLower"], ["mouthShrugUpper"],
  ["mouthPressLeft", "mouthPressRight"], ["mouthLowerDownLeft", "mouthLowerDownRight"],
  ["mouthUpperUpLeft", "mouthUpperUpRight"], ["cheekPuff"],
];


function faceFeatures(cats, lm) {
  const s = {};
  for (const c of cats) s[c.categoryName] = c.score;
  const spread = (s.mouthSmileLeft + s.mouthSmileRight +
                  s.mouthStretchLeft + s.mouthStretchRight) / 4;
  const round = (s.mouthPucker + s.mouthFunnel) / 2;
  const vw = video.videoWidth, vh = video.videoHeight;      // measure in pixels
  const d = (a, b) => Math.hypot((lm[a].x - lm[b].x) * vw, (lm[a].y - lm[b].y) * vh);
  const open = d(13, 14) / (d(61, 291) + 1e-6);             // lip gap / mouth width
  const width = d(61, 291) / (d(234, 454) + 1e-6);          // mouth width / face width
  const vec = FACE_VECTOR.map((names) =>
    names.reduce((a, n) => a + (s[n] || 0), 0) / names.length);
  vec.push(open, width);
  return { lm, bs: s, jaw: s.jawOpen, spread, round, shape: round - spread, open, vec };
}

function facePos(f) {
  const c = faceCal;
  const sSpan = Math.max(0.1, c.oo.shape - c.ee.shape);
  const jSpan = Math.max(0.15, c.ah.jaw - c.neutral.jaw);
  return { x: clamp01(0.15 + 0.7 * (f.shape - c.ee.shape) / sSpan),
           y: clamp01(0.1 + 0.8 * (f.jaw - c.neutral.jaw) / jSpan) };
}

function startCalibration() {
  if (mode !== "idle" && camDriving) endTake();
  calib = { step: 0, t0: null, acc: [], result: {} };
}

function calibStep(t) {
  if (calib.t0 === null) calib.t0 = t;
  const el = t - calib.t0;
  if (face && el > CAL_SETTLE) calib.acc.push(face);
  if (el < CAL_SEC) return;
  if (calib.acc.length < 5) {                // face not seen: repeat this step
    calib.t0 = t; calib.acc = [];
    return;
  }
  const n = calib.acc.length;
  const mean = (k) => calib.acc.reduce((a, f) => a + f[k], 0) / n;
  calib.result[CAL_STEPS[calib.step].key] =
    { jaw: mean("jaw"), shape: mean("shape"), open: mean("open") };
  calib.step++; calib.t0 = t; calib.acc = [];
  if (calib.step < CAL_STEPS.length) return;

  const r = calib.result;
  calib = null;
  if (r.oo.shape - r.ee.shape < 0.1 || r.ah.jaw - r.neutral.jaw < 0.15) {
    setStatus("couldn't tell the shapes apart — exaggerate them and press C to retry");
    return;
  }
  const again = faceCalibrated;
  faceCal = r;
  faceCalibrated = true;
  setStatus(again && state.face_points
    ? "recalibrated — the chart moved; the sounds still follow your face"
    : "chart calibrated — hold SPACE and make a sound to teach it");
}

function faceStep(res, t) {
  if (res.faceLandmarks && res.faceLandmarks.length &&
      res.faceBlendshapes && res.faceBlendshapes.length) {
    const reacquired = t - lastSeen > 0.5;                  // no glide on reacquire
    face = faceFeatures(res.faceBlendshapes[0].categories, res.faceLandmarks[0]);
    const p = facePos(face);
    const a = reacquired ? 1 : FACE_SMOOTH;
    camPos = { x: camPos.x + a * (p.x - camPos.x), y: camPos.y + a * (p.y - camPos.y) };
    faceVec = faceVec && !reacquired
      ? faceVec.map((v, j) => v + a * (face.vec[j] - v)) : face.vec.slice();
    lastSeen = t;
  } else if (t - lastSeen > 0.5) {
    face = null;
  }

  if (calib) { calibStep(t); return; }       // silent while calibrating
  if (!face) {
    faceVec = null;
    if (mode === "trace" && camDriving) endTake();
    return;
  }
  if (node) node.port.postMessage({ type: "feat", f: faceVec });
  // the face plays whenever it's not teaching (and the mouse isn't busy)
  if (mode === "idle") startTake("trace", true);
  if (!camDriving) return;
  moveTo(camPos);
}

// Self-healing: the loop must never die. A tracker error (typically the GPU
// context being lost after a while on integrated graphics) rebuilds the
// tracker on the CPU; a camera that stops delivering frames (unplugged,
// taken by another app, Windows power-saving) is reopened automatically.
let trackerErrors = 0;
let trackerFixing = false;
const forceCPU = { hand: false, face: false };
let lastFrameAt = 0;            // performance.now() of the last new video frame
let lastDetectTs = 0;           // MediaPipe needs strictly increasing timestamps
let reconnecting = false;
let lastTick = 0;               // previous camFrame time (rAF pauses in hidden tabs)
let lastFrameVideoTime = -1;    // video.currentTime when lastFrameAt was set
let stallKicked = false;        // already tried video.play() for this stall
const CAM_STALL_MS = 3000;      // no frames this long -> nudge the <video>
const CAM_DEAD_MS = 10000;      // still none -> reopen the camera (rarely needed)
const CAM_RECONNECT_TRIES = 3;

function camFrame(now) {
  if (!camOn) { camLoop = false; return; }
  requestAnimationFrame(camFrame);           // schedule first: an error can't stop it
  // a long gap means the tab was hidden / window minimized, not that the
  // camera died — restart the stall clock instead of reopening the camera
  if (now - lastTick > 500) lastFrameAt = now;
  lastTick = now;
  if (reconnecting) return;
  if (video.currentTime !== lastFrameVideoTime) {
    lastFrameVideoTime = video.currentTime;
    lastFrameAt = now;
    stallKicked = false;
  }
  const track = camStream && camStream.getVideoTracks()[0];
  if (!track || track.readyState === "ended" ||
      (lastFrameAt && now - lastFrameAt > CAM_DEAD_MS)) {
    reconnectCamera();
    return;
  }
  // a live track that just stopped advancing is usually a paused <video>:
  // resume it rather than closing/reopening the device (which can wedge
  // Windows camera drivers if done over and over)
  if (now - lastFrameAt > CAM_STALL_MS && !stallKicked) {
    stallKicked = true;
    video.play().catch(() => {});
  }
  const lm = camKind === "face" ? faceLandmarker : handLandmarker;
  if (!lm || trackerFixing) return;
  if (now - lastInferenceRun >= camInferenceInterval && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    lastInferenceRun = now;
    try {
      const started = performance.now();
      lastDetectTs = Math.max(now, lastDetectTs + 1);
      const res = lm.detectForVideo(video, lastDetectTs);
      const elapsed = performance.now() - started;
      camInferenceInterval = Math.max(CAM_MIN_INTERVAL_MS, elapsed * CAM_ADAPTIVE_HEADROOM);
      if (camKind === "face") faceStep(res, now / 1000);
      else handStep(res, now / 1000);
      trackerErrors = 0;
    } catch (err) {
      console.error("tracker:", err);
      if (++trackerErrors >= 3) rebuildTracker(camKind);
    }
  }
}

async function rebuildTracker(kind) {
  if (trackerFixing) return;
  trackerFixing = true;
  setStatus(`${kind} tracker stopped — restarting it…`);
  if (mode !== "idle" && camDriving) endTake();
  try { (kind === "face" ? faceLandmarker : handLandmarker).close(); } catch (e) {}
  if (kind === "face") faceLandmarker = null; else handLandmarker = null;
  forceCPU[kind] = true;                     // the GPU path is what failed
  try {
    await loadTracker(kind);
    trackerErrors = 0;
    setStatus(`${kind} tracking back on`);
  } catch (err) {
    console.error("tracker rebuild:", err);
    setStatus(`couldn't restart the ${kind} tracker — reload the page`);
  }
  trackerFixing = false;
}

async function reconnectCamera() {
  if (reconnecting) return;
  reconnecting = true;
  const kind = camKind;
  if (mode !== "idle" && camDriving) endTake();
  setStatus("camera stopped — reconnecting…");
  stopStream();
  await new Promise((r) => setTimeout(r, 1000));   // let the driver release it
  let ok = false;
  for (let i = 0; i < CAM_RECONNECT_TRIES && camOn && camKind === kind; i++) {
    try {
      const stream = await openCamera();
      if (!camOn || camKind !== kind) {       // switched off/over meanwhile
        stream.getTracks().forEach((t) => t.stop());
        break;
      }
      camStream = stream;
      video.srcObject = camStream;
      await video.play();
      lastVideoTime = -1;
      lastFrameAt = performance.now();
      setStatus(`camera back — ${kind} tracking on`);
      ok = true;
      break;
    } catch (err) {
      stopStream();
      console.error("camera reconnect:", err);
      setStatus(`camera lost, retrying… (${err.message})`);
      await new Promise((r) => setTimeout(r, 2000 * (i + 1)));
    }
  }
  if (!ok && camKind === kind) {               // give up instead of looping forever
    camOn = false;
    camKind = "off";
    camSel.value = "off";
    legendEl.innerHTML = LEGEND_POINTER;
    stopStream();
    setStatus("camera lost — pick the camera again from the menu to retry");
  }
  reconnecting = false;
}

async function loadTracker(kind) {
  if (!mp) {
    mp = await import(`${MP_CDN}/vision_bundle.mjs`);
    visionFiles = await mp.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
  }
  // Prefer the GPU delegate (runs off the main thread, frees it for
  // drawing/network work); fall back to CPU when the device/driver can't
  // provide it, so tracking still works rather than failing outright.
  const Task_CPU = (Task, opts) => Task.createFromOptions(visionFiles, opts("CPU"));
  const create = async (Task, opts) => {
    if (forceCPU[kind]) return await Task_CPU(Task, opts);
    try {
      return await Task.createFromOptions(visionFiles, opts("GPU"));
    } catch (gpuErr) {
      return await Task.createFromOptions(visionFiles, opts("CPU"));
    }
  };
  if (kind === "hand" && !handLandmarker) {
    handLandmarker = await create(mp.HandLandmarker, (delegate) => ({
      baseOptions: { modelAssetPath: "/api/hand_model", delegate },
      numHands: 1, runningMode: "VIDEO",
      minHandDetectionConfidence: 0.6, minTrackingConfidence: 0.5,
    }));
  }
  if (kind === "face" && !faceLandmarker) {
    // The face model (478 landmarks + blendshapes) runs on the CPU: on
    // integrated graphics its GPU path loses the WebGL context, which stalls
    // Chrome's GPU process and with it the camera preview.
    faceLandmarker = await Task_CPU(mp.FaceLandmarker, (delegate) => ({
      baseOptions: { modelAssetPath: "/api/face_model", delegate },
      numFaces: 1, runningMode: "VIDEO", outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    }));
  }
}

function stopStream() {
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  camStream = null;
  video.srcObject = null;
}

// Open the webcam, explaining the usual failures instead of a bare error.
// The preferred 640x480 front camera is only a hint: if the camera can't do
// it, fall back to whatever it offers.
async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("the browser blocks cameras on this address — open " +
                    "http://localhost:8321 or http://127.0.0.1:8321 instead");
  }
  const tries = [
    { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    true,
  ];
  let last = null;
  for (const video of tries) {
    try {
      return await navigator.mediaDevices.getUserMedia({ video });
    } catch (err) {
      last = err;
      if (err.name === "NotAllowedError" || err.name === "SecurityError") break;
    }
  }
  const why = {
    NotAllowedError: "permission denied — click the camera icon in the address " +
                     "bar, allow it, and reload (also check Windows Settings › " +
                     "Privacy & security › Camera)",
    SecurityError: "permission denied by the browser",
    NotFoundError: "no camera found — is it plugged in / enabled?",
    NotReadableError: "the camera is busy — close Teams, Webex, Zoom or any " +
                      "other app or tab using it, then try again",
    AbortError: "the camera is busy — close other apps using it, then try again",
    OverconstrainedError: "the camera doesn't support the requested settings",
  }[last && last.name];
  throw new Error(why || (last && (last.message || last.name)) || "unknown error");
}

async function setCamera(kind) {
  if (mode !== "idle" && camDriving) endTake();
  camOn = false;
  camKind = "off";
  camGesture = null; face = null; faceVec = null; calib = null;
  legendEl.innerHTML = LEGEND_POINTER;
  if (node) node.port.postMessage({ type: "face", on: false });
  if (kind === "off") { stopStream(); return; }
  camSel.disabled = true;
  try {
    // open the camera first, so a camera problem isn't hidden behind (or
    // delayed by) the tracker download, and report the two separately
    if (!camStream) {
      setStatus("opening camera…");
      camStream = await openCamera();
      video.srcObject = camStream;
      await video.play();
    }
    setStatus(`loading ${kind} tracker…`);
    try {
      await loadTracker(kind);
    } catch (err) {
      throw new Error(`couldn't load the ${kind} tracker (${err.message || err}) — ` +
                      "check your internet connection (it loads from a CDN)");
    }
    camKind = kind;
    camOn = true;
    lastVideoTime = -1;
    lastSeen = 0;
    lastFrameAt = performance.now();
    trackerErrors = 0;
    camInferenceInterval = CAM_MIN_INTERVAL_MS;
    if (kind === "face") {
      legendEl.innerHTML = LEGEND_FACE;
      setStatus("face tracking on — hold SPACE and make a sound to teach it");
    } else {
      setStatus("hand tracking on — pinch + sing to teach, two fingers to play");
    }
    if (!camLoop) { camLoop = true; requestAnimationFrame(camFrame); }
  } catch (err) {
    stopStream();
    camSel.value = "off";
    console.error("camera:", err);
    setStatus("camera failed: " + err.message);
  }
  camSel.disabled = false;
  camSel.blur();                             // keep SPACE/C for the instrument
}

camSel.addEventListener("change", () => setCamera(camSel.value));
// always hand the webcam back to Windows when the page goes away
window.addEventListener("pagehide", () => { camOn = false; stopStream(); });

// SPACE (hold) = sing to teach at the current position — the face's way to
// record, since the face itself is busy steering; works with mouse/hand too.
// C = redo the face calibration.
let spaceTake = false;
window.addEventListener("keydown", (ev) => {
  if (ev.code === "KeyC" && camKind === "face" && !ev.repeat &&
      !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
    startCalibration();
    return;
  }
  if (ev.code !== "Space") return;
  ev.preventDefault();                       // don't press focused buttons/selects
  if (ev.repeat || spaceTake || !node || calib) return;
  if (mode === "trace" && camDriving) endTake();   // face play yields; mouse play wins
  if (mode !== "idle") return;
  spaceTake = true;
  startTake("demo", camKind === "face");
}, true);
window.addEventListener("keyup", (ev) => {
  if (ev.code !== "Space") return;
  ev.preventDefault();
  if (!spaceTake) return;
  spaceTake = false;
  if (mode === "demo") endTake();
}, true);

// UP = sharper (smaller across-width), DOWN = smoother; the sliders do the rest
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") return;
  ev.preventDefault();
  if (!node) return;
  const factor = ev.key === "ArrowUp" ? 0.85 : 1 / 0.85;
  postConfig({ sigma: state.sigma * factor });
});

document.getElementById("clear").addEventListener("click", async () => {
  state = await fetch("/api/clear", { method: "POST" }).then(r => r.json());
  demoPaths = [];
  sendMap();
  setStatus("map cleared");
  hud();
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

// face mode: the camera image plus clean outlines of the parts that shape
// sound — lips, jaw, cheeks — each brightening as that part moves, with a
// small live label. A red frame while recording; a hint when the face is lost.
const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    409, 270, 269, 267, 0, 37, 39, 40, 185];
const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                    415, 310, 311, 312, 13, 82, 81, 80, 191];
const JAW_LINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                  377, 400, 378, 379, 365, 397, 288, 361, 323, 454];
const CHEEKS = [50, 280];

function drawFace() {
  const w = canvas.width, h = canvas.height;
  ctx2d.save();
  if (face && video.readyState >= 2) {
    const lm = face.lm, bs = face.bs;
    const px = (i) => { const u = camToUnit(lm[i]); return [u.x * w, u.y * h]; };
    const line = (idx, close) => {
      ctx2d.beginPath();
      idx.forEach((i, k) => { const [x, y] = px(i); k ? ctx2d.lineTo(x, y) : ctx2d.moveTo(x, y); });
      if (close) ctx2d.closePath();
      ctx2d.stroke();
    };
    const label = (text, x, y, color, alignRight = false) => {
      ctx2d.font = "12px ui-monospace, Consolas, monospace";
      ctx2d.textBaseline = "middle";
      const tw = ctx2d.measureText(text).width;
      if (alignRight) x -= tw;              // x is the right edge
      x = Math.min(Math.max(x, 8), w - tw - 8);   // keep on the pad
      ctx2d.fillStyle = "rgba(7,7,13,0.65)";
      ctx2d.fillRect(x - 4, y - 9, tw + 8, 18);
      ctx2d.fillStyle = color;
      ctx2d.fillText(text, x, y);
    };
    const pct = (v) => `${Math.round(100 * clamp01(v))}%`;
    ctx2d.lineJoin = ctx2d.lineCap = "round";
    ctx2d.textAlign = "left";

    // jaw: brightens as it opens
    const jaw = clamp01(face.jaw / 0.6);
    ctx2d.strokeStyle = `rgba(90,200,130,${0.3 + 0.7 * jaw})`;
    ctx2d.lineWidth = 2 + 2 * jaw;
    line(JAW_LINE, false);
    const [chinX, chinY] = px(152);
    label(`jaw ${pct(jaw)}`, chinX + 14, chinY + 14, "#5ac882");

    // cheeks: circles grow with a puff (or a smile pushing them up)
    const puff = clamp01(Math.max(bs.cheekPuff || 0, 0.6 * face.spread));
    const [ax, ay] = px(CHEEKS[0]), [bx, by] = px(CHEEKS[1]);
    const r0 = 0.12 * Math.hypot(ax - bx, ay - by);
    ctx2d.strokeStyle = `rgba(230,163,92,${0.35 + 0.65 * puff})`;
    ctx2d.lineWidth = 2;
    for (const [x, y] of [[ax, ay], [bx, by]]) {
      ctx2d.beginPath();
      ctx2d.arc(x, y, r0 * (1 + 0.8 * puff), 0, 2 * Math.PI);
      ctx2d.stroke();
    }
    const rightmost = ax > bx ? [ax, ay] : [bx, by];
    label(`cheeks ${pct(puff)}`, rightmost[0] + r0 * 1.9, rightmost[1], "#e6a35c");

    // lips: brighter as they open, pinker as they round (pucker/funnel)
    const open = clamp01(face.open / 0.5), round = clamp01(face.round / 0.6);
    const lipColor = `rgba(${230 + 25 * round},${92 - 40 * round},${92 + 120 * round},`;
    ctx2d.strokeStyle = lipColor + `${0.55 + 0.45 * open})`;
    ctx2d.lineWidth = 2.5;
    line(LIPS_OUTER, true);
    ctx2d.strokeStyle = lipColor + `${0.35 + 0.65 * open})`;
    ctx2d.lineWidth = 1.5;
    line(LIPS_INNER, true);
    const [cx, cy] = px(LIPS_OUTER[0]), [cx2] = px(LIPS_OUTER[10]);
    const lipsX = Math.min(cx, cx2) - 4;
    label(`lips open ${pct(open)} · round ${pct(round)}`, lipsX - 8, cy, "#f08a8a", true);
  }
  if (mode === "demo") {
    ctx2d.strokeStyle = "#e65c5c";
    ctx2d.lineWidth = 6;
    ctx2d.strokeRect(3, 3, w - 6, h - 6);
  }
  if (!face) {
    ctx2d.font = "13px ui-monospace, Consolas, monospace";
    ctx2d.textAlign = "center";
    ctx2d.textBaseline = "middle";
    ctx2d.fillStyle = "#c8c8d4";
    ctx2d.fillText("looking for a face…", w / 2, h / 2);
  }
  ctx2d.restore();
}

function drawCalibration() {
  const w = canvas.width, h = canvas.height;
  const st = CAL_STEPS[calib.step];
  const el = calib.t0 === null ? 0 : performance.now() / 1000 - calib.t0;
  ctx2d.save();
  ctx2d.fillStyle = "rgba(7,7,13,0.6)";
  ctx2d.fillRect(0, h / 2 - 60, w, 120);
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "middle";
  ctx2d.fillStyle = "#5c5c72";
  ctx2d.font = "13px ui-monospace, Consolas, monospace";
  ctx2d.fillText(`calibrating ${calib.step + 1}/${CAL_STEPS.length}` +
                 (face ? "" : " — no face seen"), w / 2, h / 2 - 36);
  ctx2d.fillStyle = "#5ac882";
  ctx2d.font = "24px ui-monospace, Consolas, monospace";
  ctx2d.fillText(st.say, w / 2, h / 2);
  ctx2d.fillStyle = "#23233a";
  ctx2d.fillRect(w / 2 - 120, h / 2 + 32, 240, 6);
  ctx2d.fillStyle = "#5ac882";
  ctx2d.fillRect(w / 2 - 120, h / 2 + 32, 240 * clamp01(el / CAL_SEC), 6);
  ctx2d.restore();
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx2d.fillStyle = "#0b0b14";
  ctx2d.fillRect(0, 0, w, h);
  if (camOn && video.readyState >= 2) {        // mirrored preview (dimmed under a pointer)
    const vw = video.videoWidth, vh = video.videoHeight;
    const side = Math.min(vw, vh);             // center square crop: no stretch
    ctx2d.save();
    ctx2d.globalAlpha = camKind === "face" ? 1 : 0.25;
    // display-only lift for dull webcams; tracking reads the raw video element
    ctx2d.filter = "brightness(1.15) contrast(1.1) saturate(1.2)";
    ctx2d.scale(-1, 1);
    ctx2d.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side,
                    -w, 0, w, h);
    ctx2d.restore();
  }
  // taught paths only mean something where (x, y) is what's learned; in face
  // mode the chart is just a jittery picture of a 19-number face shape
  if (camKind !== "face") {
    for (const p of demoPaths) drawPath(decimate(p), "rgba(130,190,130,0.9)", 2);
    if (mode === "demo") drawPath(decimate(livePath), "#dc5a5a", 2);
  }
  // playing leaves no trace — just the dot
  const colors = { idle: "#9696aa", demo: "#e65c5c", trace: "#5ac882" };
  if (camKind === "face") drawFace();
  if (camKind === "hand") {                    // hand ring
    ctx2d.strokeStyle = colors[camGesture === "pinch" ? "demo"
                             : camGesture === "two" ? "trace" : "idle"];
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.arc(camPos.x * w, camPos.y * h, 12, 0, 2 * Math.PI);
    ctx2d.stroke();
  }
  if (camKind !== "face") {
    ctx2d.fillStyle = colors[mode];
    ctx2d.beginPath();
    ctx2d.arc(pos.x * w, pos.y * h, mode === "idle" ? 4 : 7, 0, 2 * Math.PI);
    ctx2d.fill();
  }
  if (calib) drawCalibration();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

document.getElementById("start").addEventListener("click", async function () {
  this.remove();
  await boot();
}, { once: true });
