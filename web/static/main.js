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
let state = { seconds: 0, model: "…", mode: "aniso", sigma: 0.15, sigma_par: 0.04, points: null };
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

  audioCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
  // cache-bust: addModule ignores hard-refresh, so pin the worklet version
  await audioCtx.audioWorklet.addModule("/static/worklet.js?v=3");
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
  cursor = [[0, pos.x, pos.y]];
  mode = kind;
  if (kind === "demo") {
    livePath = [[pos.x, pos.y]];
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
  node.port.postMessage({ type: "pos", ...pos });
  if (mode !== "idle") {
    cursor.push([(performance.now() - takeT0) / 1000, pos.x, pos.y]);
    if (mode === "demo") livePath.push([pos.x, pos.y]);
  }
}

canvas.addEventListener("pointerdown", (ev) => {
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

// ── camera hand tracking (MediaPipe Tasks, fully in-browser) ────────────────
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const video = document.getElementById("cam");
const camBtn = document.getElementById("camera");
let landmarker = null;
let camOn = false;
let camStream = null;
let camPos = { x: 0.5, y: 0.5 };
let camGesture = null;          // raw (with hysteresis) gesture
let camGestureT = 0;            // when it last changed
let lastSeen = 0;               // last time a hand was detected
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

function camFrame(now) {
  if (!camOn) return;
  if (now - lastInferenceRun >= camInferenceInterval && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    lastInferenceRun = now;
    const started = performance.now();
    const res = landmarker.detectForVideo(video, now);
    const elapsed = performance.now() - started;
    camInferenceInterval = Math.max(CAM_MIN_INTERVAL_MS, elapsed * CAM_ADAPTIVE_HEADROOM);
    const t = now / 1000;
    if (res.landmarks && res.landmarks.length) {
      const c = classify(res.landmarks[0], camGesture);
      // map through the same center square crop the preview is drawn with
      const vw = video.videoWidth, vh = video.videoHeight;
      const side = Math.min(vw, vh);
      const cx = (c.x * vw - (vw - side) / 2) / side;
      const cy = (c.y * vh - (vh - side) / 2) / side;
      camPos = { x: Math.min(1, Math.max(0, 1 - cx)),    // mirror
                 y: Math.min(1, Math.max(0, cy)) };
      if (c.g !== camGesture) { camGesture = c.g; camGestureT = t; }
      lastSeen = t;
    } else if (t - lastSeen > 0.3 && camGesture !== null) {
      camGesture = null; camGestureT = t;                // ride through dropouts
    }

    // camera may start takes when idle, but only ends takes it started
    const want = camGesture === "pinch" ? "demo"
               : camGesture === "two" ? "trace" : "idle";
    const hold = want !== "idle" ? 0.12 : 0.35;          // quick engage, slow release
    if (want !== mode && t - camGestureT >= hold && (mode === "idle" || camDriving)) {
      if (mode !== "idle") endTake();
      if (want !== "idle") { pos = camPos; startTake(want, true); }
    }
    if (camGesture !== null && (mode === "idle" || camDriving)) moveTo(camPos);
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
        numHands: 1, runningMode: "VIDEO",
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
  for (const p of demoPaths) drawPath(decimate(p), "rgba(130,190,130,0.9)", 2);
  if (mode === "demo") drawPath(decimate(livePath), "#dc5a5a", 2);
  // playing leaves no trace — just the dot
  const colors = { idle: "#9696aa", demo: "#e65c5c", trace: "#5ac882" };
  if (camOn) {                                 // hand ring
    ctx2d.strokeStyle = colors[camGesture === "pinch" ? "demo"
                             : camGesture === "two" ? "trace" : "idle"];
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.arc(camPos.x * w, camPos.y * h, 12, 0, 2 * Math.PI);
    ctx2d.stroke();
  }
  ctx2d.fillStyle = colors[mode];
  ctx2d.beginPath();
  ctx2d.arc(pos.x * w, pos.y * h, mode === "idle" ? 4 : 7, 0, 2 * Math.PI);
  ctx2d.fill();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

document.getElementById("start").addEventListener("click", async function () {
  this.remove();
  await boot();
}, { once: true });
