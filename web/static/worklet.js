/* Gesture instrument worklet: RBF map -> DDSP decoder -> harmonic+noise synth.
 *
 * Everything real-time happens here, at the audio rate, with no network:
 *  - RBF ridge map: (x, y) -> [log f0, loudness, z16]   (W from the server)
 *  - decoder: MLP -> GRU -> MLP -> heads (amp, 60 harmonic weights, 65 noise
 *    band gains), a faithful port of ddsp.py's StreamingSynth
 *  - synth: phase-accumulated oscillator bank + FFT-filtered noise (OLA)
 *
 * The node has one input (the mic, recorded on demand for demonstration
 * takes) and one output (the instrument).
 *
 * The model is trained at 16 kHz with HOP=256 (62.5 fps frames). The context
 * may run at another rate; frame rate is kept at 62.5 fps and the noise
 * filter's 65 bands always span 0..8 kHz (silence above).
 */

const MODEL_SR = 16000;

class Instrument extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { manifest, weights, constants } = options.processorOptions;
    this.C = constants;                      // SR, HOP, N_HARMONICS, N_NOISE, Z_DIM
    const all = new Float32Array(weights);
    this.T = {};                             // name -> {d: Float32Array, shape}
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.T[t.name] = { d: all.subarray(t.offset, t.offset + n), shape: t.shape };
    }

    this.hop = Math.round(this.C.HOP * sampleRate / MODEL_SR);
    this.noiseFFT = 1 << Math.ceil(Math.log2(2 * this.hop));   // 512 at 16 kHz
    this._initFFT(this.noiseFFT);

    // mapper (empty until the server sends one)
    this.pts = null;                         // Float32Array, rows [x, y, ux, uy, 18 targets]
    this.nPts = 0;
    this.nOut = 2 + this.C.Z_DIM;
    this.sigma = 0.15;                       // across-stroke width
    this.sigmaPar = 0.04;                    // along-stroke width (aniso mode)
    this.fade = 0.0;                         // confidence gain (silence off-data)

    // control state (written by messages, read by the render loop)
    this.x = 0.5; this.y = 0.5;
    this.gate = 0.0;
    this.gain = 0.0;

    // scratch buffers (allocated once; resetSynth only clears state)
    this.h = new Float32Array(256);          // GRU hidden
    this.noiseTail = new Float32Array(this.noiseFFT - this.hop);
    this._p = new Float32Array(this.nOut);
    this._x18 = new Float32Array(2 + this.C.Z_DIM);
    this._h1 = new Float32Array(256);
    this._h2 = new Float32Array(256);
    this._g = new Float32Array(768);         // GRU input-side pre-activations
    this._gh = new Float32Array(768);        // GRU hidden-side pre-activations
    this._cat = new Float32Array(256 + 2 + this.C.Z_DIM);
    this._h3 = new Float32Array(256);
    this._harm = new Float32Array(this.C.N_HARMONICS);
    this._noise = new Float32Array(this.C.N_NOISE);
    this.resetSynth();

    // output queue + mic recording
    this.queue = new Float32Array(0);
    this.recording = false;
    this.recChunks = [];

    this.port.onmessage = (e) => this._onMessage(e.data);
    this.port.postMessage({ type: "ready", sampleRate });
  }

  _onMessage(m) {
    if (m.type === "map") {
      this.sigma = m.sigma;
      this.sigmaPar = m.sigmaPar;
      if (m.points && m.points.length) {
        const stride = 4 + this.nOut;
        this.nPts = m.points.length;
        this.pts = new Float32Array(this.nPts * stride);
        for (let i = 0; i < this.nPts; i++)
          for (let j = 0; j < stride; j++)
            this.pts[i * stride + j] = m.points[i][j];
        if (m.mode !== "aniso")                 // round kernel: drop the tangents
          for (let i = 0; i < this.nPts; i++) {
            this.pts[i * stride + 2] = 0; this.pts[i * stride + 3] = 0;
          }
      } else {
        this.pts = null; this.nPts = 0;
      }
    } else if (m.type === "pos") {
      this.x = m.x; this.y = m.y;
    } else if (m.type === "gate") {
      this.gate = m.on ? 1.0 : 0.0;
    } else if (m.type === "record") {
      if (m.on) { this.recChunks = []; this.recording = true; }
      else if (this.recording) {
        this.recording = false;
        let n = 0;
        for (const c of this.recChunks) n += c.length;
        const buf = new Float32Array(n);
        let o = 0;
        for (const c of this.recChunks) { buf.set(c, o); o += c.length; }
        this.recChunks = [];
        this.port.postMessage({ type: "take", audio: buf.buffer }, [buf.buffer]);
      }
    }
  }

  // ── mapper ────────────────────────────────────────────────────────────────
  // Kernel smoothing: out = Gaussian-weighted average of the samples (so it
  // never leaves the demonstrated range). Each sample's kernel is narrow along
  // its stroke tangent (sigmaPar) and wide across it (sigma); zero tangents
  // (round mode, crossings, stalls) make it round. Returns the kernel value at
  // the nearest sample — a confidence in (0, 1] applied as an output gain.
  predict(out) {
    const sx2 = 2 * this.sigma * this.sigma;
    const sp2 = 2 * this.sigmaPar * this.sigmaPar;
    const stride = 4 + this.nOut, P = this.pts;
    out.fill(0);
    let sw = 0, wmax = 0;
    for (let i = 0; i < this.nPts; i++) {
      const o = i * stride;
      const dx = this.x - P[o], dy = this.y - P[o + 1];
      const along = dx * P[o + 2] + dy * P[o + 3];
      const across2 = Math.max(dx * dx + dy * dy - along * along, 0);
      const w = Math.exp(-along * along / sp2 - across2 / sx2);
      sw += w;
      if (w > wmax) wmax = w;
      for (let j = 0; j < this.nOut; j++) out[j] += w * P[o + 4 + j];
    }
    const inv = 1 / (sw + 1e-12);
    for (let j = 0; j < this.nOut; j++) out[j] *= inv;
    return wmax;
  }

  // ── decoder ───────────────────────────────────────────────────────────────
  resetSynth() {                             // fresh state on next note
    this.h.fill(0);
    this.phase = 0.0;
    this.prev = null;                        // {f0, amp, harm}
    this.noiseTail.fill(0);
  }

  _lin(name, x, out) {                       // out = W x + b, W (rows, cols)
    const W = this.T[name + ".weight"].d, b = this.T[name + ".bias"].d;
    const rows = this.T[name + ".weight"].shape[0];
    const cols = this.T[name + ".weight"].shape[1];
    for (let r = 0; r < rows; r++) {
      let s = b[r];
      const o = r * cols;
      for (let c = 0; c < cols; c++) s += W[o + c] * x[c];
      out[r] = s;
    }
  }

  _lnLeaky(name, x) {                        // in-place LayerNorm then LeakyReLU
    const g = this.T[name + ".weight"].d, b = this.T[name + ".bias"].d;
    const n = x.length;
    let mu = 0;
    for (let i = 0; i < n; i++) mu += x[i];
    mu /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mu; v += d * d; }
    const inv = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) {
      const y = (x[i] - mu) * inv * g[i] + b[i];
      x[i] = y >= 0 ? y : 0.01 * y;          // LeakyReLU(0.01)
    }
  }

  _expSig(v) { const s = 1 / (1 + Math.exp(-v)); return 2 * Math.pow(s, 2.3) + 1e-7; }

  decodeFrame(f0, loud, z) {
    const x18 = this._x18;
    x18[0] = (Math.log2(Math.max(f0, 1)) - 5.9) / 3.3;
    x18[1] = (loud + 8) / 8;
    for (let i = 0; i < this.C.Z_DIM; i++) x18[2 + i] = z[i];

    this._lin("mlp_in.0", x18, this._h1); this._lnLeaky("mlp_in.1", this._h1);
    this._lin("mlp_in.3", this._h1, this._h2); this._lnLeaky("mlp_in.4", this._h2);

    // GRU cell (PyTorch gate order r, z, n)
    const Wih = this.T["gru.weight_ih_l0"].d, Whh = this.T["gru.weight_hh_l0"].d;
    const bih = this.T["gru.bias_ih_l0"].d, bhh = this.T["gru.bias_hh_l0"].d;
    const gi = this._g, h = this.h, xin = this._h2;
    for (let r = 0; r < 768; r++) {
      let si = bih[r], sh = bhh[r];
      const oi = r * 256;
      for (let c = 0; c < 256; c++) { si += Wih[oi + c] * xin[c]; sh += Whh[oi + c] * h[c]; }
      // pack: gi holds i-part, temporarily stash h-part in the upper usage below
      gi[r] = si; this._gh[r] = sh;
    }
    const gh = this._gh;
    for (let k = 0; k < 256; k++) {
      const r = 1 / (1 + Math.exp(-(gi[k] + gh[k])));
      const zg = 1 / (1 + Math.exp(-(gi[256 + k] + gh[256 + k])));
      const n = Math.tanh(gi[512 + k] + r * gh[512 + k]);
      h[k] = (1 - zg) * n + zg * h[k];
    }

    const cat = this._cat;
    cat.set(h, 0); cat.set(x18, 256);
    this._lin("mlp_out.0", cat, this._h3); this._lnLeaky("mlp_out.1", this._h3);

    const Wa = this.T["head_amp.weight"].d, ba = this.T["head_amp.bias"].d;
    let a = ba[0];
    for (let c = 0; c < 256; c++) a += Wa[c] * this._h3[c];
    const amp = this._expSig(a);

    const Whm = this.T["head_harm.weight"].d, bhm = this.T["head_harm.bias"].d;
    let hs = 0;
    for (let r = 0; r < this.C.N_HARMONICS; r++) {
      let s = bhm[r];
      const o = r * 256;
      for (let c = 0; c < 256; c++) s += Whm[o + c] * this._h3[c];
      this._harm[r] = this._expSig(s); hs += this._harm[r];
    }
    for (let r = 0; r < this.C.N_HARMONICS; r++) this._harm[r] /= (hs + 1e-7);

    const Wn = this.T["head_noise.weight"].d, bn = this.T["head_noise.bias"].d;
    for (let r = 0; r < this.C.N_NOISE; r++) {
      let s = bn[r];
      const o = r * 256;
      for (let c = 0; c < 256; c++) s += Wn[o + c] * this._h3[c];
      this._noise[r] = this._expSig(s);
    }
    return amp;
  }

  // ── synth: one HOP-sized block ────────────────────────────────────────────
  renderFrame() {
    const hop = this.hop;
    const out = new Float32Array(hop);
    let newGain = this.gain + 0.35 * (this.gate - this.gain);   // ~50 ms ramp
    if (newGain < 1e-3 && this.gate === 0) newGain = 0;

    if (newGain <= 0 || !this.pts) {
      this.gain = newGain;
      this.fade = 0;
      this.resetSynth();
      return out;
    }

    const p = this._p;
    const conf = this.predict(p);
    const f0 = Math.min(600, Math.max(60, Math.exp(p[0])));
    const amp = this.decodeFrame(f0, p[1], p.subarray(2));
    const harm = this._harm, H = this.C.N_HARMONICS;

    if (!this.prev)
      this.prev = { f0, amp, harm: Float32Array.from(harm) };
    const pv = this.prev;

    const nyq = Math.min(sampleRate / 2, MODEL_SR / 2);   // trained band only
    let phase = this.phase;
    for (let i = 0; i < hop; i++) {
      const r = (i + 1) / hop;
      const f = pv.f0 + (f0 - pv.f0) * r;
      const a = pv.amp + (amp - pv.amp) * r;
      phase += 2 * Math.PI * f / sampleRate;
      const kMax = Math.min(H, Math.floor(nyq / f));
      let s = 0;
      for (let k = 1; k <= kMax; k++) {
        const w = pv.harm[k - 1] + (harm[k - 1] - pv.harm[k - 1]) * r;
        s += Math.sin(phase * k) * w;
      }
      out[i] = a * s;
    }
    this.phase = phase % (2 * Math.PI);

    this.addNoise(out);

    // gain ramp for gating (attack/release) x confidence fade (off-data silence)
    for (let i = 0; i < hop; i++) {
      const r = (i + 1) / hop;
      out[i] *= (this.gain + (newGain - this.gain) * r)
              * (this.fade + (conf - this.fade) * r);
    }
    this.gain = newGain;
    this.fade = conf;

    pv.f0 = f0; pv.amp = amp; pv.harm.set(harm);
    return out;
  }

  addNoise(out) {
    const N = this.noiseFFT, hop = this.hop, bins = N / 2;
    const re = this._fre || (this._fre = new Float32Array(N));
    const im = this._fim || (this._fim = new Float32Array(N));
    re.fill(0); im.fill(0);
    for (let i = 0; i < hop; i++) re[i] = Math.random() * 2 - 1;
    this._fft(re, im, false);
    // real filter gain per bin: 65 bands span 0..MODEL_SR/2, zero above
    const nb = this.C.N_NOISE, noise = this._noise;
    for (let k = 0; k <= bins; k++) {
      const frac = (k * sampleRate / N) / (MODEL_SR / 2);
      let g = 0;
      if (frac <= 1) {
        const pos = frac * (nb - 1), i0 = Math.floor(pos), t = pos - i0;
        g = i0 >= nb - 1 ? noise[nb - 1] : noise[i0] * (1 - t) + noise[i0 + 1] * t;
      }
      re[k] *= g; im[k] *= g;
      if (k > 0 && k < bins) { re[N - k] *= g; im[N - k] *= g; }
    }
    this._fft(re, im, true);
    for (let i = 0; i < hop; i++)
      out[i] += re[i] + (i < this.noiseTail.length ? this.noiseTail[i] : 0);
    for (let i = 0; i < N - hop; i++) this.noiseTail[i] = re[hop + i];
  }

  // iterative radix-2 complex FFT (in-place); inverse includes 1/N
  _initFFT(N) {
    this._rev = new Uint32Array(N);
    const bits = Math.log2(N);
    for (let i = 0; i < N; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this._rev[i] = r;
    }
    this._cos = new Float32Array(N / 2);
    this._sin = new Float32Array(N / 2);
    for (let i = 0; i < N / 2; i++) {
      this._cos[i] = Math.cos(-2 * Math.PI * i / N);
      this._sin[i] = Math.sin(-2 * Math.PI * i / N);
    }
  }

  _fft(re, im, inverse) {
    const N = re.length, rev = this._rev;
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? -1 : 1;
    for (let len = 2; len <= N; len <<= 1) {
      const half = len >> 1, step = N / len;
      for (let i = 0; i < N; i += len)
        for (let j = 0; j < half; j++) {
          const wr = this._cos[j * step], wi = sgn * this._sin[j * step];
          const a = i + j, b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
    }
    if (inverse) for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
  }

  // ── audio callback ────────────────────────────────────────────────────────
  process(inputs, outputs) {
    if (this.recording && inputs[0] && inputs[0][0])
      this.recChunks.push(Float32Array.from(inputs[0][0]));

    const out = outputs[0][0];
    let q = this.queue;
    while (q.length < out.length) {
      const f = this.renderFrame();
      const nq = new Float32Array(q.length + f.length);
      nq.set(q, 0); nq.set(f, q.length);
      q = nq;
    }
    out.set(q.subarray(0, out.length));
    this.queue = q.subarray(out.length);
    return true;
  }
}

registerProcessor("instrument", Instrument);
