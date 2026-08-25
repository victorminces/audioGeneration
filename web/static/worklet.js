/* Gesture instrument worklet: RBF map -> DDSP decoder -> harmonic+noise synth.
 *
 * Everything real-time happens here, at the audio rate, with no network:
 *  - kernel map: gesture point in [0,1]^D -> [log f0, relative level, z16]  (samples
 *    from the server; D is 2 for the mouse pad, 4 for the full-body features)
 *  - decoder: MLP -> GRU -> MLP -> heads (60 harmonic weights, 65 noise
 *    band gains, noise fraction), a port of ddsp.py's StreamingSynth. Level
 *    is not an input: it multiplies the mixed output and nothing else
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
const MAX_DIM = 8;                 // gesture-space capacity; 2 (pad) and 4 (body) today

class Instrument extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { manifest, weights, constants } = options.processorOptions;
    this.C = constants;                      // SR, HOP, N_HARMONICS, N_NOISE, Z_DIM

    // One decoder per harmonic mode, both resident, so switching between them
    // is a pointer change rather than a reload. `weights` is {mode: ArrayBuffer}.
    //
    // How each checkpoint's weights read f0 is baked in at training, so the
    // scale is per model. It differs from how high the oscillator may actually
    // run on purpose: the decoder extrapolates past its training range
    // cleanly, so playback is limited by what the analyser can measure.
    this.modes = manifest.modes;
    this.M = {};
    for (const mode of this.modes) {
      const man = manifest.models[mode];
      const all = new Float32Array(weights[mode]);
      const T = {};
      for (const t of man.tensors) {
        const n = t.shape.reduce((a, b) => a * b, 1);
        T[t.name] = { d: all.subarray(t.offset, t.offset + n), shape: t.shape };
      }
      this.M[mode] = {
        T, harmMode: man.harm_mode, name: man.name,
        f0Lo: Math.log2(man.f0_scale_min),
        f0Span: Math.log2(man.f0_scale_max / man.f0_scale_min),
        // where this model's timbre code sits in a map row:
        // [log_f0, rel, z_mode0(16), z_mode1(16), ...]
        zOff: 2 + this.modes.indexOf(mode) * this.C.Z_DIM,
      };
    }
    this._useModel(this.modes[0]);

    this.envFMin = this.C.ENV_F_MIN || 40;
    this.envSpan = Math.log2((MODEL_SR / 2) / this.envFMin);
    this.envLo = Math.log2(this.envFMin);

    this.f0PlayMin = this.C.F0_PLAY_MIN || 60;
    this.f0PlayMax = this.C.F0_PLAY_MAX || 2000;

    // The whole of the level path. The map's column 1 is level *relative* to
    // the performed level of the take it came from, so amp = exp(rel +
    // REF_LOUD) puts it back on an absolute footing; the noise branch is
    // divided by NOISE_UNIT_RMS so both branches reach the mix at unit level.
    this.refLoud = this.C.REF_LOUD !== undefined ? this.C.REF_LOUD : -2.0;
    this.noiseUnitRms = this.C.NOISE_UNIT_RMS || 0.576;

    this.hop = Math.round(this.C.HOP * sampleRate / MODEL_SR);
    this.noiseFFT = 1 << Math.ceil(Math.log2(2 * this.hop));   // 512 at 16 kHz
    this._initFFT(this.noiseFFT);

    // mapper (empty until the server sends one)
    this.pts = null;                         // Float32Array, rows [p_1..p_D, u_1..u_D, 18 targets]
    this.nPts = 0;
    this.dim = 2;                            // D of the gesture space (2 = pad, 4 = body)
    this.nOut = 2 + this.modes.length * this.C.Z_DIM;
    this.sigma = 0.15;                       // across-stroke width
    this.sigmaPar = 0.04;                    // along-stroke width (aniso mode)
    this.reach = 1.0;                        // gain radius, in multiples of sigma
    this.breath = 1.0;                       // noise-branch trim (1 = as trained)
    this.fade = 0.0;                         // confidence gain (silence off-data)

    // control state (written by messages, read by the render loop). Fixed
    // capacity so nothing allocates on the audio thread when D changes.
    this.pos = new Float32Array(MAX_DIM).fill(0.5);
    this.gate = 0.0;
    this.gain = 0.0;

    // scratch buffers (allocated once; resetSynth only clears state)
    this.h = new Float32Array(256);          // GRU hidden
    this.noiseTail = new Float32Array(this.noiseFFT - this.hop);
    this._p = new Float32Array(this.nOut);
    // decoder input is [f0, z...]: level is not part of it, by design
    this._x = new Float32Array(1 + this.C.Z_DIM);
    this._h1 = new Float32Array(256);
    this._h2 = new Float32Array(256);
    this._g = new Float32Array(768);         // GRU input-side pre-activations
    this._gh = new Float32Array(768);        // GRU hidden-side pre-activations
    this._cat = new Float32Array(256 + 1 + this.C.Z_DIM);
    this._h3 = new Float32Array(256);
    this._harm = new Float32Array(this.C.N_HARMONICS);
    this._noise = new Float32Array(this.C.N_NOISE);
    this._env = new Float32Array(this.C.N_ENV || 65);       // filter-mode envelope
    this._nbuf = new Float32Array(this.hop);   // unit-RMS noise branch
    this._gbuf = new Float32Array(this.noiseFFT / 2 + 1);   // noise bin gains
    this.resetSynth();

    // output queue + mic recording
    this.queue = new Float32Array(0);
    this.recording = false;
    this.recChunks = [];
    this._lvlSum = 0; this._lvlN = 0; this._lvlBlocks = 0;   // mic level meter

    this.port.onmessage = (e) => this._onMessage(e.data);
    this.port.postMessage({ type: "ready", sampleRate });
  }

  _onMessage(m) {
    if (m.type === "breath") {
      this.breath = m.value;
    } else if (m.type === "model") {
      if (this.M[m.mode] && m.mode !== this.mode) {
        this._useModel(m.mode);
        // the GRU hidden state belongs to the decoder that produced it, so it
        // is meaningless to the other one — start the new model clean
        this.resetSynth();
        this.port.postMessage({ type: "model", mode: m.mode,
                                name: this.M[m.mode].name });
      }
    } else if (m.type === "map") {
      this.sigma = m.sigma;
      this.sigmaPar = m.sigmaPar;
      if (m.reach) this.reach = m.reach;
      const D = Math.min(m.dim || 2, MAX_DIM);
      this.dim = D;
      if (m.points && m.points.length) {
        const stride = 2 * D + this.nOut;
        this.nPts = m.points.length;
        this.pts = new Float32Array(this.nPts * stride);
        for (let i = 0; i < this.nPts; i++)
          for (let j = 0; j < stride; j++)
            this.pts[i * stride + j] = m.points[i][j];
        if (m.mode !== "aniso")                 // round kernel: drop the tangents
          for (let i = 0; i < this.nPts; i++)
            for (let j = 0; j < D; j++) this.pts[i * stride + D + j] = 0;
      } else {
        this.pts = null; this.nPts = 0;
      }
    } else if (m.type === "pos") {
      const p = m.p;                            // array of D coordinates
      for (let j = 0; j < this.dim && j < p.length; j++) this.pos[j] = p[j];
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
    const D = this.dim, stride = 2 * D + this.nOut, P = this.pts, q = this.pos;
    out.fill(0);
    let sw = 0, emin = Infinity;
    for (let i = 0; i < this.nPts; i++) {
      const o = i * stride;
      // squared distance and the component along this sample's tangent, both
      // accumulated over D dims (the 2D case is just D = 2 of this loop)
      let d2 = 0, along = 0;
      for (let j = 0; j < D; j++) {
        const dj = q[j] - P[o + j];
        d2 += dj * dj;
        along += dj * P[o + D + j];
      }
      const across2 = Math.max(d2 - along * along, 0);
      const e = along * along / sp2 + across2 / sx2;
      const w = Math.exp(-e);
      sw += w;
      if (e < emin) emin = e;
      const base = o + 2 * D;
      for (let j = 0; j < this.nOut; j++) out[j] += w * P[base + j];
    }
    const inv = 1 / (sw + 1e-12);
    for (let j = 0; j < this.nOut; j++) out[j] *= inv;
    // widening sigma by `reach` divides the exponent by reach^2 — averaging
    // stays as sharp as sigma says, only the fade to silence gets gentler
    return emin === Infinity ? 0 : Math.exp(-emin / (this.reach * this.reach));
  }

  // ── decoder ───────────────────────────────────────────────────────────────
  _useModel(mode) {
    const m = this.M[mode];
    this.mode = mode;
    this.T = m.T;                            // _lin/_lnLeaky read through this
    this.harmMode = m.harmMode;
    this.f0Lo = m.f0Lo;
    this.f0Span = m.f0Span;
    this.zOff = m.zOff;
  }

  // Hz -> fractional index into the log-spaced envelope bands. Mirrors
  // ddsp.env_band; the two must agree or the filter model reads its own
  // envelope at the wrong frequencies.
  _envBand(f) {
    const pos = (Math.log2(Math.max(f, this.envFMin)) - this.envLo)
              / this.envSpan * (this.C.N_ENV - 1);
    return pos < 0 ? 0 : (pos > this.C.N_ENV - 1 ? this.C.N_ENV - 1 : pos);
  }

  resetSynth() {                             // fresh state on next note
    this.h.fill(0);
    this.phase = 0.0;
    this.prev = null;                        // {f0, amp, mix, harm}
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

  // (f0, z) -> harm/noise shapes + noise fraction, all dimensionless. Level is
  // deliberately absent: it multiplies the result in renderFrame and never
  // reaches the network. Returns mix.
  decodeFrame(f0, z) {
    const x18 = this._x;
    x18[0] = (Math.log2(Math.max(f0, 1)) - this.f0Lo) / this.f0Span;
    for (let i = 0; i < this.C.Z_DIM; i++) x18[1 + i] = z[i];

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

    const H = this.C.N_HARMONICS;
    let hs = 0;
    if (this.harmMode === "amp") {
      // amplitude mode: one weight per harmonic, straight out of the head
      const Whm = this.T["head_harm.weight"].d, bhm = this.T["head_harm.bias"].d;
      for (let r = 0; r < H; r++) {
        let s = bhm[r];
        const o = r * 256;
        for (let c = 0; c < 256; c++) s += Whm[o + c] * this._h3[c];
        this._harm[r] = this._expSig(s); hs += this._harm[r];
      }
    } else {
      // filter mode: a spectral envelope on absolute frequency, read at k*f0
      const NE = this.C.N_ENV, env = this._env;
      const We = this.T["head_env.weight"].d, be = this.T["head_env.bias"].d;
      for (let r = 0; r < NE; r++) {
        let s = be[r];
        const o = r * 256;
        for (let c = 0; c < 256; c++) s += We[o + c] * this._h3[c];
        env[r] = this._expSig(s);
      }
      for (let k = 1; k <= H; k++) {
        const pos = this._envBand(f0 * k);
        let i0 = Math.floor(pos);
        if (i0 > NE - 2) i0 = NE - 2;
        const v = env[i0] + (env[i0 + 1] - env[i0]) * (pos - i0);
        this._harm[k - 1] = v; hs += v;
      }
    }
    for (let r = 0; r < H; r++) this._harm[r] /= (hs + 1e-7);

    const Wn = this.T["head_noise.weight"].d, bn = this.T["head_noise.bias"].d;
    let ns = 0;
    for (let r = 0; r < this.C.N_NOISE; r++) {
      let s = bn[r];
      const o = r * 256;
      for (let c = 0; c < 256; c++) s += Wn[o + c] * this._h3[c];
      this._noise[r] = this._expSig(s); ns += this._noise[r] * this._noise[r];
    }
    // unit RMS across bands, matching ddsp.Decoder — the mix below assumes it
    const nrms = Math.sqrt(ns / this.C.N_NOISE);
    for (let r = 0; r < this.C.N_NOISE; r++) this._noise[r] /= (nrms + 1e-7);

    const Wx = this.T["head_mix.weight"].d, bx = this.T["head_mix.bias"].d;
    let m = bx[0];
    for (let c = 0; c < 256; c++) m += Wx[c] * this._h3[c];
    return 1 / (1 + Math.exp(-m));            // noise fraction in [0, 1]
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
    const f0 = Math.min(this.f0PlayMax, Math.max(this.f0PlayMin, Math.exp(p[0])));
    // `breath` trims the noise fraction: the balance the decoder learned comes
    // from speech, which carries more breath than a sustained instrument tone
    // wants. 1 = exactly as trained.
    // each model has its own slice of the row; z from one is meaningless to
    // the other, so the offset moves with the active model
    const z = p.subarray(this.zOff, this.zOff + this.C.Z_DIM);
    const mix = Math.min(1, Math.max(0, this.decodeFrame(f0, z) * this.breath));
    const amp = Math.exp(p[1] + this.refLoud);
    const harm = this._harm, H = this.C.N_HARMONICS;

    if (!this.prev)
      this.prev = { f0, amp, mix, harm: Float32Array.from(harm) };
    const pv = this.prev;

    this.renderNoise(this._nbuf);

    const nyq = Math.min(sampleRate / 2, MODEL_SR / 2);   // trained band only
    let phase = this.phase;
    for (let i = 0; i < hop; i++) {
      const r = (i + 1) / hop;
      const f = pv.f0 + (f0 - pv.f0) * r;
      const a = pv.amp + (amp - pv.amp) * r;
      const mx = pv.mix + (mix - pv.mix) * r;
      phase += 2 * Math.PI * f / sampleRate;
      const kMax = Math.min(H, Math.floor(nyq / f));
      let s = 0, ss = 0;
      for (let k = 1; k <= kMax; k++) {
        const w = pv.harm[k - 1] + (harm[k - 1] - pv.harm[k - 1]) * r;
        s += Math.sin(phase * k) * w;
        ss += w * w;
      }
      // sum_k w_k sin(k phase) has RMS sqrt(sum w_k^2 / 2); dividing it out
      // puts both branches at unit RMS, and the quadrature weights keep the
      // sum at unit RMS for any mix — so the output RMS is `a` by construction
      const hb = s / (Math.sqrt(ss / 2) + 1e-7);
      out[i] = a * (Math.sqrt(1 - mx) * hb + Math.sqrt(mx) * this._nbuf[i]);
    }
    this.phase = phase % (2 * Math.PI);

    // gain ramp for gating (attack/release) x confidence fade (off-data silence)
    for (let i = 0; i < hop; i++) {
      const r = (i + 1) / hop;
      out[i] *= (this.gain + (newGain - this.gain) * r)
              * (this.fade + (conf - this.fade) * r);
    }
    this.gain = newGain;
    this.fade = conf;

    pv.f0 = f0; pv.amp = amp; pv.mix = mix; pv.harm.set(harm);
    return out;
  }

  // One hop of band-filtered noise at unit RMS, written into `buf`. Unit
  // because the quadrature mix above assumes both branches arrive at the same
  // level; NOISE_UNIT_RMS is what unit-RMS band gains actually produce through
  // this FFT/overlap-add path (measured, and within 1.2% of the torch path's,
  // so the two implementations agree on what a given mix sounds like).
  renderNoise(buf) {
    const N = this.noiseFFT, hop = this.hop, bins = N / 2;
    const re = this._fre || (this._fre = new Float32Array(N));
    const im = this._fim || (this._fim = new Float32Array(N));
    re.fill(0); im.fill(0);
    for (let i = 0; i < hop; i++) re[i] = Math.random() * 2 - 1;
    this._fft(re, im, false);
    // real filter gain per bin: 65 bands span 0..MODEL_SR/2, zero above.
    // Built first, then normalised over the bins themselves rather than over
    // the 65 band values — the bands cover a fixed 0..8 kHz however many bins
    // this transform has, so at a 48 kHz context two thirds of them are zero
    // and band-level normalisation would leave the noise branch far too quiet.
    // Matches ddsp.py's synthesize(); see the comment there.
    const nb = this.C.N_NOISE, noise = this._noise, gb = this._gbuf;
    let gs = 0;
    for (let k = 0; k <= bins; k++) {
      const frac = (k * sampleRate / N) / (MODEL_SR / 2);
      let g = 0;
      if (frac <= 1) {
        const pos = frac * (nb - 1), i0 = Math.floor(pos), t = pos - i0;
        g = i0 >= nb - 1 ? noise[nb - 1] : noise[i0] * (1 - t) + noise[i0 + 1] * t;
      }
      gb[k] = g; gs += g * g;
    }
    const gn = 1 / (Math.sqrt(gs / (bins + 1)) + 1e-7);
    for (let k = 0; k <= bins; k++) {
      const g = gb[k] * gn;
      re[k] *= g; im[k] *= g;
      if (k > 0 && k < bins) { re[N - k] *= g; im[N - k] *= g; }
    }
    this._fft(re, im, true);
    const inv = 1 / this.noiseUnitRms;
    for (let i = 0; i < hop; i++)
      buf[i] = (re[i] + (i < this.noiseTail.length ? this.noiseTail[i] : 0)) * inv;
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
    const mic = inputs[0] && inputs[0][0];
    // Mic level, always — not just while recording. The front end needs it
    // before a take starts, to learn the room's noise floor during the hold
    // gesture and so tell you live whether you are loud enough to be captured.
    if (mic) {
      for (let i = 0; i < mic.length; i++) this._lvlSum += mic[i] * mic[i];
      this._lvlN += mic.length;
      if (++this._lvlBlocks >= 8) {            // ~15 posts/sec at 128-frame blocks
        this.port.postMessage({ type: "level",
                                rms: Math.sqrt(this._lvlSum / this._lvlN) });
        this._lvlSum = 0; this._lvlN = 0; this._lvlBlocks = 0;
      }
    }
    if (this.recording && mic)
      this.recChunks.push(Float32Array.from(mic));

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
