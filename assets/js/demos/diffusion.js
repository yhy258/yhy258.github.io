// A diffusion model on 2D points, run in the page. The data is a mixture of Gaussians, so the score of every
// noised distribution p_t is known in closed form and no network is needed: the reverse process below uses it.
//
// In a post:  <figure class="widget" data-demo="diffusion"><figcaption>...</figcaption></figure>
// and in the post's front matter:  scripts: [/assets/js/demos/diffusion.js]
(function () {
  "use strict";

  const N = 700;            // particles
  const STEPS = 240;        // reverse steps from t = 1 to t = 0
  const VIEW_X = 3.4;       // the canvas shows at least [-VIEW_X, VIEW_X] across
  const VIEW_Y = 2.7;       // and [-VIEW_Y, VIEW_Y] down
  const BETA_MIN = 0.1;
  const BETA_MAX = 20;

  // Variance-preserving schedule: beta(t) is linear, alphaBar(t) = exp(-integral of beta)
  const beta = (t) => BETA_MIN + t * (BETA_MAX - BETA_MIN);
  const alphaBar = (t) => Math.exp(-(BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t * t));

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  // Each dataset: the means of equally weighted Gaussians with standard deviation s
  const DATASETS = {
    "Ring of Gaussians": () => {
      const mu = [];
      for (let k = 0; k < 8; k++) mu.push([2 * Math.cos((k * Math.PI) / 4), 2 * Math.sin((k * Math.PI) / 4)]);
      return { mu, s: 0.14 };
    },
    "Two moons": () => {
      const mu = [];
      for (let k = 0; k < 24; k++) {
        const a = (k / 23) * Math.PI;
        mu.push([1.5 * Math.cos(a) - 0.75, 1.5 * Math.sin(a) - 0.4]);
        mu.push([0.75 - 1.5 * Math.cos(a), 0.4 - 1.5 * Math.sin(a)]);
      }
      return { mu, s: 0.09 };
    },
    "Spiral": () => {
      const mu = [];
      for (let k = 0; k < 70; k++) {
        const a = 0.6 + (k / 69) * 3.4 * Math.PI;
        const r = 0.2 * a;
        mu.push([r * Math.cos(a), r * Math.sin(a)]);
      }
      return { mu, s: 0.07 };
    },
  };

  function el(tag, props, children) {
    const node = Object.assign(document.createElement(tag), props || {});
    (children || []).forEach((c) => node.append(c));
    return node;
  }

  function mount(root) {
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#0e7c86";

    const canvas = el("canvas", { role: "img" });
    canvas.setAttribute("aria-label", "Points being noised and denoised by a diffusion process");
    const slider = el("input", { type: "range", min: 0, max: 1, step: 0.001, value: 0 });
    slider.setAttribute("aria-label", "Diffusion time t");
    const readout = el("span", { className: "widget__readout" });
    const play = el("button", { type: "button", className: "primary", textContent: "Generate from noise" });
    const resample = el("button", { type: "button", textContent: "New data" });
    const dataset = el("select", {}, Object.keys(DATASETS).map((name) => el("option", { textContent: name })));
    const sampler = el("select", {}, [
      el("option", { value: "sde", textContent: "Reverse SDE" }),
      el("option", { value: "ode", textContent: "Probability-flow ODE" }),
    ]);
    const field = el("input", { type: "checkbox" });

    const controls = [
      el("div", { className: "widget__controls" }, [
        el("label", { className: "widget__slider" }, [slider, readout]),
      ]),
      el("div", { className: "widget__controls" }, [
        play,
        resample,
        el("label", {}, ["Data", dataset]),
        el("label", {}, ["Sampler", sampler]),
        el("label", {}, [field, "Show score field"]),
      ]),
    ];
    root.prepend(canvas, ...controls);

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, scale = 1;
    let data = DATASETS[dataset.value]();
    const x0 = new Float32Array(2 * N);    // clean samples
    const eps = new Float32Array(2 * N);   // the noise each sample gets in the forward process
    const pos = new Float32Array(2 * N);   // what is drawn
    let t = 0;
    let running = null;                    // the animation frame of a reverse run

    function sampleData() {
      data = DATASETS[dataset.value]();
      for (let i = 0; i < N; i++) {
        const m = data.mu[Math.floor(Math.random() * data.mu.length)];
        x0[2 * i] = m[0] + data.s * randn();
        x0[2 * i + 1] = m[1] + data.s * randn();
        eps[2 * i] = randn();
        eps[2 * i + 1] = randn();
      }
    }

    // Forward process in closed form: x_t = sqrt(alphaBar) x_0 + sqrt(1 - alphaBar) eps
    function forward() {
      const ab = alphaBar(t);
      const a = Math.sqrt(ab), sig = Math.sqrt(1 - ab);
      for (let i = 0; i < 2 * N; i++) pos[i] = a * x0[i] + sig * eps[i];
    }

    // Score of p_t at (x, y). p_t is a mixture of N(a mu_k, v I) with v = a^2 s^2 + (1 - alphaBar).
    const out = [0, 0];
    function score(x, y, time) {
      const ab = alphaBar(time);
      const a = Math.sqrt(ab);
      const v = ab * data.s * data.s + (1 - ab);
      const mu = data.mu;
      let best = -Infinity;
      const logw = score.buf.length >= mu.length ? score.buf : (score.buf = new Float64Array(mu.length));
      for (let k = 0; k < mu.length; k++) {
        const dx = x - a * mu[k][0], dy = y - a * mu[k][1];
        logw[k] = -(dx * dx + dy * dy) / (2 * v);
        if (logw[k] > best) best = logw[k];
      }
      let sum = 0, sx = 0, sy = 0;
      for (let k = 0; k < mu.length; k++) {
        const w = Math.exp(logw[k] - best);
        sum += w;
        sx += w * (a * mu[k][0] - x);
        sy += w * (a * mu[k][1] - y);
      }
      out[0] = sx / (sum * v);
      out[1] = sy / (sum * v);
      return out;
    }
    score.buf = new Float64Array(0);

    // One Euler step of the reverse process, from t to t - dt
    function reverseStep(dt) {
      const b = beta(t);
      const sde = sampler.value === "sde";
      const noise = Math.sqrt(b * dt);
      for (let i = 0; i < N; i++) {
        const x = pos[2 * i], y = pos[2 * i + 1];
        const s = score(x, y, t);
        if (sde) {
          pos[2 * i] = x + (0.5 * b * x + b * s[0]) * dt + noise * randn();
          pos[2 * i + 1] = y + (0.5 * b * y + b * s[1]) * dt + noise * randn();
        } else {
          pos[2 * i] = x + 0.5 * b * (x + s[0]) * dt;
          pos[2 * i + 1] = y + 0.5 * b * (y + s[1]) * dt;
        }
      }
      t = Math.max(0, t - dt);
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      W = canvas.clientWidth;
      H = Math.round(Math.min(W * 0.8, 460));
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      scale = Math.min(W / (2 * VIEW_X), H / (2 * VIEW_Y));
      draw();
    }

    const px = (x) => W / 2 + x * scale;
    const py = (y) => H / 2 - y * scale;

    function draw() {
      if (!W) return;
      ctx.clearRect(0, 0, W, H);

      if (field.checked) {
        // Arrows point along the score; their length saturates so that the field stays readable near t = 0
        const ab = alphaBar(t);
        const v = ab * data.s * data.s + (1 - ab);
        const cell = 34;
        ctx.strokeStyle = "rgba(60, 70, 80, 0.38)";
        ctx.fillStyle = "rgba(60, 70, 80, 0.38)";
        ctx.lineWidth = 1;
        for (let cx = cell / 2 + ((W % cell) / 2); cx < W; cx += cell) {
          for (let cy = cell / 2 + ((H % cell) / 2); cy < H; cy += cell) {
            const s = score((cx - W / 2) / scale, (H / 2 - cy) / scale, t);
            const mag = Math.hypot(s[0], s[1]);
            if (mag < 1e-9) continue;
            const len = cell * 0.42 * Math.tanh(mag * Math.sqrt(v) * 0.6);
            const ux = s[0] / mag, uy = -s[1] / mag;
            const ex = cx + ux * len, ey = cy + uy * len;
            ctx.beginPath();
            ctx.moveTo(cx - ux * len * 0.4, cy - uy * len * 0.4);
            ctx.lineTo(ex, ey);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(ex + ux * 3, ey + uy * 3);
            ctx.lineTo(ex - uy * 2.4, ey + ux * 2.4);
            ctx.lineTo(ex + uy * 2.4, ey - ux * 2.4);
            ctx.fill();
          }
        }
      }

      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.55;
      for (let i = 0; i < N; i++) {
        ctx.beginPath();
        ctx.arc(px(pos[2 * i]), py(pos[2 * i + 1]), 2.1, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      const ab = alphaBar(t);
      slider.value = t;
      readout.textContent =
        "t = " + t.toFixed(2) + " · signal " + Math.sqrt(ab).toFixed(2) + " · noise " + Math.sqrt(1 - ab).toFixed(2);
    }

    function stop() {
      if (running !== null) cancelAnimationFrame(running);
      running = null;
      play.textContent = "Generate from noise";
    }

    function generate() {
      if (running !== null) {
        stop();
        return;
      }
      for (let i = 0; i < 2 * N; i++) pos[i] = randn();   // p_1 is (almost exactly) a standard normal
      t = 1;
      play.textContent = "Stop";
      const frame = () => {
        for (let k = 0; k < 2 && t > 0; k++) reverseStep(1 / STEPS);
        draw();
        if (t > 0) {
          running = requestAnimationFrame(frame);
        } else {
          // What was generated becomes the data, so the slider now noises the model's own samples
          x0.set(pos);
          stop();
        }
      };
      running = requestAnimationFrame(frame);
    }

    slider.addEventListener("input", () => {
      stop();
      t = Number(slider.value);
      forward();
      draw();
    });
    play.addEventListener("click", generate);
    resample.addEventListener("click", () => {
      stop();
      sampleData();
      forward();
      draw();
    });
    dataset.addEventListener("change", () => {
      stop();
      t = 0;
      sampleData();
      forward();
      draw();
    });
    field.addEventListener("change", draw);
    new ResizeObserver(resize).observe(canvas);

    sampleData();
    forward();
    resize();
  }

  document.querySelectorAll('[data-demo="diffusion"]').forEach(mount);
})();
