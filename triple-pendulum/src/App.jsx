import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const G = 9.81;
const DT = 1 / 240;
const SUBSTEPS = 3;
const TRAIL_MAX = 260;

function solve3x3(A, b) {
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const div = m[col][col] || 1e-12;
    for (let j = col; j < 4; j++) m[col][j] /= div;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const factor = m[r][col];
      for (let j = col; j < 4; j++) m[r][j] -= factor * m[col][j];
    }
  }
  return [m[0][3], m[1][3], m[2][3]];
}

function accelerations(state, params) {
  const { m1, m2, m3, L1, L2, L3, damping } = params;
  const [t1, t2, t3, w1, w2, w3] = state;

  const d12 = t1 - t2;
  const d13 = t1 - t3;
  const d23 = t2 - t3;

  const M = [
    [(m1 + m2 + m3) * L1, (m2 + m3) * L2 * Math.cos(d12), m3 * L3 * Math.cos(d13)],
    [(m2 + m3) * L1 * Math.cos(d12), (m2 + m3) * L2, m3 * L3 * Math.cos(d23)],
    [m3 * L1 * Math.cos(d13), m3 * L2 * Math.cos(d23), m3 * L3],
  ];

  const F = [
    -(m1 + m2 + m3) * G * Math.sin(t1) - (m2 + m3) * L2 * w2 * w2 * Math.sin(d12) - m3 * L3 * w3 * w3 * Math.sin(d13) - damping * w1,
    -(m2 + m3) * G * Math.sin(t2) + (m2 + m3) * L1 * w1 * w1 * Math.sin(d12) - m3 * L3 * w3 * w3 * Math.sin(d23) - damping * w2,
    -m3 * G * Math.sin(t3) + m3 * L1 * w1 * w1 * Math.sin(d13) + m3 * L2 * w2 * w2 * Math.sin(d23) - damping * w3,
  ];

  const a = solve3x3(M, F);
  return [w1, w2, w3, a[0], a[1], a[2]];
}

function rk4(state, dt, params) {
  const k1 = accelerations(state, params);
  const s2 = state.map((v, i) => v + (dt / 2) * k1[i]);
  const k2 = accelerations(s2, params);
  const s3 = state.map((v, i) => v + (dt / 2) * k2[i]);
  const k3 = accelerations(s3, params);
  const s4 = state.map((v, i) => v + dt * k3[i]);
  const k4 = accelerations(s4, params);
  return state.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

function getPositions(state, params) {
  const { L1, L2, L3 } = params;
  const [t1, t2, t3] = state;
  const x1 = L1 * Math.sin(t1);
  const y1 = L1 * Math.cos(t1);
  const x2 = x1 + L2 * Math.sin(t2);
  const y2 = y1 + L2 * Math.cos(t2);
  const x3 = x2 + L3 * Math.sin(t3);
  const y3 = y2 + L3 * Math.cos(t3);
  return { x1, y1, x2, y2, x3, y3 };
}

function energy(state, params) {
  const { m1, m2, m3, L1, L2, L3 } = params;
  const [t1, t2, t3, w1, w2, w3] = state;

  const x1d = L1 * Math.cos(t1) * w1;
  const y1d = -L1 * Math.sin(t1) * w1;
  const x2d = x1d + L2 * Math.cos(t2) * w2;
  const y2d = y1d - L2 * Math.sin(t2) * w2;
  const x3d = x2d + L3 * Math.cos(t3) * w3;
  const y3d = y2d - L3 * Math.sin(t3) * w3;

  const T = 0.5 * m1 * (x1d * x1d + y1d * y1d) + 0.5 * m2 * (x2d * x2d + y2d * y2d) + 0.5 * m3 * (x3d * x3d + y3d * y3d);
  const { y1, y2, y3 } = getPositions(state, params);
  const U = -G * (m1 * y1 + m2 * y2 + m3 * y3);
  return T + U;
}

export default function TriplePendulumSimulation() {
  const [running, setRunning] = useState(true);
  const [theta1, setTheta1] = useState(1.55);
  const [theta2, setTheta2] = useState(1.25);
  const [theta3, setTheta3] = useState(0.95);
  const [damping, setDamping] = useState(0.02);
  const [speed, setSpeed] = useState(1.0);
  const [trailOn, setTrailOn] = useState(true);

  const params = useMemo(
    () => ({ m1: 1, m2: 1, m3: 1, L1: 1, L2: 1, L3: 1, damping }),
    [damping]
  );

  const initialState = useMemo(() => [theta1, theta2, theta3, 0, 0, 0], [theta1, theta2, theta3]);
  const stateRef = useRef(initialState);
  const trailRef = useRef([]);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const lastRef = useRef(0);
  useEffect(() => {
    stateRef.current = initialState;
    trailRef.current = [];
  }, [initialState, params]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "#08111f");
      grad.addColorStop(1, "#111827");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      const scale = Math.min(w, h) / 7.5;
      const ox = w / 2;
      const oy = h / 5;

      const { x1, y1, x2, y2, x3, y3 } = getPositions(stateRef.current, params);
      const p0 = [ox, oy];
      const p1 = [ox + x1 * scale, oy + y1 * scale];
      const p2 = [ox + x2 * scale, oy + y2 * scale];
      const p3 = [ox + x3 * scale, oy + y3 * scale];

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 10; i++) {
        const gy = oy + i * 42;
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }

      if (trailOn && trailRef.current.length > 1) {
        for (let i = 1; i < trailRef.current.length; i++) {
          const a = trailRef.current[i - 1];
          const b = trailRef.current[i];
          const alpha = i / trailRef.current.length;
          ctx.strokeStyle = `rgba(125, 211, 252, ${alpha * 0.75})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      }

      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(...p0);
      ctx.lineTo(...p1);
      ctx.lineTo(...p2);
      ctx.lineTo(...p3);
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.95)";
      [p0, p1, p2, p3].forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], i === 0 ? 6 : 10, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.font = "12px sans-serif";
      ctx.fillText(`Energy: ${energy(stateRef.current, params).toFixed(3)}`, 16, 28);
      ctx.fillText(`Damping: ${damping.toFixed(3)}`, 16, 48);
      ctx.fillText(`Speed: ${speed.toFixed(2)}x`, 16, 68);
    };

    const animate = (ts) => {
      if (!lastRef.current) lastRef.current = ts;
      const elapsed = Math.min((ts - lastRef.current) / 1000, 0.03);
      lastRef.current = ts;

      if (running) {
        let accumulator = elapsed * speed;
        while (accumulator > 0) {
          const step = Math.min(accumulator, DT * SUBSTEPS);
          const subdt = step / SUBSTEPS;
          for (let i = 0; i < SUBSTEPS; i++) {
            stateRef.current = rk4(stateRef.current, subdt, params);
          }
          accumulator -= step;
        }

        const { x3, y3 } = getPositions(stateRef.current, params);
        const rect = canvas.getBoundingClientRect();
        const scale = Math.min(rect.width, rect.height) / 7.5;
        trailRef.current.push([rect.width / 2 + x3 * scale, rect.height / 5 + y3 * scale]);
        if (trailRef.current.length > TRAIL_MAX) trailRef.current.shift();
      }

      draw();
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("resize", resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [params, running, trailOn, damping, speed]);

  const reset = () => {
    stateRef.current = [theta1, theta2, theta3, 0, 0, 0];
    trailRef.current = [];
  };

  return (
    <div className="app">
      <div className="layout">
        <section className="panel canvas-panel">
          <div className="panel-header">
            <h1>Triple Pendulum Simulation</h1>
            <p>Nonlinear RK4 model with trail rendering, damping, and live controls.</p>
          </div>
          <canvas ref={canvasRef} className="sim-canvas" />
        </section>

        <section className="panel controls-panel">
          <div className="buttons">
            <button type="button" onClick={() => setRunning((v) => !v)}>
              {running ? "Pause" : "Play"}
            </button>
            <button type="button" className="secondary" onClick={reset}>
              Reset
            </button>
          </div>

          <label className="control">
            <div className="control-row">
              <span>Top angle</span>
              <span>{theta1.toFixed(2)} rad</span>
            </div>
            <input type="range" value={theta1} min={-3.0} max={3.0} step={0.01} onChange={(e) => setTheta1(Number(e.target.value))} />
          </label>

          <label className="control">
            <div className="control-row">
              <span>Middle angle</span>
              <span>{theta2.toFixed(2)} rad</span>
            </div>
            <input type="range" value={theta2} min={-3.0} max={3.0} step={0.01} onChange={(e) => setTheta2(Number(e.target.value))} />
          </label>

          <label className="control">
            <div className="control-row">
              <span>Bottom angle</span>
              <span>{theta3.toFixed(2)} rad</span>
            </div>
            <input type="range" value={theta3} min={-3.0} max={3.0} step={0.01} onChange={(e) => setTheta3(Number(e.target.value))} />
          </label>

          <label className="control">
            <div className="control-row">
              <span>Damping</span>
              <span>{damping.toFixed(3)}</span>
            </div>
            <input type="range" value={damping} min={0} max={0.2} step={0.001} onChange={(e) => setDamping(Number(e.target.value))} />
          </label>

          <label className="control">
            <div className="control-row">
              <span>Simulation speed</span>
              <span>{speed.toFixed(2)}x</span>
            </div>
            <input type="range" value={speed} min={0.1} max={2.5} step={0.05} onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>

          <div className="control-row trail-toggle">
            <span>Trail</span>
            <button type="button" className="secondary" onClick={() => setTrailOn((v) => !v)}>
              {trailOn ? "On" : "Off"}
            </button>
          </div>

          <div className="notes">
            <p>Tiny angle changes can produce dramatically different long-term motion.</p>
            <p>Higher damping settles faster; lower damping keeps the system energetic.</p>
            <p>Adjust angles, then press reset to restart from the updated state.</p>
          </div>
        </section>
      </div>
    </div>
  );
}
