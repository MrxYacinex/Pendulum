import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TriplePendulumParams:
    g: float = 9.81
    m_cart: float = 2.0
    m1: float = 1.0
    m2: float = 1.0
    m3: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    l3: float = 1.0
    damping: float = 0.08
    dt: float = 0.01
    substeps: int = 4
    max_u: float = 18.0  # interpreted as max cart force command [N]
    # Actuator realism
    actuator_tau: float = 0.06
    max_u_rate: float = 300.0
    max_u_jerk: float = 6000.0
    # Friction realism
    joint_coulomb: float = 0.05
    cart_friction: float = 0.20
    # Observation realism
    obs_delay_steps: int = 2
    obs_noise_std: float = 0.01
    # Domain randomization (used during training)
    randomize_domain: bool = False
    # Curriculum
    use_curriculum: bool = True
    # Reward shaping
    smooth_u_weight: float = 0.02
    alive_bonus: float = 0.2
    w_angle: float = 10.0
    w_ang_vel: float = 0.20
    w_x: float = 0.90
    w_xdot: float = 0.12
    w_u: float = 0.002
    w_upright_shape: float = 2.0
    upright_sigma: float = 0.12
    fail_x_penalty: float = 2000.0
    horizon_balance_bonus: float = 800.0
    # Reset distribution bounds used by curriculum interpolation.
    reset_e_min: float = 0.03
    reset_e_max: float = 0.45
    reset_x_min: float = 0.02
    reset_x_max: float = 0.30
    reset_xd_min: float = 0.02
    reset_xd_max: float = 0.26
    reset_w_min: float = 0.03
    reset_w_max: float = 0.30
    x_limit: float = 4.0
    max_steps: int = 2000


class TriplePendulumUprightEnv(gym.Env):
    """
    Nonlinear triple-pendulum with force-driven horizontal cart/base control.

    State:
      [x, th1, th2, th3, x_dot, w1, w2, w3]
    Control:
      u = cart force command (actuator filtered)
    Goal:
      Stabilize upright th_i = pi (mod 2*pi), while keeping cart near x=0.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, params: TriplePendulumParams | None = None):
        super().__init__()
        self.p = params if params is not None else TriplePendulumParams()
        self.state = np.zeros(8, dtype=np.float64)
        self.u_actual = 0.0
        self.u_rate = 0.0
        self.u_prev = 0.0
        self.steps = 0
        self.training_progress = 0.0
        self.obs_buffer = deque(maxlen=self.p.obs_delay_steps + 1)
        self.runtime = {}

        self.action_space = spaces.Box(
            low=np.array([-self.p.max_u], dtype=np.float32),
            high=np.array([self.p.max_u], dtype=np.float32),
            dtype=np.float32,
        )

        high = np.array(
            [
                2.0 * self.p.x_limit,  # x
                np.pi,  # e1
                np.pi,  # e2
                np.pi,  # e3
                20.0,  # x_dot
                30.0,  # w1
                30.0,  # w2
                30.0,  # w3
                self.p.max_u,  # applied force
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _sample_runtime_params(self):
        alpha = float(np.clip(self.training_progress, 0.0, 1.0))
        if not self.p.randomize_domain:
            self.runtime = {
                "g": self.p.g,
                "m1": self.p.m1,
                "m2": self.p.m2,
                "m3": self.p.m3,
                "l1": self.p.l1,
                "l2": self.p.l2,
                "l3": self.p.l3,
                "damping": self.p.damping,
                "joint_coulomb": self.p.joint_coulomb,
                "cart_friction": self.p.cart_friction,
            }
            return

        rng = self.np_random
        # Curriculum on randomization breadth: narrow early, broader later.
        mass_span = 0.05 + 0.10 * alpha
        len_span = 0.03 + 0.07 * alpha
        g_span = 0.005 + 0.015 * alpha
        damp_span = 0.20 + 0.40 * alpha
        fric_span = 0.20 + 0.40 * alpha
        self.runtime = {
            "g": self.p.g * rng.uniform(1.0 - g_span, 1.0 + g_span),
            "m1": self.p.m1 * rng.uniform(1.0 - mass_span, 1.0 + mass_span),
            "m2": self.p.m2 * rng.uniform(1.0 - mass_span, 1.0 + mass_span),
            "m3": self.p.m3 * rng.uniform(1.0 - mass_span, 1.0 + mass_span),
            "l1": self.p.l1 * rng.uniform(1.0 - len_span, 1.0 + len_span),
            "l2": self.p.l2 * rng.uniform(1.0 - len_span, 1.0 + len_span),
            "l3": self.p.l3 * rng.uniform(1.0 - len_span, 1.0 + len_span),
            "damping": self.p.damping * rng.uniform(1.0 - damp_span, 1.0 + damp_span),
            "joint_coulomb": self.p.joint_coulomb * rng.uniform(1.0 - fric_span, 1.0 + fric_span),
            "cart_friction": self.p.cart_friction * rng.uniform(1.0 - fric_span, 1.0 + fric_span),
        }

    def set_training_progress(self, progress: float):
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def get_state(self):
        return self.state.copy()

    def _apply_actuator(self, u_cmd: float) -> float:
        dt = self.p.dt
        tau = max(self.p.actuator_tau, 1e-3)
        target_rate = (u_cmd - self.u_actual) / tau
        max_jerk_step = self.p.max_u_jerk * dt
        self.u_rate = float(
            np.clip(
                target_rate,
                self.u_rate - max_jerk_step,
                self.u_rate + max_jerk_step,
            )
        )
        self.u_rate = float(np.clip(self.u_rate, -self.p.max_u_rate, self.p.max_u_rate))
        self.u_actual = float(np.clip(self.u_actual + self.u_rate * dt, -self.p.max_u, self.p.max_u))
        return self.u_actual

    def _get_clean_obs(self) -> np.ndarray:
        x, th1, th2, th3, x_dot, w1, w2, w3 = self.state
        e = wrap_to_pi(np.array([th1 - np.pi, th2 - np.pi, th3 - np.pi], dtype=np.float64))
        obs = np.array([x, e[0], e[1], e[2], x_dot, w1, w2, w3, self.u_actual], dtype=np.float32)
        return obs

    def _delayed_noisy_obs(self) -> np.ndarray:
        delayed = self.obs_buffer[0].copy()
        if self.p.obs_noise_std > 0.0:
            noise_scales = np.array(
                [0.02, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.02],
                dtype=np.float32,
            )
            delayed += self.np_random.normal(0.0, self.p.obs_noise_std, size=delayed.shape).astype(np.float32) * noise_scales
        return delayed.astype(np.float32)

    def _dynamics(self, z: np.ndarray, a_base: float) -> np.ndarray:
        x, th1, th2, th3, x_dot, w1, w2, w3 = z
        p = self.runtime

        d12 = th1 - th2
        d13 = th1 - th3
        d23 = th2 - th3

        # Nonlinear coupled inertia matrix for absolute-angle coordinates.
        m11 = (p["m1"] + p["m2"] + p["m3"]) * p["l1"]
        m22 = (p["m2"] + p["m3"]) * p["l2"]
        m33 = p["m3"] * p["l3"]
        m12 = (p["m2"] + p["m3"]) * p["l2"] * np.cos(d12)
        m13 = p["m3"] * p["l3"] * np.cos(d13)
        m23 = p["m3"] * p["l3"] * np.cos(d23)
        M = np.array(
            [
                [m11, m12, m13],
                [m12, m22, m23],
                [m13, m23, m33],
            ],
            dtype=np.float64,
        )

        # Drift terms (gravity + velocity couplings + viscous damping).
        f1 = (
            -(p["m1"] + p["m2"] + p["m3"]) * p["g"] * np.sin(th1)
            - (p["m2"] + p["m3"]) * p["l2"] * w2 * w2 * np.sin(d12)
            - p["m3"] * p["l3"] * w3 * w3 * np.sin(d13)
            - p["damping"] * w1
            - p["joint_coulomb"] * np.tanh(30.0 * w1)
        )
        f2 = (
            -(p["m2"] + p["m3"]) * p["g"] * np.sin(th2)
            + (p["m2"] + p["m3"]) * p["l1"] * w1 * w1 * np.sin(d12)
            - p["m3"] * p["l3"] * w3 * w3 * np.sin(d23)
            - p["damping"] * w2
            - p["joint_coulomb"] * np.tanh(30.0 * w2)
        )
        f3 = (
            -p["m3"] * p["g"] * np.sin(th3)
            + p["m3"] * p["l1"] * w1 * w1 * np.sin(d13)
            + p["m3"] * p["l2"] * w2 * w2 * np.sin(d23)
            - p["damping"] * w3
            - p["joint_coulomb"] * np.tanh(30.0 * w3)
        )

        # Base acceleration generalized forcing: cart acceleration couples
        # through cosine projection to each joint.
        f1 += -(p["m1"] + p["m2"] + p["m3"]) * a_base * np.cos(th1)
        f2 += -(p["m2"] + p["m3"]) * a_base * np.cos(th2)
        f3 += -p["m3"] * a_base * np.cos(th3)

        rhs = np.array([f1, f2, f3], dtype=np.float64)
        alpha = np.linalg.solve(M, rhs)

        dz = np.array([x_dot, w1, w2, w3, a_base, alpha[0], alpha[1], alpha[2]], dtype=np.float64)
        return dz

    def _base_accel_from_force(self, force: float, x_dot: float) -> float:
        p = self.runtime
        cart_drag = p["cart_friction"] * np.tanh(12.0 * x_dot)
        return (force - cart_drag) / max(self.p.m_cart, 1e-6)

    def _dynamics_with_force(self, z: np.ndarray, force: float) -> np.ndarray:
        a_base = self._base_accel_from_force(force, z[4])
        return self._dynamics(z, a_base)

    def _rk4_step(self, z: np.ndarray, force: float, dt: float) -> np.ndarray:
        k1 = self._dynamics_with_force(z, force)
        k2 = self._dynamics_with_force(z + 0.5 * dt * k1, force)
        k3 = self._dynamics_with_force(z + 0.5 * dt * k2, force)
        k4 = self._dynamics_with_force(z + dt * k3, force)
        z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z_next

    def _reward(self, obs: np.ndarray, u: float) -> float:
        x, e1, e2, e3, x_dot, w1, w2, w3, _ = obs
        e_vec = np.array([e1, e2, e3], dtype=np.float64)
        w_vec = np.array([w1, w2, w3], dtype=np.float64)
        r = 0.0
        r -= self.p.w_angle * float(e_vec @ e_vec)
        r -= self.p.w_ang_vel * float(w_vec @ w_vec)
        r -= self.p.w_x * (x * x)
        r -= self.p.w_xdot * (x_dot * x_dot)
        u_norm = u / max(self.p.max_u, 1e-6)
        du_norm = (u - self.u_prev) / max(self.p.max_u, 1e-6)
        r -= self.p.w_u * (u_norm * u_norm)
        r -= self.p.smooth_u_weight * (du_norm * du_norm)
        # Smooth upright shaping instead of hard threshold-only reward.
        r += self.p.w_upright_shape * np.exp(-float(e_vec @ e_vec) / (self.p.upright_sigma**2))
        r += self.p.alive_bonus

        return float(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._sample_runtime_params()

        # Curriculum reset: broaden perturbations as training progresses.
        alpha = float(np.clip(self.training_progress, 0.0, 1.0)) if self.p.use_curriculum else 1.0
        e_span = self.p.reset_e_min + (self.p.reset_e_max - self.p.reset_e_min) * alpha
        x_span = self.p.reset_x_min + (self.p.reset_x_max - self.p.reset_x_min) * alpha
        xd_span = self.p.reset_xd_min + (self.p.reset_xd_max - self.p.reset_xd_min) * alpha
        w_span = self.p.reset_w_min + (self.p.reset_w_max - self.p.reset_w_min) * alpha

        e0 = self.np_random.uniform(low=-e_span, high=e_span, size=(3,))
        x0 = self.np_random.uniform(low=-x_span, high=x_span)
        xd0 = self.np_random.uniform(low=-xd_span, high=xd_span)
        w0 = self.np_random.uniform(low=-w_span, high=w_span, size=(3,))

        self.state = np.array(
            [
                x0,
                np.pi + e0[0],
                np.pi + e0[1],
                np.pi + e0[2],
                xd0,
                w0[0],
                w0[1],
                w0[2],
            ],
            dtype=np.float64,
        )
        self.u_actual = 0.0
        self.u_rate = 0.0
        self.u_prev = 0.0
        self.obs_buffer.clear()
        clean = self._get_clean_obs()
        for _ in range(self.p.obs_delay_steps + 1):
            self.obs_buffer.append(clean.copy())
        return self._delayed_noisy_obs(), {}

    def step(self, action):
        self.steps += 1
        u_cmd = float(np.clip(action[0], -self.p.max_u, self.p.max_u))
        u = self._apply_actuator(u_cmd)
        a_base = self._base_accel_from_force(u, self.state[4])

        z = self.state
        dt = self.p.dt / self.p.substeps
        for _ in range(self.p.substeps):
            z = self._rk4_step(z, u, dt)
        self.state = z

        clean_obs = self._get_clean_obs()
        reward = self._reward(clean_obs, u)
        self.u_prev = u
        self.obs_buffer.append(clean_obs.copy())
        obs = self._delayed_noisy_obs()

        x = float(clean_obs[0])
        terminated = abs(x) > self.p.x_limit or np.any(~np.isfinite(clean_obs))
        truncated = self.steps >= self.p.max_steps
        max_abs_e = float(np.max(np.abs(clean_obs[1:4])))
        if terminated:
            remaining_frac = max(0.0, (self.p.max_steps - self.steps) / max(1, self.p.max_steps))
            # Penalize early failures more strongly to avoid "fail-fast" reward hacking.
            reward -= self.p.fail_x_penalty * (1.0 + remaining_frac)
        elif truncated and max_abs_e < 0.18:
            reward += self.p.horizon_balance_bonus
        info = {
            "u_cmd": u_cmd,
            "u_applied": u,  # applied cart force
            "a_base": a_base,
            "force_cmd": u_cmd,
            "force_applied": u,
            "max_abs_e": max_abs_e,
            "is_balanced": bool(max_abs_e < 0.18),
            "sat_flag": bool(abs(u) > 0.98 * self.p.max_u),
        }
        return obs, reward, terminated, truncated, info


def rollout(
    model: PPO,
    env: TriplePendulumUprightEnv,
    horizon: int = 1800,
    frame_stack: int = 1,
    obs_transform=None,
):
    obs, _ = env.reset()
    traj = []
    u_hist = []
    r_hist = []
    done = False
    balanced_steps = 0
    sat_steps = 0
    max_cart = 0.0

    stack = deque(maxlen=max(1, frame_stack))
    first_obs = obs_transform(obs.copy()) if obs_transform is not None else obs.copy()
    for _ in range(max(1, frame_stack)):
        stack.append(first_obs.copy())

    for _ in range(horizon):
        if done:
            break
        policy_obs = np.concatenate(list(stack), axis=0)
        act, _ = model.predict(policy_obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(act)
        obs_t = obs_transform(obs.copy()) if obs_transform is not None else obs.copy()
        stack.append(obs_t)
        u_hist.append(float(info.get("u_applied", act[0])))
        r_hist.append(float(r))
        traj.append(env.state.copy())
        balanced_steps += int(info.get("is_balanced", False))
        sat_steps += int(info.get("sat_flag", False))
        max_cart = max(max_cart, abs(float(env.state[0])))
        done = terminated or truncated

    steps = max(1, len(r_hist))
    metrics = {
        "balanced_fraction": balanced_steps / steps,
        "saturation_fraction": sat_steps / steps,
        "max_cart_abs": max_cart,
        "mean_abs_u": float(np.mean(np.abs(u_hist))) if len(u_hist) > 0 else 0.0,
    }
    return np.array(traj), np.array(u_hist), np.array(r_hist), metrics


def traj_to_points(traj: np.ndarray, p: TriplePendulumParams):
    x_cart = traj[:, 0]
    th1 = traj[:, 1]
    th2 = traj[:, 2]
    th3 = traj[:, 3]

    x1 = x_cart + p.l1 * np.sin(th1)
    y1 = -p.l1 * np.cos(th1)
    x2 = x1 + p.l2 * np.sin(th2)
    y2 = y1 - p.l2 * np.cos(th2)
    x3 = x2 + p.l3 * np.sin(th3)
    y3 = y2 - p.l3 * np.cos(th3)
    return x_cart, th1, th2, th3, x1, y1, x2, y2, x3, y3


class LiveTrainingVizCallback(BaseCallback):
    """
    Live visualization during PPO training.
    Every `eval_every_steps`, runs a short deterministic rollout and updates:
      1) reward trend
      2) angle/control traces for the current policy
      3) animated pendulum rollout preview
    """

    def __init__(
        self,
        eval_env: TriplePendulumUprightEnv,
        eval_every_steps: int = 10_000,
        horizon: int = 700,
        preview_count: int = 4,
        frame_stack: int = 1,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every_steps = eval_every_steps
        self.horizon = horizon
        self.preview_count = max(1, preview_count)
        self.frame_stack = max(1, frame_stack)
        self.next_eval = eval_every_steps
        self.eval_returns = []
        self.eval_steps = []

        self.fig = None
        self.ax_reward = None
        self.ax_traces = None
        self.ax_anim = None
        self.reward_line = None
        self.angle_lines = None
        self.control_line = None
        self.pend_line = None
        self.pivot_dot = None
        self.tip_trace = None
        self.status_text = None
        self.preview_envs = []
        self.preview_fig = None
        self.preview_axes = []

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        current = self.training_env
        while current is not None:
            if isinstance(current, VecNormalize):
                return current.normalize_obs(obs[np.newaxis, ...])[0]
            current = getattr(current, "venv", None)
        return obs

    def _on_training_start(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        gs = self.fig.add_gridspec(2, 2)
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_traces = self.fig.add_subplot(gs[1, 0])
        self.ax_anim = self.fig.add_subplot(gs[:, 1])

        self.reward_line, = self.ax_reward.plot([], [], lw=2.0, label="Eval return")
        self.ax_reward.set_title("Learning progress")
        self.ax_reward.set_xlabel("Timesteps")
        self.ax_reward.set_ylabel("Return")
        self.ax_reward.grid(True, alpha=0.3)
        self.ax_reward.legend()

        self.angle_lines = [
            self.ax_traces.plot([], [], lw=1.8, label="e1")[0],
            self.ax_traces.plot([], [], lw=1.8, label="e2")[0],
            self.ax_traces.plot([], [], lw=1.8, label="e3")[0],
        ]
        self.control_line = self.ax_traces.plot([], [], lw=1.4, alpha=0.8, label="u")[0]
        self.ax_traces.set_title("Current policy rollout (angles and control)")
        self.ax_traces.set_xlabel("time [s]")
        self.ax_traces.set_ylabel("value")
        self.ax_traces.grid(True, alpha=0.3)
        self.ax_traces.legend(loc="upper right")

        self.ax_anim.set_title("Live stabilization preview")
        self.ax_anim.set_aspect("equal")
        self.ax_anim.grid(True, alpha=0.3)
        self.pend_line, = self.ax_anim.plot([], [], "o-", lw=2.2)
        self.pivot_dot, = self.ax_anim.plot([], [], "ro", ms=5)
        self.tip_trace, = self.ax_anim.plot([], [], lw=1.2, alpha=0.8)
        self.status_text = self.ax_anim.text(0.02, 0.98, "", transform=self.ax_anim.transAxes, va="top")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

        # Side-by-side preview panel showing multiple rollouts.
        if self.preview_count > 1:
            rows = int(np.ceil(np.sqrt(self.preview_count)))
            cols = int(np.ceil(self.preview_count / rows))
            self.preview_fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
            axes = np.array(axes).reshape(-1)
            self.preview_axes = list(axes)
            for ax in self.preview_axes:
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.25)
            for ax in self.preview_axes[self.preview_count :]:
                ax.set_visible(False)
            self.preview_fig.suptitle("Parallel policy previews (different initial states)")
            self.preview_fig.tight_layout()
            self.preview_fig.canvas.draw_idle()
            plt.pause(0.001)

            # Independent eval envs for parallel-preview diversity.
            for i in range(self.preview_count):
                p_i = TriplePendulumParams(**vars(self.eval_env.p))
                env_i = TriplePendulumUprightEnv(p_i)
                env_i.reset(seed=1234 + i)
                self.preview_envs.append(env_i)

    def _update_live_view(self):
        traj, u_hist, r_hist, metrics = rollout(
            self.model,
            self.eval_env,
            horizon=self.horizon,
            frame_stack=self.frame_stack,
            obs_transform=self._obs_transform,
        )
        if len(traj) == 0:
            return

        total_return = float(np.sum(r_hist))
        self.eval_returns.append(total_return)
        self.eval_steps.append(int(self.num_timesteps))

        # Reward curve
        self.reward_line.set_data(self.eval_steps, self.eval_returns)
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        # Trace plot
        p = self.eval_env.p
        dt = p.dt
        t = np.arange(len(traj)) * dt
        _, th1, th2, th3, _, _, _, _, _, _ = traj_to_points(traj, p)
        e = wrap_to_pi(np.column_stack([th1 - np.pi, th2 - np.pi, th3 - np.pi]))
        self.angle_lines[0].set_data(t, e[:, 0])
        self.angle_lines[1].set_data(t, e[:, 1])
        self.angle_lines[2].set_data(t, e[:, 2])
        self.control_line.set_data(t[: len(u_hist)], u_hist)
        self.ax_traces.relim()
        self.ax_traces.autoscale_view()

        # Animation panel
        x_cart, _, _, _, x1, y1, x2, y2, x3, y3 = traj_to_points(traj, p)
        pad = p.l1 + p.l2 + p.l3 + 1.0
        self.ax_anim.set_xlim(np.min(x_cart) - pad, np.max(x_cart) + pad)
        self.ax_anim.set_ylim(-(p.l1 + p.l2 + p.l3) - 0.8, 1.5)
        trace_x = []
        trace_y = []
        stride = max(1, len(traj) // 220)
        for k in range(0, len(traj), stride):
            xs = [x_cart[k], x1[k], x2[k], x3[k]]
            ys = [0.0, y1[k], y2[k], y3[k]]
            self.pend_line.set_data(xs, ys)
            self.pivot_dot.set_data([x_cart[k]], [0.0])
            trace_x.append(x3[k])
            trace_y.append(y3[k])
            self.tip_trace.set_data(trace_x, trace_y)
            self.status_text.set_text(
                f"step={self.num_timesteps}\n"
                f"eval_return={total_return:.1f}\n"
                f"mean_step_r={np.mean(r_hist):.2f}\n"
                f"bal_frac={metrics['balanced_fraction']:.2f}\n"
                f"sat_frac={metrics['saturation_fraction']:.2f}"
            )
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

        # Multi-preview update (periodic snapshots across initial conditions).
        if self.preview_fig is not None and self.preview_axes:
            for i in range(self.preview_count):
                axp = self.preview_axes[i]
                traj_i, _, r_i, m_i = rollout(
                    self.model,
                    self.preview_envs[i],
                    horizon=min(self.horizon, 500),
                    frame_stack=self.frame_stack,
                    obs_transform=self._obs_transform,
                )
                axp.clear()
                axp.grid(True, alpha=0.25)
                axp.set_aspect("equal")

                if len(traj_i) == 0:
                    axp.set_title(f"Preview {i + 1}: empty")
                    continue

                x_cart_i, _, _, _, x1_i, y1_i, x2_i, y2_i, x3_i, y3_i = traj_to_points(traj_i, self.preview_envs[i].p)
                pad_i = self.preview_envs[i].p.l1 + self.preview_envs[i].p.l2 + self.preview_envs[i].p.l3 + 0.8
                axp.set_xlim(np.min(x_cart_i) - pad_i, np.max(x_cart_i) + pad_i)
                axp.set_ylim(-(self.preview_envs[i].p.l1 + self.preview_envs[i].p.l2 + self.preview_envs[i].p.l3) - 0.6, 1.3)

                # Tip trace and final pose of this rollout.
                axp.plot(x3_i, y3_i, lw=1.0, alpha=0.75)
                k = -1
                axp.plot(
                    [x_cart_i[k], x1_i[k], x2_i[k], x3_i[k]],
                    [0.0, y1_i[k], y2_i[k], y3_i[k]],
                    "o-",
                    lw=1.8,
                )
                axp.plot([x_cart_i[k]], [0.0], "ro", ms=4)
                axp.set_title(
                    f"P{i + 1}  R={np.sum(r_i):.0f}  B={m_i['balanced_fraction']:.2f}",
                    fontsize=9,
                )

            self.preview_fig.canvas.draw_idle()
            plt.pause(0.001)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._update_live_view()
            self.next_eval += self.eval_every_steps
        return True

    def _on_training_end(self) -> None:
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        if self.preview_fig is not None:
            self.preview_fig.canvas.draw_idle()
            plt.pause(0.001)


class TrainProgressCallback(BaseCallback):
    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = max(1, int(total_steps))

    def _on_step(self) -> bool:
        progress = float(np.clip(self.num_timesteps / self.total_steps, 0.0, 1.0))
        # Propagate curriculum progress into all parallel env instances.
        self.training_env.env_method("set_training_progress", progress)
        return True


class FixedSeedEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_params: TriplePendulumParams,
        eval_every_steps: int,
        seeds: list[int],
        horizon: int,
        log_file: Path,
        frame_stack: int = 1,
    ):
        super().__init__()
        self.eval_params = eval_params
        self.eval_every_steps = max(1, eval_every_steps)
        self.seeds = seeds
        self.horizon = horizon
        self.log_file = log_file
        self.frame_stack = max(1, frame_stack)
        self.next_eval = self.eval_every_steps

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        current = self.training_env
        while current is not None:
            if isinstance(current, VecNormalize):
                return current.normalize_obs(obs[np.newaxis, ...])[0]
            current = getattr(current, "venv", None)
        return obs

    def _on_training_start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.write_text(
                "timesteps,mean_return,success_rate,mean_balanced_fraction,mean_sat_fraction,mean_max_cart,mean_abs_u\n",
                encoding="utf-8",
            )

    def _evaluate(self):
        returns = []
        success = 0
        balanced_fracs = []
        sat_fracs = []
        max_carts = []
        mean_abs_us = []

        for seed in self.seeds:
            env = TriplePendulumUprightEnv(TriplePendulumParams(**vars(self.eval_params)))
            obs, _ = env.reset(seed=seed)
            done = False
            ep_ret = 0.0
            last_obs = obs
            stack = deque(maxlen=self.frame_stack)
            first = self._obs_transform(obs.copy())
            for _ in range(self.frame_stack):
                stack.append(first.copy())

            balanced_steps = 0
            sat_steps = 0
            steps = 0
            abs_us = []
            max_cart = 0.0

            while not done and steps < self.horizon:
                policy_obs = np.concatenate(list(stack), axis=0)
                act, _ = self.model.predict(policy_obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(act)
                stack.append(self._obs_transform(obs.copy()))
                ep_ret += float(r)
                steps += 1
                done = terminated or truncated
                last_obs = obs
                balanced_steps += int(info.get("is_balanced", False))
                sat_steps += int(info.get("sat_flag", False))
                abs_us.append(abs(float(info.get("u_applied", act[0]))))
                max_cart = max(max_cart, abs(float(env.state[0])))

            returns.append(ep_ret)
            e = np.abs(last_obs[1:4])
            if (steps >= self.horizon) and (np.max(e) < 0.18):
                success += 1
            denom = max(1, steps)
            balanced_fracs.append(balanced_steps / denom)
            sat_fracs.append(sat_steps / denom)
            max_carts.append(max_cart)
            mean_abs_us.append(float(np.mean(abs_us)) if abs_us else 0.0)

        row = (
            f"{self.num_timesteps},"
            f"{np.mean(returns):.6f},"
            f"{success / len(self.seeds):.6f},"
            f"{np.mean(balanced_fracs):.6f},"
            f"{np.mean(sat_fracs):.6f},"
            f"{np.mean(max_carts):.6f},"
            f"{np.mean(mean_abs_us):.6f}\n"
        )
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(row)
        print(
            "[fixed-eval] "
            f"step={self.num_timesteps} "
            f"ret={np.mean(returns):.1f} "
            f"succ={success / len(self.seeds):.2f} "
            f"bal={np.mean(balanced_fracs):.2f} "
            f"sat={np.mean(sat_fracs):.2f}"
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.eval_every_steps
        return True


def plot_training_curve(log_file: Path):
    if not log_file.exists():
        print(f"Monitor file not found: {log_file}")
        return

    data = np.genfromtxt(log_file, delimiter=",", names=True, skip_header=1)
    if data.size == 0:
        print("Monitor file exists but has no episodes yet.")
        return

    if data.ndim == 0:
        rewards = np.array([float(data["r"])])
    else:
        rewards = data["r"]

    window = min(50, len(rewards))
    if window >= 2:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed = rewards

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.4, label="Episode reward")
    if len(smoothed) > 0:
        x = np.arange(window - 1, window - 1 + len(smoothed))
        ax.plot(x, smoothed, linewidth=2.0, label=f"Moving avg ({window})")
    ax.set_title("RL Training Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_rollout(traj: np.ndarray, u_hist: np.ndarray, dt: float, p: TriplePendulumParams):
    if len(traj) == 0:
        print("Empty rollout, nothing to animate.")
        return

    x_cart, th1, th2, th3, x1, y1, x2, y2, x3, y3 = traj_to_points(traj, p)

    t = np.arange(len(traj)) * dt
    e = wrap_to_pi(np.column_stack([th1 - np.pi, th2 - np.pi, th3 - np.pi]))

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(t, e[:, 0], label="e1")
    axs[0].plot(t, e[:, 1], label="e2")
    axs[0].plot(t, e[:, 2], label="e3")
    axs[0].set_ylabel("angle error [rad]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(t[: len(u_hist)], u_hist, label="u_applied = cart force")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("control")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    plt.tight_layout()

    fig2, ax = plt.subplots(figsize=(7, 6))
    pad = p.l1 + p.l2 + p.l3 + 1.0
    ax.set_xlim(np.min(x_cart) - pad, np.max(x_cart) + pad)
    ax.set_ylim(-(p.l1 + p.l2 + p.l3) - 0.8, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Trained RL policy rollout")

    line, = ax.plot([], [], "o-", lw=2)
    pivot, = ax.plot([], [], "ro", ms=6)
    trace, = ax.plot([], [], lw=1.2, alpha=0.8)
    trace_x, trace_y = [], []

    def update(frame):
        xs = [x_cart[frame], x1[frame], x2[frame], x3[frame]]
        ys = [0.0, y1[frame], y2[frame], y3[frame]]

        line.set_data(xs, ys)
        pivot.set_data([x_cart[frame]], [0.0])

        trace_x.append(x3[frame])
        trace_y.append(y3[frame])
        trace.set_data(trace_x, trace_y)
        return line, pivot, trace

    FuncAnimation(fig2, update, frames=len(traj), interval=max(10, int(1000 * dt)), blit=True)
    plt.show()


def linearize_env_dynamics(params: TriplePendulumParams):
    env = TriplePendulumUprightEnv(params)
    env._sample_runtime_params()
    z0 = np.array([0.0, np.pi, np.pi, np.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u0 = 0.0

    n = z0.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((n, 1), dtype=np.float64)
    eps_x = 1e-6
    eps_u = 1e-6

    for i in range(n):
        dz = np.zeros_like(z0)
        dz[i] = eps_x
        f_plus = env._dynamics_with_force(z0 + dz, u0)
        f_minus = env._dynamics_with_force(z0 - dz, u0)
        A[:, i] = (f_plus - f_minus) / (2.0 * eps_x)

    f_plus_u = env._dynamics_with_force(z0, u0 + eps_u)
    f_minus_u = env._dynamics_with_force(z0, u0 - eps_u)
    B[:, 0] = (f_plus_u - f_minus_u) / (2.0 * eps_u)
    return A, B


def compute_lqr_gain(A: np.ndarray, B: np.ndarray):
    # Weights tuned for local upright stabilization around small errors.
    Q = np.diag([30.0, 700.0, 700.0, 700.0, 20.0, 100.0, 100.0, 100.0])
    R = np.array([[0.8]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def lqr_policy_cmd(K: np.ndarray, state: np.ndarray, max_u: float):
    x, th1, th2, th3, x_dot, w1, w2, w3 = state
    e = wrap_to_pi(np.array([th1 - np.pi, th2 - np.pi, th3 - np.pi], dtype=np.float64))
    z_err = np.array([x, e[0], e[1], e[2], x_dot, w1, w2, w3], dtype=np.float64)
    u_cmd = float(-(K @ z_err)[0])
    return float(np.clip(u_cmd, -max_u, max_u))


def run_sanity_lqr(
    episodes: int = 20,
    horizon: int = 1200,
    do_viz: bool = True,
):
    # Keep this near-ideal to test local controllability first.
    p_base = TriplePendulumParams(
        randomize_domain=False,
        obs_delay_steps=0,
        obs_noise_std=0.0,
        use_curriculum=False,
        actuator_tau=0.0,
        max_u_rate=1e9,
        max_u_jerk=1e9,
        joint_coulomb=0.0,
        cart_friction=0.0,
        damping=0.02,
    )
    A, B = linearize_env_dynamics(p_base)
    eig = np.linalg.eigvals(A)
    ctrb = B
    for i in range(1, A.shape[0]):
        ctrb = np.hstack([ctrb, np.linalg.matrix_power(A, i) @ B])
    print("[sanity_lqr] controllability_rank:", int(np.linalg.matrix_rank(ctrb)), "/", A.shape[0])
    print("[sanity_lqr] max_real_eig(A):", float(np.max(np.real(eig))))
    K = compute_lqr_gain(A, B)

    def set_small_initial_state(env: TriplePendulumUprightEnv, seed: int):
        rng = np.random.default_rng(seed)
        e0 = rng.uniform(-0.04, 0.04, size=(3,))
        x0 = float(rng.uniform(-0.03, 0.03))
        xd0 = float(rng.uniform(-0.03, 0.03))
        w0 = rng.uniform(-0.05, 0.05, size=(3,))
        env.state = np.array([x0, np.pi + e0[0], np.pi + e0[1], np.pi + e0[2], xd0, w0[0], w0[1], w0[2]], dtype=np.float64)
        env.u_actual = 0.0
        env.u_rate = 0.0
        env.u_prev = 0.0
        env.obs_buffer.clear()
        clean = env._get_clean_obs()
        for _ in range(env.p.obs_delay_steps + 1):
            env.obs_buffer.append(clean.copy())

    def evaluate_case(label: str, params: TriplePendulumParams):
        env = TriplePendulumUprightEnv(params)
        returns = []
        successes = 0
        balanced_fracs = []
        max_carts = []
        sat_fracs = []
        best = None

        for ep in range(episodes):
            env.reset(seed=7000 + ep)
            set_small_initial_state(env, seed=91000 + ep)
            obs = env._delayed_noisy_obs()
            done = False
            ep_ret = 0.0
            steps = 0
            balanced_steps = 0
            sat_steps = 0
            max_cart = 0.0
            traj = []
            u_hist = []
            last_obs = obs

            while not done and steps < horizon:
                u_cmd = lqr_policy_cmd(K, env.get_state(), params.max_u)
                obs, r, terminated, truncated, info = env.step(np.array([u_cmd], dtype=np.float32))
                ep_ret += float(r)
                steps += 1
                last_obs = obs
                done = terminated or truncated
                balanced_steps += int(info.get("is_balanced", False))
                sat_steps += int(info.get("sat_flag", False))
                max_cart = max(max_cart, abs(float(env.state[0])))
                traj.append(env.state.copy())
                u_hist.append(float(info.get("u_applied", u_cmd)))

            e_last = np.abs(last_obs[1:4])
            if steps >= horizon and np.max(e_last) < 0.18:
                successes += 1
            returns.append(ep_ret)
            balanced_fracs.append(balanced_steps / max(1, steps))
            sat_fracs.append(sat_steps / max(1, steps))
            max_carts.append(max_cart)
            if best is None or ep_ret > best["ret"]:
                best = {"ret": ep_ret, "traj": np.array(traj), "u_hist": np.array(u_hist)}

        print(f"[sanity_lqr:{label}] episodes:", episodes)
        print(f"[sanity_lqr:{label}] success_rate:", successes / episodes)
        print(f"[sanity_lqr:{label}] mean_return:", float(np.mean(returns)))
        print(f"[sanity_lqr:{label}] mean_balanced_fraction:", float(np.mean(balanced_fracs)))
        print(f"[sanity_lqr:{label}] mean_saturation_fraction:", float(np.mean(sat_fracs)))
        print(f"[sanity_lqr:{label}] mean_max_cart:", float(np.mean(max_carts)))
        return best, params

    best_nom, p_nom = evaluate_case(
        "nominal_limits",
        TriplePendulumParams(**{**vars(p_base), "max_u": 18.0, "x_limit": 4.0}),
    )
    best_relaxed, p_rel = evaluate_case(
        "relaxed_limits",
        TriplePendulumParams(**{**vars(p_base), "max_u": 60.0, "x_limit": 10.0}),
    )

    if do_viz:
        if best_nom is not None and len(best_nom["traj"]) > 0:
            animate_rollout(best_nom["traj"], best_nom["u_hist"], p_nom.dt, p_nom)
        if best_relaxed is not None and len(best_relaxed["traj"]) > 0:
            animate_rollout(best_relaxed["traj"], best_relaxed["u_hist"], p_rel.dt, p_rel)


def sweep_lqr_feasibility(episodes: int = 10, horizon: int = 1200):
    base = TriplePendulumParams(
        randomize_domain=False,
        obs_delay_steps=0,
        obs_noise_std=0.0,
        use_curriculum=False,
        actuator_tau=0.0,
        max_u_rate=1e9,
        max_u_jerk=1e9,
        joint_coulomb=0.0,
        cart_friction=0.0,
        damping=0.02,
    )
    A, B = linearize_env_dynamics(base)
    K = compute_lqr_gain(A, B)

    max_u_grid = [18.0, 24.0, 30.0, 40.0, 60.0, 80.0]
    x_limit_grid = [3.0, 4.0, 6.0, 8.0, 10.0, 14.0]
    rows = []

    for mu in max_u_grid:
        for xl in x_limit_grid:
            p = TriplePendulumParams(**{**vars(base), "max_u": mu, "x_limit": xl})
            env = TriplePendulumUprightEnv(p)
            succ = 0
            bal = []
            sat = []
            mx = []
            for ep in range(episodes):
                env.reset(seed=10000 + ep)
                rng = np.random.default_rng(120000 + ep)
                e0 = rng.uniform(-0.04, 0.04, size=(3,))
                x0 = float(rng.uniform(-0.03, 0.03))
                xd0 = float(rng.uniform(-0.03, 0.03))
                w0 = rng.uniform(-0.05, 0.05, size=(3,))
                env.state = np.array([x0, np.pi + e0[0], np.pi + e0[1], np.pi + e0[2], xd0, w0[0], w0[1], w0[2]], dtype=np.float64)
                env.u_actual = 0.0
                env.u_rate = 0.0
                env.u_prev = 0.0
                env.obs_buffer.clear()
                clean = env._get_clean_obs()
                for _ in range(env.p.obs_delay_steps + 1):
                    env.obs_buffer.append(clean.copy())

                steps = 0
                b = 0
                s = 0
                done = False
                last = clean
                while not done and steps < horizon:
                    u = lqr_policy_cmd(K, env.get_state(), p.max_u)
                    obs, _, term, trunc, info = env.step(np.array([u], dtype=np.float32))
                    done = term or trunc
                    steps += 1
                    b += int(info.get("is_balanced", False))
                    s += int(info.get("sat_flag", False))
                    last = obs
                if steps >= horizon and np.max(np.abs(last[1:4])) < 0.18:
                    succ += 1
                bal.append(b / max(1, steps))
                sat.append(s / max(1, steps))
                mx.append(abs(float(env.state[0])))

            row = {
                "max_force": mu,
                "x_limit": xl,
                "success_rate": succ / episodes,
                "balanced_fraction": float(np.mean(bal)),
                "saturation_fraction": float(np.mean(sat)),
                "mean_abs_x_end": float(np.mean(mx)),
            }
            rows.append(row)
            print(
                f"[lqr-sweep] max_force={mu:>5.1f} x_limit={xl:>4.1f} "
                f"succ={row['success_rate']:.2f} bal={row['balanced_fraction']:.2f} "
                f"sat={row['saturation_fraction']:.2f} x_end={row['mean_abs_x_end']:.2f}"
            )

    rows_sorted = sorted(rows, key=lambda r: (r["success_rate"], r["balanced_fraction"]), reverse=True)
    print("[lqr-sweep] top configurations:")
    for r in rows_sorted[:8]:
        print(
            f"  max_force={r['max_force']:.1f}, x_limit={r['x_limit']:.1f}, "
            f"succ={r['success_rate']:.2f}, bal={r['balanced_fraction']:.2f}, "
            f"sat={r['saturation_fraction']:.2f}"
        )


def make_env_fn(params_template: TriplePendulumParams, seed: int, rank: int):
    def _init():
        # Create an independent params object per worker env.
        env_params = TriplePendulumParams(**vars(params_template))
        env = TriplePendulumUprightEnv(env_params)
        env.reset(seed=seed + rank)
        return env

    return _init


def train(
    total_steps: int,
    model_path: Path,
    log_dir: Path,
    seed: int,
    live: bool = False,
    eval_every: int = 10_000,
    num_envs: int = 1,
    preview_count: int = 4,
    frame_stack: int = 1,
    enable_fixed_eval: bool = True,
    eval_seed_count: int = 8,
    eval_horizon: int = 1200,
    load_model_path: Path | None = None,
    train_params_override: TriplePendulumParams | None = None,
    eval_params_override: TriplePendulumParams | None = None,
):
    # Train with randomized physics for robustness, evaluate/preview on nominal physics.
    p_train = train_params_override if train_params_override is not None else TriplePendulumParams(randomize_domain=True)
    p_eval = eval_params_override if eval_params_override is not None else TriplePendulumParams(randomize_domain=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env_fn(p_train, seed=seed, rank=i) for i in range(max(1, num_envs))]
    if num_envs <= 1:
        vec_env_base = DummyVecEnv(env_fns)
    else:
        vec_env_base = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env_mon = VecMonitor(vec_env_base, filename=str(log_dir / "monitor.csv"))

    vecnorm_path_for_load = None
    if load_model_path is not None:
        candidate = load_model_path.with_suffix(".vecnormalize.pkl")
        if candidate.exists():
            vecnorm_path_for_load = candidate

    if vecnorm_path_for_load is not None:
        vec_env = VecNormalize.load(str(vecnorm_path_for_load), vec_env_mon)
        vec_env.training = True
        vec_env.norm_reward = True
        print(f"Loaded vecnormalize stats from {vecnorm_path_for_load}")
    else:
        vec_env = VecNormalize(vec_env_mon, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    if load_model_path is not None and load_model_path.exists():
        model = PPO.load(str(load_model_path), env=vec_env)
        print(f"Resuming from model: {load_model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=512,
            gamma=0.995,
            gae_lambda=0.97,
            ent_coef=0.001,
            clip_range=0.2,
            verbose=1,
            seed=seed,
        )
    callbacks = [TrainProgressCallback(total_steps=total_steps)]
    if enable_fixed_eval:
        seeds = [1000 + i for i in range(max(1, eval_seed_count))]
        callbacks.append(
            FixedSeedEvalCallback(
                eval_params=p_eval,
                eval_every_steps=eval_every,
                seeds=seeds,
                horizon=max(200, eval_horizon),
                log_file=log_dir / "fixed_eval_metrics.csv",
                frame_stack=frame_stack,
            )
        )
    if live:
        callbacks.append(
            LiveTrainingVizCallback(
                eval_env=TriplePendulumUprightEnv(p_eval),
                eval_every_steps=eval_every,
                horizon=700,
                preview_count=preview_count,
                frame_stack=frame_stack,
            )
        )
    model.learn(total_timesteps=total_steps, progress_bar=True, callback=callbacks)
    model.save(str(model_path))
    current = vec_env
    vecnorm = None
    while current is not None:
        if isinstance(current, VecNormalize):
            vecnorm = current
            break
        current = getattr(current, "venv", None)
    if vecnorm is not None:
        vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
        vecnorm.save(str(vecnorm_path))
        print(f"Saved vecnormalize stats to {vecnorm_path}")
    meta = {"frame_stack": int(frame_stack)}
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model meta to {meta_path}")
    vec_env.close()
    print(f"Saved model to {model_path}")


def visualize(model_path: Path, horizon: int):
    p = TriplePendulumParams()
    env = TriplePendulumUprightEnv(p)
    frame_stack = 1
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame_stack = max(1, int(meta.get("frame_stack", 1)))
        except Exception:
            frame_stack = 1

    obs_transform = None
    vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
    if vecnorm_path.exists():
        dummy = DummyVecEnv([lambda: TriplePendulumUprightEnv(TriplePendulumParams())])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        obs_transform = lambda obs: vecnorm.normalize_obs(obs[np.newaxis, ...])[0]
        dummy.close()
        print(f"Loaded vecnormalize stats from {vecnorm_path}")
    else:
        print("No vecnormalize stats found, using raw observations for visualization.")

    model = PPO.load(str(model_path))
    traj, u_hist, r_hist, metrics = rollout(
        model,
        env,
        horizon=horizon,
        frame_stack=frame_stack,
        obs_transform=obs_transform,
    )
    print(f"Rollout length: {len(traj)} steps")
    if len(r_hist) > 0:
        print(f"Mean step reward: {np.mean(r_hist):.3f}")
        print(
            f"Balanced fraction: {metrics['balanced_fraction']:.3f} | "
            f"Saturation fraction: {metrics['saturation_fraction']:.3f} | "
            f"Max |x|: {metrics['max_cart_abs']:.3f}"
        )
    animate_rollout(traj, u_hist, p.dt, p)


def main():
    parser = argparse.ArgumentParser(description="Train and visualize RL stabilization for nonlinear triple pendulum.")
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "train_fast",
            "train_debug_easy",
            "train_live",
            "viz",
            "train_viz",
            "curve",
            "sanity_lqr",
            "sanity_sweep",
        ],
        default="train_viz",
    )
    parser.add_argument("--steps", type=int, default=250_000, help="PPO training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=1800)
    parser.add_argument("--model", type=str, default="ppo_triple_pendulum_upright")
    parser.add_argument("--logdir", type=str, default="rl_logs")
    parser.add_argument("--eval-every", type=int, default=10_000, help="Live mode update interval in timesteps")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments for PPO training")
    parser.add_argument("--preview-count", type=int, default=4, help="Number of side-by-side live preview rollouts")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of stacked observations for policy memory")
    parser.add_argument("--eval-seeds", type=int, default=8, help="Fixed-eval seed count (set 0 to disable)")
    parser.add_argument("--eval-horizon", type=int, default=1200, help="Fixed-eval rollout horizon")
    parser.add_argument("--load-model", type=str, default="", help="Load existing model and continue training from it")
    parser.add_argument("--sanity-episodes", type=int, default=20, help="Episode count for sanity_lqr mode")
    parser.add_argument("--sanity-viz", action="store_true", help="Show visualization for best sanity_lqr episode")
    args = parser.parse_args()

    model_path = Path(args.model).with_suffix(".zip")
    log_dir = Path(args.logdir)
    load_model_path = Path(args.load_model).with_suffix(".zip") if args.load_model else None
    resolved_frame_stack = max(1, min(8, args.frame_stack))

    if load_model_path is not None:
        if not load_model_path.exists():
            raise FileNotFoundError(f"Load model not found: {load_model_path}")
        load_meta = load_model_path.with_suffix(".meta.json")
        if load_meta.exists():
            meta = json.loads(load_meta.read_text(encoding="utf-8"))
            loaded_fs = int(meta.get("frame_stack", resolved_frame_stack))
            if args.frame_stack != 1 and args.frame_stack != loaded_fs:
                raise ValueError(
                    f"Frame-stack mismatch: loaded model expects {loaded_fs}, "
                    f"but --frame-stack={args.frame_stack} was provided."
                )
            resolved_frame_stack = loaded_fs

    if args.mode in ("train", "train_fast", "train_debug_easy", "train_live", "train_viz"):
        fast_mode = args.mode == "train_fast"
        debug_easy = args.mode == "train_debug_easy"

        train_override = None
        eval_override = None
        if debug_easy:
            # Learnability check mode:
            # - no delay/noise/randomization
            # - easier reset curriculum
            # - lower smoothing penalty so policy can explore authority
            train_override = TriplePendulumParams(
                randomize_domain=False,
                obs_delay_steps=0,
                obs_noise_std=0.0,
                use_curriculum=False,
                max_u=40.0,
                actuator_tau=0.005,
                max_u_rate=1200.0,
                max_u_jerk=100000.0,
                smooth_u_weight=0.0,
                w_x=1.2,
                w_xdot=0.15,
                w_u=0.0,
                fail_x_penalty=5000.0,
                horizon_balance_bonus=1200.0,
                damping=0.02,
                joint_coulomb=0.0,
                cart_friction=0.0,
                reset_e_min=0.01,
                reset_e_max=0.04,
                reset_x_min=0.01,
                reset_x_max=0.03,
                reset_xd_min=0.005,
                reset_xd_max=0.03,
                reset_w_min=0.01,
                reset_w_max=0.04,
            )
            eval_override = TriplePendulumParams(
                randomize_domain=False,
                obs_delay_steps=0,
                obs_noise_std=0.0,
                use_curriculum=False,
                max_u=40.0,
                actuator_tau=0.005,
                max_u_rate=1200.0,
                max_u_jerk=100000.0,
                smooth_u_weight=0.0,
                w_x=1.2,
                w_xdot=0.15,
                w_u=0.0,
                fail_x_penalty=5000.0,
                horizon_balance_bonus=1200.0,
                damping=0.02,
                joint_coulomb=0.0,
                cart_friction=0.0,
                reset_e_min=0.01,
                reset_e_max=0.04,
                reset_x_min=0.01,
                reset_x_max=0.03,
                reset_xd_min=0.005,
                reset_xd_max=0.03,
                reset_w_min=0.01,
                reset_w_max=0.04,
            )
        train(
            total_steps=args.steps,
            model_path=model_path,
            log_dir=log_dir,
            seed=args.seed,
            live=(args.mode == "train_live"),
            eval_every=(10_000 if debug_easy else (max(25_000, args.eval_every) if fast_mode else args.eval_every)),
            num_envs=max(1, args.num_envs),
            preview_count=(1 if (fast_mode or debug_easy) else max(1, min(8, args.preview_count))),
            frame_stack=(resolved_frame_stack if load_model_path is not None else (1 if (fast_mode or debug_easy) else resolved_frame_stack)),
            enable_fixed_eval=(False if fast_mode else args.eval_seeds > 0),
            eval_seed_count=(8 if debug_easy else (0 if fast_mode else max(1, args.eval_seeds))),
            eval_horizon=(1200 if debug_easy else (600 if fast_mode else max(200, args.eval_horizon))),
            load_model_path=load_model_path,
            train_params_override=train_override,
            eval_params_override=eval_override,
        )

    if args.mode == "curve":
        plot_training_curve(log_dir / "monitor.csv")
        return

    if args.mode in ("viz", "train_viz"):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run with --mode train first.")
        visualize(model_path=model_path, horizon=args.horizon)

    if args.mode == "sanity_lqr":
        run_sanity_lqr(
            episodes=max(1, args.sanity_episodes),
            horizon=max(200, args.eval_horizon),
            do_viz=bool(args.sanity_viz),
        )
    if args.mode == "sanity_sweep":
        sweep_lqr_feasibility(
            episodes=max(4, args.sanity_episodes),
            horizon=max(200, args.eval_horizon),
        )


if __name__ == "__main__":
    main()
