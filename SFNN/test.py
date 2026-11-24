"""
Paper 1: Structurally Flexible Neural Networks (SFNN), rebuilt to follow the paper's setup
and show its results on three small, visual robot tasks.

THE PAPER'S IDEA
    Evolve the BUILDING BLOCKS of a network (neuron types + synapse learning rules), not
    its weights. The same building blocks are dropped into tasks with DIFFERENT numbers
    of inputs and outputs, wired up RANDOMLY every lifetime, and must learn during life.

THE THREE TASKS (different input/output sizes, like CartPole / Acrobot / MountainCar)
    A  Cart to flag      2 inputs (distance to flag, speed)       3 actions (left, coast, right)
    B  Turn to lamp      4 inputs (4 light sensors on the robot)  2 actions (turn left, turn right)
    C  Walk to treasure  4 inputs (treasure is right/left/up/down) 4 actions (up, down, left, right)
    Reward each step = progress toward the goal. Episode score = how close it stays (0-1).

THE NETWORK (as in the paper, simplified)
    16 neurons in total. Per task, the first ni are input neurons, the last no are output
    neurons, the rest are a hidden "reservoir". Allowed connections: input->hidden,
    hidden->hidden, hidden->output, output->hidden (the chosen action is fed back).
    3 neuron types (input, hidden, output), each with its own gain and bias.
    3 synapse types (by the sending neuron's type), each with its own learning rule:
        dw = lr * [ A*pre*post + B*pre + C*post + D + E*w
                    + (reward - expected) * (A'*pre*post + B'*pre + C'*post + D' + E'*w) ]
    (The paper's synapses are GRUs with vector activity; this 10-number rule is the
     simplification that lets evolution run live in Python.)
    Every lifetime: NEW random wiring (50% of allowed connections) and tiny random
    weights (-0.1..0.1). 2 network ticks per step. Output neuron that fires = action.

LIFETIME AND FITNESS (as in the paper)
    A lifetime = 8 episodes of 25 steps on ONE task; the network keeps learning across
    episodes. Episode k is weighted k/36, so later episodes count more.
    Each task's lifetime score is normalized (0 = random policy, 1 = hand-written policy),
    and the three are MULTIPLIED: a solution must work on all three tasks.

FOUR VERSIONS (evolved side by side with CMA-ES, population 32), as in the paper
    SFNN                3 neuron types + 3 synapse types, new random sparse wiring each life
    SFNN, one type      one neuron type and one rule for every synapse
    SFNN, fully conn.   every allowed connection present
    Fixed-wiring SFNN   evolved on ONE sparse wiring that never changes

FOUR VIEWS (radio buttons at the bottom)
    Evolution (Fig. 4)    combined and per-task scores over generations
    Watch a lifetime      the current SFNN solution living one lifetime, animated:
                          the robot, the live weight matrix, and the score of each episode
    Robustness (Fig. 6)   per-episode scores with random wiring, fixed wiring, and shuffled
                          inputs/outputs, plus the same network with learning switched off
    Same start (Fig. 5)   ONE random starting network placed in each task: three different
                          weight matrices after 8 episodes

Controls: Evolve on/off, +25 generations (fast), Reset, View, Task, Shuffle,
          Learning on, New lifetime, Run test, New start.
Keys: space = evolve on/off, f = +25 generations, n = new lifetime.

Run:   python sfnn_app.py
       python sfnn_app.py --save shot.png --gens 250 --view test     (headless screenshot)
Needs: numpy, matplotlib (with a GUI backend such as TkAgg or QtAgg; not the inline
       notebook backend, which cannot run the animation timer)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons

# ===========================================================================
# Settings
# ===========================================================================
N = 16                         # neurons per network (inputs + hidden + outputs)
T_EP, N_EP = 25, 8             # steps per episode, episodes per lifetime
POP, MAX_GENS = 32, 600
TASKS = [dict(key="A", name="Cart to flag", ni=2, no=3,
              inputs=["distance", "speed"], actions=["left", "coast", "right"]),
         dict(key="B", name="Turn to lamp", ni=4, no=2,
              inputs=["front-left", "front-right", "back-left", "back-right"], actions=["turn left", "turn right"]),
         dict(key="C", name="Walk to treasure", ni=4, no=4,
              inputs=["is right", "is left", "is up", "is down"], actions=["up", "down", "left", "right"])]
NI_MAX, NO_MAX = 4, 4
SENS = np.radians([45, -45, 135, -135])
NR = 10                        # numbers per synapse rule

METHODS = ["SFNN", "SFNN, one type", "SFNN, fully connected", "Fixed-wiring SFNN"]
KEYS = ["sfnn", "single", "fully", "fixed"]
CFG = {"sfnn": dict(single=False, wiring="random"), "single": dict(single=True, wiring="random"),
       "fully": dict(single=False, wiring="full"), "fixed": dict(single=False, wiring="fixed")}
FIXED = {t: (np.random.default_rng(100 + t).random((N, N)) < 0.5).astype(float) for t in range(3)}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"sfnn": "#1baf7a", "single": "#6250d6", "fully": "#eb6834", "fixed": "#888780"}
C_POS, C_NEG = "#1baf7a", "#eb6834"


# ===========================================================================
# CMA-ES (standard, maximizes)
# ===========================================================================
class CMAES:
    def __init__(self, x0, sigma0, popsize, seed=0):
        self.n = n = len(x0); self.rng = np.random.default_rng(seed)
        self.lam, self.mu = popsize, popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum()
        me = self.me = 1 / np.sum(self.w ** 2)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(x0, float); self.sigma = sigma0; self.C = np.eye(n)
        self.pc = np.zeros(n); self.ps = np.zeros(n); self.g = 0; self._eig()

    def _eig(self):
        ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-20))

    def ask(self):
        self.Y = (self.rng.standard_normal((self.lam, self.n)) * self.D) @ self.B.T
        return self.m + self.sigma * self.Y

    def tell(self, fit):
        o = np.argsort(-np.asarray(fit)); Ys = self.Y[o[:self.mu]]; yw = self.w @ Ys
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * (self.B @ ((self.B.T @ yw) / self.D))
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Ys * self.w[:, None]).T @ Ys)
        self.C = (self.C + self.C.T) / 2
        self.sigma = min(self.sigma * np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1)), 10)
        self.g += 1; self._eig()


# ===========================================================================
# Tasks (vectorized: every row can be a different task)
#   state s: A -> (x, -, v)   B -> (bearing, -, -)   C -> (x, y, -)
# ===========================================================================
def env_reset(task, rng):
    s = np.zeros((len(task), 3))
    A, B, C = task == 0, task == 1, task == 2
    s[A, 0] = rng.choice([-1, 1], A.sum()) * rng.uniform(0.5, 1.0, A.sum())
    s[B, 0] = rng.uniform(-np.pi, np.pi, B.sum())
    ang = rng.uniform(0, 2 * np.pi, C.sum()); s[C, 0], s[C, 1] = np.cos(ang), np.sin(ang)
    return s


def env_obs(task, s):
    o = np.zeros((len(task), NI_MAX))
    A, B, C = task == 0, task == 1, task == 2
    o[A, 0], o[A, 1] = s[A, 0], 4 * s[A, 2]
    o[B] = np.maximum(0, np.cos(s[B, 0:1] - SENS[None]))
    px, py = s[C, 0], s[C, 1]
    o[C] = np.stack([np.maximum(0, -px), np.maximum(0, px), np.maximum(0, -py), np.maximum(0, py)], 1)
    return o


def env_dist(task, s):
    d = np.zeros(len(task))
    d[task == 0] = np.abs(s[task == 0, 0])
    d[task == 1] = np.abs(s[task == 1, 0]) / np.pi
    d[task == 2] = np.hypot(s[task == 2, 0], s[task == 2, 1])
    return d


def env_close(task, s):
    c = np.zeros(len(task))
    c[task == 0] = np.maximum(0, 1 - np.abs(s[task == 0, 0]))
    c[task == 1] = (1 + np.cos(s[task == 1, 0])) / 2
    c[task == 2] = np.maximum(0, 1 - np.hypot(s[task == 2, 0], s[task == 2, 1]))
    return c


def env_step(task, s, a):
    s = s.copy(); A, B, C = task == 0, task == 1, task == 2
    v = 0.85 * s[A, 2] + 0.05 * (a[A] - 1); x = s[A, 0] + v
    hit = np.abs(x) > 1.5; s[A, 0] = np.clip(x, -1.5, 1.5); v[hit] = 0; s[A, 2] = v
    s[B, 0] = (s[B, 0] + np.where(a[B] == 0, -0.3, 0.3) + np.pi) % (2 * np.pi) - np.pi
    d = np.array([[0, .12], [0, -.12], [-.12, 0], [.12, 0]])[np.clip(a[C], 0, 3)]
    s[C, 0:2] = np.clip(s[C, 0:2] + d, -1.5, 1.5)
    return s


def hand_policy(task, s):
    a = np.zeros(len(task), int)
    A, B, C = task == 0, task == 1, task == 2
    a[A] = np.where(s[A, 0] + 3 * s[A, 2] > 0, 0, 2)
    a[B] = np.where(s[B, 0] > 0, 0, 1)
    x, y = s[C, 0], s[C, 1]
    a[C] = np.where(np.abs(x) > np.abs(y), np.where(x > 0, 2, 3), np.where(y > 0, 1, 0))
    return a


def baselines(rng, n=400):
    """Average episode score of a random policy (LO) and a hand-written policy (HI), per task."""
    lo, hi = [], []
    for t in range(3):
        task = np.full(n, t); sr = env_reset(task, rng); sh = sr.copy(); cr = ch = 0
        for _ in range(T_EP):
            sr = env_step(task, sr, rng.integers(0, TASKS[t]["no"], n)); cr = cr + env_close(task, sr)
            sh = env_step(task, sh, hand_policy(task, sh)); ch = ch + env_close(task, sh)
        lo.append(cr.mean() / T_EP); hi.append(ch.mean() / T_EP)
    return np.array(lo), np.array(hi)


LO, HI = baselines(np.random.default_rng(0))
EP_W = np.arange(1, N_EP + 1) / np.arange(1, N_EP + 1).sum()


def normalize(score, task):
    return np.clip((score - LO[task]) / (HI[task] - LO[task]), 0, 1)


# ===========================================================================
# Network structure and parameters
# ===========================================================================
def neuron_types(task):
    """0 = input, 1 = hidden, 2 = output. Inputs first, outputs last."""
    ni = np.array([TASKS[t]["ni"] for t in task]); no = np.array([TASKS[t]["no"] for t in task])
    idx = np.arange(N)[None]; ty = np.ones((len(task), N), int)
    ty[idx < ni[:, None]] = 0; ty[idx >= (N - no)[:, None]] = 2
    return ty, ni, no


def allowed(ty):
    pre, post = ty[:, :, None], ty[:, None, :]
    ok = ((pre == 0) & (post == 1)) | ((pre == 1) & (post == 1)) | ((pre == 1) & (post == 2)) | ((pre == 2) & (post == 1))
    return (ok & ~np.eye(N, dtype=bool)[None]).astype(float)


def dim(single):
    return (1 if single else 3) * (2 + NR + 1) + 1          # per type: gain, bias, rule, lr ; + action sharpness


def unpack(th, single):
    th = np.atleast_2d(th); P = len(th); nt = 1 if single else 3
    per = th[:, :nt * (2 + NR + 1)].reshape(P, nt, 2 + NR + 1)
    if single:
        per = np.repeat(per, 3, axis=1)
    return dict(g=per[:, :, 0], b=per[:, :, 1], R=per[:, :, 2:2 + NR],
                lr=0.05 * np.exp(np.clip(per[:, :, -1], -3, 2)), beta=3 * np.exp(np.clip(th[:, -1], -3, 2)))


# ===========================================================================
# Lifetime: many rows (lifetimes) at once
# ===========================================================================
class Lifetime:
    def __init__(self, p, task, rng, wiring="random", masks=None, shuffle=False, W0=None, learning=True):
        self.p, self.task, self.rng = p, np.asarray(task), rng
        B = self.B = len(self.task)
        self.ty, self.ni, self.no = neuron_types(self.task)
        al = allowed(self.ty)
        if wiring == "random":
            self.M = al * (rng.random((B, N, N)) < 0.5)
        elif wiring == "full":
            self.M = al
        else:
            self.M = np.stack([(masks or FIXED)[t] for t in self.task]) * al
        self.W = (rng.uniform(-0.1, 0.1, (B, N, N)) if W0 is None else np.array(W0, float)) * self.M
        self.perm_in = np.tile(np.arange(NI_MAX), (B, 1)); self.perm_out = np.tile(np.arange(NO_MAX), (B, 1))
        if shuffle:
            for b in range(B):
                self.perm_in[b, :self.ni[b]] = rng.permutation(self.ni[b])
                self.perm_out[b, :self.no[b]] = rng.permutation(self.no[b])
        self.g = np.take_along_axis(p["g"], self.ty, 1); self.bias = np.take_along_axis(p["b"], self.ty, 1)
        self.Rpre = np.take_along_axis(p["R"], self.ty[:, :, None], 1)       # rule chosen by the SENDING neuron's type
        self.lrpre = np.take_along_axis(p["lr"], self.ty, 1) * (1.0 if learning else 0.0)
        self.is_in, self.is_out = self.ty == 0, self.ty == 2
        self.act = np.zeros((B, N)); self.rbar = np.zeros(B)
        self.ep_scores = np.zeros((B, N_EP)); self.ep = 0; self.t = 0
        self.s = env_reset(self.task, rng); self.last = None

    def step(self):
        B = self.B
        obs = env_obs(self.task, self.s)
        o = np.take_along_axis(obs, self.perm_in, 1)                         # (possibly shuffled) wires to input neurons
        xin = np.zeros((B, N)); xin[:, :NI_MAX] = o; xin *= self.is_in
        d0 = env_dist(self.task, self.s)
        for _ in range(2):                                                    # two ticks per step, as in the paper
            z = np.einsum('bi,bij->bj', self.act, self.W)
            self.act = np.where(self.is_in, np.tanh(self.g * xin + self.bias), np.tanh(self.g * z + self.bias))
        out = np.full((B, NO_MAX), -1e9)
        for k in range(NO_MAX):
            ok = k < self.no
            out[ok, k] = self.act[np.where(ok)[0], (N - self.no + k)[ok]]
        logits = self.p["beta"][:, None] * out
        pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        a_net = (pr.cumsum(1) > self.rng.random((B, 1))).argmax(1)            # which output neuron fires
        a_env = self.perm_out[np.arange(B), a_net]                            # which actuator it is wired to
        self.s = env_step(self.task, self.s, a_env)
        r = np.clip(4 * (d0 - env_dist(self.task, self.s)), -1, 1)            # reward = progress toward the goal
        onehot = np.zeros((B, N)); onehot[np.arange(B), N - self.no + a_net] = 1
        self.act = np.where(self.is_out, onehot, self.act)                    # fired output is fed back to hidden
        pre, post, w = self.act[:, :, None], self.act[:, None, :], self.W
        c = [self.Rpre[:, :, None, k] for k in range(NR)]
        mod = (r - self.rbar)[:, None, None]
        dw = (c[0] * pre * post + c[1] * pre + c[2] * post + c[3] + c[8] * w
              + mod * (c[4] * pre * post + c[5] * pre + c[6] * post + c[7] + c[9] * w))
        self.W = np.clip(self.W + self.lrpre[:, :, None] * dw * self.M, -2, 2)
        self.rbar += 0.1 * (r - self.rbar)
        self.ep_scores[:, self.ep] += env_close(self.task, self.s) / T_EP
        self.last = dict(a_net=a_net, a_env=a_env, r=r, obs=obs)
        self.t += 1
        if self.t % T_EP == 0:
            self.ep += 1
            if self.ep < N_EP:
                self.s = env_reset(self.task, self.rng)

    def run(self):
        while self.ep < N_EP:
            self.step()
        return self.ep_scores


def evaluate(X, key, rng, shuffle=False, wiring=None, learning=True, lifetimes=1):
    """Returns fitness, normalized score per task [P,3], episode scores [P,3,lifetimes,8]."""
    cfg = CFG[key]; P = len(np.atleast_2d(X)); p = unpack(X, cfg["single"])
    rows = np.repeat(np.arange(P), 3 * lifetimes); task = np.tile(np.repeat(np.arange(3), lifetimes), P)
    L = Lifetime({k: v[rows] for k, v in p.items()}, task, rng, wiring or cfg["wiring"], FIXED, shuffle,
                 learning=learning)
    ep = L.run()
    norm = normalize(ep @ EP_W, task).reshape(P, 3, lifetimes).mean(2)
    fit = np.prod(0.05 + norm, 1) / 1.05 ** 3           # product, slightly softened for a small population
    return fit, norm, ep.reshape(P, 3, lifetimes, N_EP)


# ===========================================================================
# App
# ===========================================================================
VIEWS = ["Evolution (Fig. 4)", "Watch a lifetime", "Robustness (Fig. 6)", "Same start (Fig. 5)"]


class App:
    def __init__(self):
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
            "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.fig = plt.figure(figsize=(15, 10))
        try:
            self.fig.canvas.manager.set_window_title("Paper 1: SFNN")
        except AttributeError:
            pass
        self.title = self.fig.text(0.04, 0.965, "", fontsize=13, va="center")
        self.subtitle = self.fig.text(0.04, 0.935, "", fontsize=9.5, color=INK2, va="center")
        self.view, self.task, self.shuffle, self.learning = 0, 0, False, True
        self._build_axes()
        self._controls()
        self.timer = self.fig.canvas.new_timer(interval=90)
        self.timer.add_callback(self._tick)
        self.evolving = False
        self.test = None
        self.fig5 = None
        self.reset()

    # ---------------- layout: one set of axes per view ----------------
    def _build_axes(self):
        f = self.fig; box = dict(left=0.05, right=0.985, top=0.88, bottom=0.26)
        g = f.add_gridspec(2, 2, hspace=0.45, wspace=0.18, **box)
        self.ax_evo = [f.add_subplot(g[i, j]) for i in range(2) for j in range(2)]
        g = f.add_gridspec(2, 2, hspace=0.42, wspace=0.15, width_ratios=[1.25, 1], **box)
        self.ax_world = f.add_subplot(g[:, 0]); self.ax_w = f.add_subplot(g[0, 1]); self.ax_ep = f.add_subplot(g[1, 1])
        g = f.add_gridspec(1, 3, wspace=0.18, **{**box, "bottom": 0.36})
        self.ax_test = [f.add_subplot(g[0, j]) for j in range(3)]
        g = f.add_gridspec(2, 3, hspace=0.35, wspace=0.12, **box)
        self.ax_f5 = [[f.add_subplot(g[i, j]) for j in range(3)] for i in range(2)]
        self.views = [self.ax_evo, [self.ax_world, self.ax_w, self.ax_ep], self.ax_test,
                      [a for row in self.ax_f5 for a in row]]

    def _show_view(self):
        for i, axs in enumerate(self.views):
            for a in axs:
                a.set_visible(i == self.view)
        self.test_note.set_visible(self.view == 2)
        if getattr(self, "test_legend", None) is not None:
            self.test_legend.set_visible(self.view == 2 and self.test is not None)

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig
        self.b_evo = Button(f.add_axes([0.04, 0.115, 0.09, 0.045]), "Evolve: off")
        self.b_ff = Button(f.add_axes([0.14, 0.115, 0.09, 0.045]), "+25 generations")
        self.b_rst = Button(f.add_axes([0.04, 0.06, 0.09, 0.045]), "Reset")
        self.b_new = Button(f.add_axes([0.14, 0.06, 0.09, 0.045]), "New lifetime")
        self.b_test = Button(f.add_axes([0.04, 0.005, 0.09, 0.045]), "Run test")
        self.b_f5 = Button(f.add_axes([0.14, 0.005, 0.09, 0.045]), "New start")
        self.b_evo.on_clicked(lambda e: self.toggle_evolve())
        self.b_ff.on_clicked(lambda e: self.fast_forward())
        self.b_rst.on_clicked(lambda e: self.reset())
        self.b_new.on_clicked(lambda e: self.new_lifetime())
        self.b_test.on_clicked(lambda e: self.run_test())
        self.b_f5.on_clicked(lambda e: self.run_fig5(new=True))
        ax = f.add_axes([0.27, 0.01, 0.17, 0.15]); ax.set_title("view", loc="left", fontsize=9.5, color=INK2)
        self.r_view = RadioButtons(ax, VIEWS, active=0, activecolor=COL["sfnn"]); self.r_view.on_clicked(self._set_view)
        ax2 = f.add_axes([0.47, 0.03, 0.15, 0.12]); ax2.set_title("task (watch view)", loc="left", fontsize=9.5, color=INK2)
        self.r_task = RadioButtons(ax2, [f"{t['key']}  {t['name']}" for t in TASKS], active=0,
                                   activecolor=COL["sfnn"]); self.r_task.on_clicked(self._set_task)
        ax3 = f.add_axes([0.64, 0.05, 0.16, 0.09])
        self.c_opt = CheckButtons(ax3, ["shuffle inputs & outputs", "learning on"], [False, True])
        self.c_opt.on_clicked(self._set_opt)
        for a in (ax, ax2, ax3):
            for s in a.spines.values():
                s.set_visible(False)
        f.text(0.82, 0.135, "Keys: space = evolve on/off,\nf = +25 generations,\nn = new lifetime",
               fontsize=9, color=INK2, va="top", linespacing=1.5)
        self.test_note = f.text(0.05, 0.285, "", fontsize=9.5, color=INK, va="top", linespacing=1.5)
        f.canvas.mpl_connect("key_press_event", self._key)

    def _key(self, e):
        {" ": self.toggle_evolve, "f": self.fast_forward, "n": self.new_lifetime}.get(e.key, lambda: None)()

    def _set_view(self, label):
        self.view = VIEWS.index(label)
        if self.view == 2 and self.test is None:
            self.run_test(); return
        if self.view == 3 and self.fig5 is None:
            self.run_fig5(new=True); return
        self.draw()

    def _set_task(self, label):
        self.task = "ABC".index(label[0]); self.new_lifetime()

    def _set_opt(self, label):
        if label.startswith("shuffle"):
            self.shuffle = not self.shuffle
        else:
            self.learning = not self.learning
        self.new_lifetime()

    def toggle_evolve(self):
        self.evolving = not self.evolving
        self.b_evo.label.set_text("Evolve: on" if self.evolving else "Evolve: off")
        self.fig.canvas.draw_idle()

    def start(self):
        self.timer.start()

    # ---------------- evolution ----------------
    def reset(self):
        if self.evolving:
            self.toggle_evolve()
        self.es = {k: CMAES(np.zeros(dim(CFG[k]["single"])), 0.5, POP, seed=i) for i, k in enumerate(KEYS)}
        self.hist = {k: dict(prod=[], tasks=[]) for k in KEYS}
        self.rng = np.random.default_rng(0)
        self.gen = 0; self.test = None; self.fig5 = None
        self.new_lifetime()

    def evolve_one(self):
        if self.gen >= MAX_GENS:
            return
        for k in KEYS:
            X = self.es[k].ask()
            f, norm, _ = evaluate(X, k, self.rng)
            self.es[k].tell(f)
            self.hist[k]["prod"].append(np.prod(norm, 1).mean()); self.hist[k]["tasks"].append(norm.mean(0))
        self.gen += 1

    def fast_forward(self, n=25):
        for _ in range(n):
            self.evolve_one()
        self.test = None; self.fig5 = None
        self.new_lifetime()
        if self.view == 2:
            self.run_test()
        elif self.view == 3:
            self.run_fig5(new=False)

    # ---------------- watch a lifetime ----------------
    def new_lifetime(self):
        seed = int(np.random.default_rng().integers(1 << 30))
        p = unpack(self.es["sfnn"].m, False)
        self.life = Lifetime(p, [self.task], np.random.default_rng(seed), "random", shuffle=self.shuffle,
                             learning=self.learning)
        self.trail = [self.life.s[0].copy()]
        self.draw()

    def life_step(self):
        if self.life.ep >= N_EP:
            self.new_lifetime(); return
        ep_before = self.life.ep
        self.life.step()
        self.trail = [self.life.s[0].copy()] if self.life.ep != ep_before else self.trail + [self.life.s[0].copy()]

    # ---------------- Fig. 6: robustness ----------------
    def run_test(self, n=60):
        r = np.random.default_rng(7); res = {}
        for k in ["sfnn", "fixed"]:
            th = np.repeat(self.es[k].m[None], n, 0)
            for cond, kw in [("random wiring", dict(wiring="random")), ("fixed wiring", dict(wiring="fixed")),
                             ("random + shuffled", dict(wiring="random", shuffle=True)),
                             ("learning off", dict(wiring="random", learning=False))]:
                _, norm, ep = evaluate(th, k, r, **kw)
                res[k, cond] = (ep.mean((0, 2)), np.prod(norm, 1).mean())
        self.test = res
        self.draw()

    # ---------------- Fig. 5: same start, three tasks ----------------
    def run_fig5(self, new=True):
        if new or self.fig5 is None:
            seed = int(np.random.default_rng().integers(1 << 30))
            r = np.random.default_rng(seed)
            self.f5_W0 = r.uniform(-0.1, 0.1, (N, N)); self.f5_mask = (r.random((N, N)) < 0.5).astype(float)
            self.f5_seed = seed
        p = unpack(self.es["sfnn"].m, False); out = []
        for t in range(3):
            L = Lifetime({k: v for k, v in p.items()}, [t], np.random.default_rng(self.f5_seed + t), "fixed",
                         masks={t: self.f5_mask}, W0=self.f5_W0[None])
            W_start = L.W[0].copy(); L.run()
            out.append((W_start, L.W[0].copy(), L.M[0].copy(), L.ep_scores[0].copy()))
        self.fig5 = out
        self.draw()

    # ---------------- timer ----------------
    def _tick(self):
        if self.evolving:
            self.evolve_one()
            if self.gen % 25 == 0:
                self.test = None; self.fig5 = None
        if self.view == 1:
            self.life_step()
        if self.evolving or self.view == 1:
            self.draw()

    # ================= drawing =================
    @staticmethod
    def _clean(ax, grid=True):
        ax.cla()
        if grid:
            ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def draw(self):
        self._show_view()
        self.title.set_text(f"Paper 1: SFNN — one set of evolved building blocks, three tasks of different sizes"
                            f"    generation {self.gen}")
        self.subtitle.set_text("Tasks: A cart to flag (2 inputs → 3 actions)   B turn to lamp (4 → 2)   "
                               "C walk to treasure (4 → 4).   Every lifetime: new random wiring, tiny random "
                               "weights, 8 episodes of learning.")
        [self._draw_evo, self._draw_watch, self._draw_test, self._draw_fig5][self.view]()
        self.fig.canvas.draw_idle()

    def _draw_evo(self):
        titles = ["Combined score (product of A, B, C; 1 = all three like the hand-written policy)",
                  "Task A: cart to flag", "Task B: turn to lamp", "Task C: walk to treasure"]
        for i, ax in enumerate(self.ax_evo):
            self._clean(ax)
            for name, k in zip(METHODS, KEYS):
                h = np.array(self.hist[k]["prod"]) if i == 0 else np.array([v[i - 1] for v in self.hist[k]["tasks"]])
                if len(h):
                    sm = np.convolve(h, np.ones(7) / 7, mode="valid") if len(h) >= 7 else h
                    ax.plot(np.arange(len(sm)) + len(h) - len(sm), sm, color=COL[k], lw=2.2,
                            label=f"{name} ({dim(CFG[k]['single'])} numbers)")
            ax.set_xlim(0, max(50, self.gen)); ax.set_ylim(0, 1.0 if i else 0.6)
            ax.set_title(titles[i], loc="left", fontsize=10.5); ax.set_xlabel("generation")
            ax.set_ylabel("population average" if i == 0 else "normalized score")
        if self.gen:
            self.ax_evo[0].legend(loc="upper left", frameon=False, fontsize=9)
        else:
            self.ax_evo[0].text(0.5, 0.5, "press  Evolve  (or +25 generations)", transform=self.ax_evo[0].transAxes,
                                ha="center", color=INK2)

    def _draw_watch(self):
        L, t = self.life, self.task
        ax = self.ax_world; ax.cla(); ax.set_aspect("equal"); ax.axis("off")
        s = L.s[0]
        if t == 0:
            ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.2, 1.2)
            ax.plot([-1.6, 1.6], [-0.15, -0.15], color=INK2, lw=2)
            ax.plot([0, 0], [-0.15, 0.6], color=INK, lw=1.5); ax.add_patch(plt.Polygon([[0, 0.6], [0.3, 0.5], [0, 0.4]], color="#e34948"))
            xs = [p[0] for p in self.trail]; ax.plot(xs, [-0.05] * len(xs), color=COL["sfnn"], lw=1, alpha=0.4)
            ax.add_patch(FancyBboxPatch((s[0] - 0.18, -0.12), 0.36, 0.2, boxstyle="round,pad=0.02", fc=COL["sfnn"], ec=INK))
            ax.add_patch(Circle((s[0] - 0.1, -0.13), 0.04, color=INK)); ax.add_patch(Circle((s[0] + 0.1, -0.13), 0.04, color=INK))
            ax.text(0, 0.75, "flag", ha="center", fontsize=9.5)
        elif t == 1:
            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 1.3)
            ax.add_patch(Circle((0, 0), 0.32, fc=SURFACE, ec=INK, lw=1.5))
            ax.annotate("", xy=(0, 0.55), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
            obs = env_obs(np.array([1]), L.s)[0]
            for k, ang in enumerate(SENS):
                px, py = 0.32 * np.sin(-ang), 0.32 * np.cos(ang)
                ax.add_patch(Circle((px, py), 0.07, color=plt.cm.YlOrBr(0.15 + 0.8 * obs[k]), ec=INK2, lw=0.6))
            lx, ly = -1.0 * np.sin(s[0]), 1.0 * np.cos(s[0])
            ax.add_patch(Circle((lx, ly), 0.13, color="#eda100", ec="#854f0b")); ax.text(lx, ly - 0.25, "lamp", ha="center", fontsize=9)
            ax.text(0, -0.55, "arrow = where the robot faces; dots = its 4 light sensors", ha="center", fontsize=8.5, color=INK2)
        else:
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
            ax.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, ec=GRID, lw=1))
            ax.plot(0, 0, marker="*", ms=22, color="#eda100", mec="#854f0b")
            tr = np.array(self.trail); ax.plot(tr[:, 0], tr[:, 1], color=COL["sfnn"], lw=1.2, alpha=0.6)
            ax.plot(s[0], s[1], "o", ms=14, color=COL["sfnn"], mec=INK)
            ax.text(0, -0.25, "treasure", ha="center", fontsize=9)
        info = TASKS[t]
        last = L.last
        act = info["actions"][int(last["a_env"][0])] if last else "–"
        ep_now = min(L.ep + 1, N_EP)
        flags = []
        if self.shuffle:
            flags.append("inputs/outputs shuffled")
        if not self.learning:
            flags.append("LEARNING OFF")
        ax.set_title(f"Task {info['key']}: {info['name']}   episode {ep_now}/{N_EP}, step {L.t % T_EP if L.ep < N_EP else T_EP}/{T_EP}"
                     f"   last action: {act}" + (f"   [{', '.join(flags)}]" if flags else ""), loc="left", fontsize=10.5)
        # live weights
        ax = self.ax_w; ax.cla()
        W = np.where(L.M[0] > 0, L.W[0], np.nan)
        ax.imshow(W, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
        ni, no = info["ni"], info["no"]
        for b in (ni - 0.5, N - no - 0.5):
            ax.axhline(b, color=INK2, lw=0.8); ax.axvline(b, color=INK2, lw=0.8)
        ax.set_xticks([ni / 2 - 0.5, (ni + N - no) / 2 - 0.5, N - no / 2 - 0.5]); ax.set_xticklabels(["in", "hidden", "out"])
        ax.set_yticks([ni / 2 - 0.5, (ni + N - no) / 2 - 0.5, N - no / 2 - 0.5]); ax.set_yticklabels(["in", "hidden", "out"])
        ax.set_xlabel("to neuron"); ax.set_ylabel("from neuron")
        ax.set_title(f"Live weights (blank = no connection; {int(L.M[0].sum())} connections)", loc="left", fontsize=10.5)
        # episode scores
        ax = self.ax_ep; self._clean(ax)
        sc = L.ep_scores[0].copy()
        if L.ep < N_EP and L.t % T_EP:
            sc[L.ep] = sc[L.ep] * T_EP / (L.t % T_EP)
        done = np.arange(N_EP) < L.ep + (1 if L.ep < N_EP and L.t % T_EP else 0)
        ax.bar(np.arange(1, N_EP + 1), np.where(done, sc, 0), color=COL["sfnn"], width=0.6)
        ax.axhline(LO[t], color=INK2, ls=":", lw=1); ax.axhline(HI[t], color=INK2, ls="--", lw=1)
        ax.text(N_EP + 1.85, LO[t] + 0.015, "random", fontsize=8.5, color=INK2, va="bottom", ha="right")
        ax.text(N_EP + 1.85, HI[t] + 0.015, "hand-written", fontsize=8.5, color=INK2, va="bottom", ha="right")
        ax.set_xlim(0.4, N_EP + 1.9); ax.set_ylim(0, 1.05); ax.set_xticks(range(1, N_EP + 1))
        ax.set_xlabel("episode in this lifetime"); ax.set_title("Score of each episode (how close it stayed)", loc="left", fontsize=10.5)

    def _draw_test(self):
        if self.test is None:
            for ax in self.ax_test:
                ax.cla(); ax.axis("off")
            self.ax_test[1].text(0.5, 0.5, "press  Run test", ha="center", color=INK2, transform=self.ax_test[1].transAxes)
            self.test_note.set_text("")
            return
        styles = {"random wiring": "-", "fixed wiring": "--", "random + shuffled": ":", "learning off": "-"}
        for t, ax in enumerate(self.ax_test):
            self._clean(ax); ax.axis("on")
            for k in ["sfnn", "fixed"]:
                for cond, ls in styles.items():
                    ep, _ = self.test[k, cond]
                    col = "#e34948" if cond == "learning off" else COL[k]
                    lw = 1.2 if cond == "learning off" else 2.2
                    ax.plot(np.arange(1, N_EP + 1), ep[t], color=col, ls=ls, lw=lw, marker="o", ms=3,
                            alpha=0.5 if (cond == "learning off" and k == "fixed") else 1,
                            label=f"{'SFNN' if k == 'sfnn' else 'Fixed-wiring SFNN'}: {cond}")
            ax.axhline(LO[t], color=INK2, ls=":", lw=1); ax.axhline(HI[t], color=INK2, ls="--", lw=1)
            ax.set_ylim(0, 1.02); ax.set_xticks(range(1, N_EP + 1)); ax.set_xlabel("episode")
            ax.set_title(f"Task {TASKS[t]['key']}: {TASKS[t]['name']}", loc="left")
            if t == 0:
                ax.set_ylabel("episode score (60 lifetimes)")
                ax.text(N_EP, LO[t] + 0.02, "random", ha="right", fontsize=8.5, color=INK2)
                ax.text(N_EP, HI[t] + 0.02, "hand-written", ha="right", fontsize=8.5, color=INK2)
        if getattr(self, "test_legend", None) is not None:
            self.test_legend.remove()
        h, l = self.ax_test[2].get_legend_handles_labels()
        self.test_legend = self.fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.55, 0.295), ncol=2,
                                           frameon=False, fontsize=8.5)
        s = {c: self.test["sfnn", c][1] for c in styles}; fx = {c: self.test["fixed", c][1] for c in styles}
        self.test_note.set_text(
            f"Combined score (product of the 3 normalized task scores), generation {self.gen}:\n"
            f"   SFNN:               random wiring {s['random wiring']:.2f}   fixed wiring {s['fixed wiring']:.2f}   "
            f"shuffled {s['random + shuffled']:.2f}   learning off {s['learning off']:.2f}\n"
            f"   Fixed-wiring SFNN: random wiring {fx['random wiring']:.2f}   fixed wiring {fx['fixed wiring']:.2f}   "
            f"shuffled {fx['random + shuffled']:.2f}   learning off {fx['learning off']:.2f}")

    def _draw_fig5(self):
        if self.fig5 is None:
            return
        for t in range(3):
            W_start, W_end, M, eps = self.fig5[t]
            ni, no = TASKS[t]["ni"], TASKS[t]["no"]
            for row, W in enumerate((W_start, W_end)):
                ax = self.ax_f5[row][t]; ax.cla()
                ax.imshow(np.where(M > 0, W, np.nan), cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
                for b in (ni - 0.5, N - no - 0.5):
                    ax.axhline(b, color=INK2, lw=0.8); ax.axvline(b, color=INK2, lw=0.8)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"Task {TASKS[t]['key']}: {TASKS[t]['name']}\nstart ({ni} in, {N - ni - no} hidden, {no} out)",
                                 loc="left", fontsize=10)
                else:
                    ax.set_title(f"after 8 episodes (last episode score {eps[-1]:.2f})", loc="left", fontsize=10)
        self.ax_f5[0][0].set_ylabel("same random start", fontsize=10)
        self.ax_f5[1][0].set_ylabel("after learning in each task", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description="SFNN simulation following the paper's setup")
    ap.add_argument("--save", default=None, help="evolve headlessly and save a screenshot")
    ap.add_argument("--gens", type=int, default=250)
    ap.add_argument("--view", default="evolution", choices=["evolution", "watch", "test", "start", "all"])
    ap.add_argument("--task", type=int, default=0)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App()
    if a.save:
        for _ in range(a.gens):
            app.evolve_one()
        names = ["evolution", "watch", "test", "start"]
        todo = names if a.view == "all" else [a.view]
        base = a.save[:-4] if a.save.endswith(".png") else a.save
        app.r_task.set_active(a.task)
        for v in todo:
            app.r_view.set_active(names.index(v))
            if v == "watch":
                for _ in range(3 * T_EP + 12):
                    app.life_step()
                app.draw()
            out = f"{base}_{v}.png" if a.view == "all" else a.save
            app.fig.savefig(out, dpi=110)
            print("Saved", out)
        print({k: round(float(np.mean(app.hist[k]["prod"][-10:])), 3) for k in KEYS if app.hist[k]["prod"]})
    else:
        app.start()
        plt.show()


if __name__ == "__main__":
    main()