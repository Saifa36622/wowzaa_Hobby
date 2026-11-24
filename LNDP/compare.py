"""
Normal NDP vs LNDP: during training (evolution) and during one life (lifelong learning).

NORMAL NDP (Najarro et al. 2023 style, simplified)
    Before life the network GROWS from one hidden seed cell: every growth step each cell
    updates its state from its neighbours, and a hidden cell may divide (the child copies
    the parent's connections). After growth, each connection gets a weight from the states
    of its two cells. Then the network is FROZEN: no weight or wiring changes during life,
    and the reward is never used. Evolved: the growth rules (45 numbers).

LNDP (Plantec et al. 2024 style, simplified; same model as lndp_app.py)
    Starts from a random network, runs spontaneous activity before life, and then KEEPS
    running its rules during life: node states, connection weights (using the reward) and
    growing / pruning connections. Evolved: those rules (53 numbers).

THE TASK (non-stationary foraging, shortened from the LNDP paper)
    Corridor [left end][center][right end]. The agent starts in the center and only sees
    which cell it is in. Food is at one end; reaching an end finishes the episode (food =
    reward 1). After every episode the food switches sides with probability p_switch.
    A frozen network cannot know the food moved (it never uses the reward), so its best
    is "always go to one side" (about half the food). Doing better needs learning during life.

WHAT YOU SEE
    Training            food found per life over generations, for both
    Same world          both current solutions living in the same corridor, side by side
    After a switch      chance of finding food vs episodes since the food last switched sides
                        (64 test lives each): this is where lifelong learning shows
    NDP network         grows before life (animated), then frozen
    LNDP network        develops before life, then weights change and connections are
                        grown (green glow) / pruned (red dashed) during life
    This life           food found so far; how much the weights changed since birth;
                        number of connections

Controls: Evolve on/off, +25 generations, Reset, Play/Pause life, New life, Test after-switch,
LNDP starting network, p_switch, speed.   Keys: space = evolve, p = play/pause, n = new life.
Run:   python ndp_vs_lndp.py
       python ndp_vs_lndp.py --save shot.png --gens 200      (headless)
Needs: numpy, matplotlib
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
NI, NO = 3, 3
CENTER = NI // 2
DH, DEDGE, TICKS, T_SA, LIFE, GROW = 2, 2, 2, 20, 120, 5
POP, MAX_GENS, K_MAX = 32, 1000, 8
IN_LABELS = ["at left end", "at center", "at right end"]
OUT_LABELS = ["go left", "stay", "go right"]
METHODS = {"ndp": "Normal NDP (grow, then frozen)", "lndp": "LNDP (keeps learning and rewiring)"}
KEYS = ["ndp", "lndp"]

SURFACE, INK, INK2, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c7c0", "#e4e3df"
COL = {"ndp": "#e87a2c", "lndp": "#1baf7a"}
C_POS, C_NEG, C_ADD, C_DEL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
BAND_L, BAND_R = "#eeedfe", "#fdf1dc"


def allow_matrix(nh):
    n = NI + nh + NO
    A = np.zeros((n, n))
    A[:NI, NI:NI + nh] = 1; A[NI:NI + nh, NI:NI + nh] = 1; A[NI:NI + nh, NI + nh:] = 1
    np.fill_diagonal(A, 0)
    return A


def type_matrix(nh):
    n = NI + nh + NO
    T = np.zeros((n, 3)); T[:NI, 0] = 1; T[NI:NI + nh, 1] = 1; T[NI + nh:, 2] = 1
    return T


# LNDP network: 3 in, 3 hidden, 3 out
NH_L = 3; N_L = NI + NH_L + NO; ALLOW_L = allow_matrix(NH_L); TYPE_L = type_matrix(NH_L)
NX = DH + 1 + 1 + 3 + DH
NZ = DEDGE + 2 + 1 + 2 + DH + DH
# NDP network: 3 in, up to 6 hidden (grown), 3 out
NH_D = 6; N_D = NI + NH_D + NO; ALLOW_D = allow_matrix(NH_D); TYPE_D = type_matrix(NH_D)
HID_D = np.zeros(N_D, bool); HID_D[NI:NI + NH_D] = True
EMB = np.random.default_rng(0).uniform(-1, 1, (N_D, DH))          # cell states before growth
SPLIT = np.random.default_rng(1).normal(0, 1, (N_D, DH))           # how a child differs from its parent


# ---------------------------------------------------------------------------
# CMA-ES (standard, maximizes)
# ---------------------------------------------------------------------------
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


def _take(th, spec):
    th = np.atleast_2d(th); P = len(th); i = 0; out = {}
    for name, shape in spec:
        n = int(np.prod(shape)) if shape else 1
        out[name] = th[:, i:i + n].reshape((P,) + shape); i += n
    return out


def act_from(v, out_slice, rng):
    out = v[:, out_slice]
    pr = np.exp(3 * (out - out.max(1, keepdims=True))); pr /= pr.sum(1, keepdims=True)
    return (pr.cumsum(1) > rng.random((len(v), 1))).argmax(1)


# ---------------------------------------------------------------------------
# Normal NDP: grow before life, then frozen
# ---------------------------------------------------------------------------
NDP_SPEC = [("G", (DH, 2 * DH + 3)), ("gb", (DH,)), ("dw", (DH,)), ("db", ()),
            ("M", (DH + 3, DH + 3)), ("c", ())]
NDP_DIM = sum(int(np.prod(s)) if s else 1 for _, s in NDP_SPEC)


class NDPAgent:
    kind = "ndp"

    def __init__(self, th, P, rng, density=None, record=False):
        p = _take(th, NDP_SPEC); self.rng = rng; self.P = P
        if p["G"].shape[0] != P:
            p = {k: np.repeat(v, P, 0) for k, v in p.items()}
        alive = np.zeros((P, N_D), bool); alive[:, :NI] = True; alive[:, NI] = True; alive[:, NI + NH_D:] = True
        A = np.zeros((P, N_D, N_D)); A[:, :NI, NI] = 1; A[:, NI, NI + NH_D:] = 1
        h = np.repeat(EMB[None], P, 0).copy()
        self.frames = []
        if record:
            self.frames.append(self._frame(p, A, alive, h))
        for _ in range(GROW):
            sym = ((A + A.transpose(0, 2, 1)) > 0).astype(float)
            m = np.einsum('pij,pjd->pid', sym, h) / np.maximum(sym.sum(2, keepdims=True), 1)
            x = np.concatenate([h, m, np.repeat(TYPE_D[None], P, 0)], -1)
            h = np.where(alive[..., None], np.tanh(np.einsum('pdx,pnx->pnd', p["G"], x) + p["gb"][:, None]), h)
            pdiv = 1 / (1 + np.exp(-(np.einsum('pd,pnd->pn', p["dw"], h) + p["db"][:, None])))
            want = alive & HID_D & (pdiv > 0.5); free = ~alive & HID_D
            can = want.any(1) & free.any(1)
            parent = np.argmax(np.where(want, pdiv, -1), 1); child = np.argmax(free, 1)
            for b in np.nonzero(can)[0]:                 # one division per network per growth step
                pa, ch = parent[b], child[b]
                A[b, :, ch] = A[b, :, pa]; A[b, ch, :] = A[b, pa, :]; A[b, pa, ch] = 1
                h[b, ch] = h[b, pa] + 0.3 * SPLIT[ch]; alive[b, ch] = True
            A = A * ALLOW_D
            if record:
                self.frames.append(self._frame(p, A, alive, h))
        self.A, self.alive = A, alive
        self.Wf = self._weights(p, A, h)
        self.v = np.zeros((P, N_D)); self.added = np.zeros((P, N_D, N_D)); self.removed = np.zeros((P, N_D, N_D))
        self.last_a = None

    @staticmethod
    def _weights(p, A, h):
        hx = np.concatenate([h, np.repeat(TYPE_D[None], len(h), 0)], -1)
        return 2 * np.tanh(np.einsum('pix,pxy,pjy->pij', hx, p["M"], hx) + p["c"][:, None, None]) * A

    def _frame(self, p, A, alive, h):
        return dict(A=A[0].copy(), W=self._weights(p, A, h)[0], alive=alive[0].copy(), v=np.zeros(N_D))

    def develop(self, record=False):
        pass                                           # growth already happened in __init__

    def W(self):
        return self.Wf

    def act(self, obs):
        for _ in range(TICKS):
            self.v[:, :NI] = obs
            self.v = np.tanh(np.einsum('pi,pij->pj', self.v, self.Wf))
        self.v[:, :NI] = obs
        a = act_from(self.v, slice(NI + NH_D, N_D), self.rng)
        self.v[:, NI + NH_D:] = np.eye(NO)[a]; self.last_a = a
        return a

    def learn(self, r):
        pass                                           # frozen: the reward is never used


# ---------------------------------------------------------------------------
# LNDP: develop, then keep learning and rewiring during life (as in lndp_app.py)
# ---------------------------------------------------------------------------
LNDP_SPEC = [("A", (DH, NX)), ("b", (DH,)), ("U", (DEDGE, NZ)), ("ub", (DEDGE,)), ("lr", ()),
             ("sg", (2 * DH,)), ("sgb", ()), ("pr", (DEDGE,)), ("prb", ())]
LNDP_DIM = sum(int(np.prod(s)) if s else 1 for _, s in LNDP_SPEC)


class LNDPAgent:
    kind = "lndp"

    def __init__(self, th, P, rng, density=0.5, record=False):
        p = _take(th, LNDP_SPEC)
        if p["A"].shape[0] != P:
            p = {k: np.repeat(v, P, 0) for k, v in p.items()}
        self.p, self.P, self.rng = p, P, rng
        self.h = rng.uniform(-1, 1, (P, N_L, DH)); self.v = np.zeros((P, N_L))
        dens = np.clip(rng.normal(density, 0.1, (P, 1, 1)), 0, 1) if 0 < density < 1 else density
        self.A = (rng.random((P, N_L, N_L)) < dens) * ALLOW_L
        self.e = rng.uniform(-1, 1, (P, N_L, N_L, DEDGE)) * self.A[..., None]
        self.r = np.zeros(P); self.lr = 0.3 * np.exp(np.clip(p["lr"], -3, 1))
        self.added = np.zeros((P, N_L, N_L)); self.removed = np.zeros((P, N_L, N_L))
        self.last_a = None; self.frames = []

    def W(self):
        return self.e[..., 0] * self.A

    def _tick(self, obs):
        W = self.W()
        for _ in range(TICKS):
            self.v[:, :NI] = obs
            self.v = np.tanh(np.einsum('pi,pij->pj', self.v, W))
        self.v[:, :NI] = obs

    def _update(self):
        p, P = self.p, self.P
        nb = (self.A + self.A.transpose(0, 2, 1)) > 0
        mh = np.einsum('pij,pjd->pid', nb, self.h) / np.maximum(nb.sum(2, keepdims=True), 1)
        x = np.concatenate([self.h, self.v[..., None], np.repeat(self.r[:, None, None], N_L, 1),
                            np.repeat(TYPE_L[None], P, 0), mh], -1)
        self.h = self.h + 0.5 * (np.tanh(np.einsum('pdx,pnx->pnd', p["A"], x) + p["b"][:, None, :]) - self.h)
        vi = np.broadcast_to(self.v[:, :, None], (P, N_L, N_L)); vj = np.broadcast_to(self.v[:, None, :], (P, N_L, N_L))
        rr = np.broadcast_to(self.r[:, None, None], (P, N_L, N_L))
        hi = np.broadcast_to(self.h[:, :, None, :], (P, N_L, N_L, DH)); hj = np.broadcast_to(self.h[:, None, :, :], (P, N_L, N_L, DH))
        z = np.concatenate([self.e, vi[..., None], vj[..., None], rr[..., None], (vi * vj)[..., None],
                            (rr * vi * vj)[..., None], hi, hj], -1)
        de = np.tanh(np.einsum('pez,pijz->pije', p["U"], z) + p["ub"][:, None, None, :])
        self.e = np.clip(self.e + self.lr[:, None, None, None] * de, -3, 3) * self.A[..., None]
        hh = np.concatenate([hi, hj], -1)
        pg = 1 / (1 + np.exp(-(np.einsum('pk,pijk->pij', p["sg"], hh) + p["sgb"][:, None, None] - 3)))
        pp = 1 / (1 + np.exp(-(np.einsum('pk,pijk->pij', p["pr"], self.e) + p["prb"][:, None, None] - 3)))
        add = (self.rng.random((P, N_L, N_L)) < pg) * ALLOW_L * (1 - self.A)
        rem = (self.rng.random((P, N_L, N_L)) < pp) * self.A
        self.A = np.clip(self.A + add - rem, 0, 1)
        self.e = np.where(add[..., None] > 0, 0.1 * self.rng.standard_normal(self.e.shape), self.e) * self.A[..., None]
        self.added, self.removed = add, rem

    def develop(self, record=False):
        """Spontaneous activity before life: noise on the inputs, no reward, same rules."""
        s = np.zeros((self.P, NI))
        if record:
            self.frames = [dict(A=self.A[0].copy(), W=self.W()[0].copy(), v=self.v[0].copy())]
        for _ in range(T_SA):
            s = s + 0.1 * (0 - s) + 0.3 * self.rng.standard_normal((self.P, NI)); self.r = np.zeros(self.P)
            self._tick(np.maximum(s, 0)); self._update()
            if record:
                self.frames.append(dict(A=self.A[0].copy(), W=self.W()[0].copy(), v=self.v[0].copy()))
        self.added[:] = 0; self.removed[:] = 0

    def act(self, obs):
        self._tick(obs)
        a = act_from(self.v, slice(NI + NH_L, N_L), self.rng)
        self.v[:, NI + NH_L:] = np.eye(NO)[a]; self.last_a = a
        return a

    def learn(self, r):
        self.r = r; self._update()


AGENT = {"ndp": NDPAgent, "lndp": LNDPAgent}
DIMS = {"ndp": NDP_DIM, "lndp": LNDP_DIM}


# ---------------------------------------------------------------------------
# The corridor, for P agents at once
# ---------------------------------------------------------------------------
class World:
    def __init__(self, P, rng, p_switch, schedule=None):
        self.P, self.rng, self.ps, self.sched = P, rng, p_switch, schedule
        self.food = np.full(P, schedule[0]) if schedule is not None else np.where(rng.random(P) < 0.5, 0, NI - 1)
        self.pos = np.full(P, CENTER); self.k = np.zeros(P, int); self.t = 0
        self.since = np.full(P, K_MAX); self.found = np.zeros(P)
        self.hits = np.zeros(K_MAX + 1); self.count = np.zeros(K_MAX + 1)

    def obs(self):
        return np.eye(NI)[self.pos]

    def step(self, a):
        self.pos = np.clip(self.pos + a - 1, 0, NI - 1); self.k += 1
        got = self.pos == self.food
        self.found += got
        reset = got | (self.pos == 0) | (self.pos == NI - 1) | (self.k >= 10)
        idx = np.minimum(self.since[reset], K_MAX)
        np.add.at(self.count, idx, 1); np.add.at(self.hits, idx, got[reset])
        self.arrived = self.pos.copy(); self.t += 1
        if self.sched is not None:
            self.food = np.full(self.P, self.sched[min(self.t, len(self.sched) - 1)])
        else:
            sw = reset & (self.rng.random(self.P) < self.ps)
            self.food = np.where(sw, NI - 1 - self.food, self.food)
            self.since = np.where(reset, np.where(sw, 0, self.since + 1), self.since)
        self.pos = np.where(reset, CENTER, self.pos); self.k = np.where(reset, 0, self.k)
        return got.astype(float)


def run_lives(th, kind, rng, p_switch, density, P=None, lives=1):
    th = np.atleast_2d(th); P = P or len(th)
    tot = np.zeros(P); hits = np.zeros(K_MAX + 1); count = np.zeros(K_MAX + 1)
    for _ in range(lives):
        ag = AGENT[kind](th, P, rng, density); ag.develop()
        w = World(P, rng, p_switch)
        for _ in range(LIFE):
            ag.learn(w.step(ag.act(w.obs())))
        tot += w.found; hits += w.hits; count += w.count
    return tot / lives, hits, count


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def screen_inches():
    try:
        import tkinter
        root = tkinter.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight(); root.destroy()
        dpi = plt.rcParams["figure.dpi"]
        return max(11, 0.94 * w / dpi), max(7, 0.86 * h / dpi)
    except Exception:
        return 16, 9.5


class App:
    def __init__(self, size):
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                             "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
        self.fig = plt.figure(figsize=size)
        try:
            self.fig.canvas.manager.set_window_title("Normal NDP vs LNDP")
        except AttributeError:
            pass
        f = self.fig
        self.axE = f.add_axes([0.045, 0.575, 0.27, 0.29]); self.axW = f.add_axes([0.355, 0.575, 0.3, 0.29])
        self.axS = f.add_axes([0.715, 0.575, 0.27, 0.29])
        self.axN1 = f.add_axes([0.015, 0.155, 0.26, 0.34]); self.axN2 = f.add_axes([0.29, 0.155, 0.26, 0.34])
        self.axF = f.add_axes([0.615, 0.4, 0.37, 0.1]); self.axD = f.add_axes([0.615, 0.285, 0.37, 0.075])
        self.axC = f.add_axes([0.615, 0.17, 0.37, 0.075])
        self.title = f.text(0.045, 0.965, "", va="center", fontweight="bold")
        self.sub = f.text(0.045, 0.932, "", va="center", color=INK2)
        self.density = 0.5; self.playing = True; self.evolving = False
        self._controls()
        f.canvas.mpl_connect("key_press_event", self._key)
        f.canvas.mpl_connect("resize_event", lambda e: self.draw())
        self.timer = f.canvas.new_timer(interval=120); self.timer.add_callback(self._tick)
        self.reset()

    # ---------------- training ----------------
    def reset(self):
        if self.evolving:
            self.toggle_evolve()
        self.es = {k: CMAES(np.zeros(DIMS[k]), 0.5, POP, seed=i) for i, k in enumerate(KEYS)}
        self.hist = {k: [] for k in KEYS}; self.best = {k: [] for k in KEYS}
        self.rng = np.random.default_rng(0); self.gen = 0; self.sec = {k: 0.0 for k in KEYS}
        self.adapt = None
        self.test_switch(); self.new_life()

    def evolve_one(self):
        if self.gen >= MAX_GENS:
            return
        for k in KEYS:
            t0 = time.perf_counter()
            X = self.es[k].ask()
            f, _, _ = run_lives(X, k, self.rng, self.s_p.val, self.density, lives=2)
            self.es[k].tell(f); self.hist[k].append(f.mean()); self.best[k].append(f.max())
            self.sec[k] = 0.8 * self.sec[k] + 0.2 * (time.perf_counter() - t0) if self.gen else time.perf_counter() - t0
        self.gen += 1

    def plus25(self):
        for _ in range(25):
            self.evolve_one()
        self.test_switch(); self.new_life()

    def test_switch(self):
        """64 test lives of each current solution; chance of food vs episodes since the last switch."""
        r = np.random.default_rng(123); self.adapt = {}
        for k in KEYS:
            f, hits, count = run_lives(self.es[k].m, k, r, self.s_p.val, self.density, P=64)
            self.adapt[k] = (hits / np.maximum(count, 1), count, f.mean())

    # ---------------- one watched life ----------------
    def new_life(self):
        seed = int(np.random.default_rng().integers(1 << 30)); r = np.random.default_rng(seed)
        sched = [int(r.integers(2)) * (NI - 1)]
        for _ in range(LIFE):
            sched.append(NI - 1 - sched[-1] if r.random() < self.s_p.val else sched[-1])
        self.sched = np.array(sched)
        self.ag, self.world, self.rec = {}, {}, {}
        for i, k in enumerate(KEYS):
            rr = np.random.default_rng(seed + 1 + i)
            ag = AGENT[k](self.es[k].m, 1, rr, self.density, record=True); ag.develop(record=True)
            self.ag[k] = ag; self.world[k] = World(1, rr, self.s_p.val, schedule=self.sched)
            W0 = ag.W()[0].copy()
            self.rec[k] = dict(W0=W0, found=[0], dW=[0.0], conns=[int(ag.A[0].sum())])
        self.dev_i = 0; self.flash = []
        self.draw()

    def life_step(self):
        if self.dev_i <= T_SA:                       # development phase (before life)
            self.dev_i += 1; return
        if self.world["lndp"].t >= LIFE:
            self.new_life(); return
        for k in KEYS:
            ag, w = self.ag[k], self.world[k]
            ag.learn(w.step(ag.act(w.obs())))
            rc = self.rec[k]
            rc["found"].append(w.found[0]); rc["conns"].append(int(ag.A[0].sum()))
            rc["dW"].append(float(np.abs(ag.W()[0] - rc["W0"]).sum()))
        L = self.ag["lndp"]
        self.flash = [(i, j, "add") for i, j in zip(*np.where(L.added[0] > 0))] + \
                     [(i, j, "del") for i, j in zip(*np.where(L.removed[0] > 0))]

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig; c = "#ecebe7"; y1, y2, h = 0.075, 0.018, 0.045
        spec = [("Evolve: off", 0.045, y1, 0.085, self.toggle_evolve), ("+25 generations", 0.135, y1, 0.095, self.plus25),
                ("Reset", 0.235, y1, 0.055, self.reset), ("Pause life", 0.045, y2, 0.085, self.toggle_play),
                ("New life", 0.135, y2, 0.07, self.new_life), ("Test after-switch", 0.21, y2, 0.08, self._test_btn)]
        self.btns = {}
        for lab, x, y, w, fn in spec:
            b = Button(f.add_axes([x, y, w, h]), lab, color=c, hovercolor="#dcdad3")
            b.on_clicked(lambda e, fn=fn: fn()); self.btns[lab] = b
        axr = f.add_axes([0.31, 0.01, 0.15, 0.11]); axr.set_title("LNDP starting network", fontsize=9, loc="left", color=INK2)
        for s in axr.spines.values():
            s.set_visible(False)
        self.r_d = RadioButtons(axr, ["empty", "random half", "full"], active=1, activecolor=COL["lndp"])
        self.r_d.on_clicked(self._set_density)
        self.s_p = Slider(f.add_axes([0.53, 0.085, 0.15, 0.022]), "p_switch", 0.0, 0.5, valinit=0.1, valstep=0.05,
                          color=COL["lndp"])
        self.s_speed = Slider(f.add_axes([0.53, 0.035, 0.15, 0.022]), "life steps / frame", 1, 10, valinit=2,
                              valstep=1, color=INK2)
        self.legend_txt = f.text(0.735, 0.092, "Network panels:  blue = positive weight, orange = negative,\n"
                                 "width = size;  green glow = connection grown,\nred dashed = connection pruned;  "
                                 "ring = chosen action\nKeys: space = evolve, p = play/pause, n = new life",
                                 va="top", color=INK2, linespacing=1.5)

    def _key(self, e):
        {" ": self.toggle_evolve, "p": self.toggle_play, "n": self.new_life}.get(e.key, lambda: None)()

    def _set_density(self, label):
        self.density = {"empty": 0.0, "random half": 0.5, "full": 1.0}[label]; self.new_life()

    def _test_btn(self):
        self.test_switch(); self.draw()

    def toggle_evolve(self):
        self.evolving = not self.evolving
        self.btns["Evolve: off"].label.set_text("Evolve: on" if self.evolving else "Evolve: off")
        self.fig.canvas.draw_idle()

    def toggle_play(self):
        self.playing = not self.playing
        self.btns["Pause life"].label.set_text("Pause life" if self.playing else "Play life")
        self.fig.canvas.draw_idle()

    def start(self):
        self.timer.start()

    def _tick(self):
        if self.evolving:
            self.evolve_one()
            if self.gen % 25 == 0:
                self.test_switch(); self.new_life(); return
        if self.playing:
            for _ in range(int(self.s_speed.val)):
                self.life_step()
        self.draw()

    # ---------------- drawing ----------------
    def _fs(self):
        w, h = self.fig.get_size_inches()
        return float(np.clip(min(w / 1.75, h) * 1.08, 7.5, 14))

    @staticmethod
    def _clean(ax):
        ax.cla(); ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _bands(self, ax, t_now):
        start = 0
        for i in range(1, t_now + 2):
            if i == t_now + 1 or self.sched[i] != self.sched[start]:
                ax.axvspan(start, i, color=BAND_L if self.sched[start] == 0 else BAND_R, lw=0, zorder=0)
                start = i

    def draw(self):
        fs = self.fs = self._fs(); ps = self.s_p.val
        self.title.set_text(f"Normal NDP (grow once, then frozen)  vs  LNDP (keeps learning and rewiring during life)"
                            f"      generation {self.gen}")
        self.title.set_fontsize(fs * 1.2)
        self.sub.set_text("Task: corridor [left end][center][right end]. Food at one end, switches sides after an "
                          f"episode with probability p_switch = {ps:.2f}. The agent only sees which cell it is in.")
        self.sub.set_fontsize(fs * 0.9)
        for b in self.btns.values():
            b.label.set_fontsize(fs * 0.78)
        for l in self.r_d.labels:
            l.set_fontsize(fs * 0.78)
        for s in (self.s_p, self.s_speed):
            s.label.set_fontsize(fs * 0.78); s.valtext.set_fontsize(fs * 0.78)
        self.legend_txt.set_fontsize(fs * 0.75)
        self._draw_training(fs, ps); self._draw_world(fs); self._draw_switch(fs, ps)
        dev = self.dev_i <= T_SA
        self._draw_net(self.axN1, "ndp", fs, dev); self._draw_net(self.axN2, "lndp", fs, dev)
        self._draw_life(fs)
        self.fig.canvas.draw_idle()

    def _draw_training(self, fs, ps):
        ax = self.axE; self._clean(ax)
        for k in KEYS:
            h = np.array(self.hist[k])
            if len(h):
                sm = np.convolve(h, np.ones(7) / 7, mode="valid") if len(h) >= 7 else h
                ax.plot(np.arange(len(sm)) + len(h) - len(sm), sm, color=COL[k], lw=2.2,
                        label=f"{METHODS[k]}, {DIMS[k]} numbers")
        ideal, fixed, rnd = LIFE * (1 - ps), LIFE * 0.5, LIFE / 3
        xmax = max(60, self.gen)
        for yv, t in ((ideal, "ideal: go where food was last"), (fixed, "best WITHOUT learning: always one side"),
                      (rnd, "random moves")):
            ax.axhline(yv, color=INK2, lw=0.9, ls=":")
            ax.text(xmax, yv + 2, t, fontsize=fs * 0.65, color=INK2, ha="right")
        ax.set_xlim(0, xmax); ax.set_ylim(0, LIFE + 40)
        ax.set_xlabel("generation", fontsize=fs * 0.8); ax.tick_params(labelsize=fs * 0.7)
        ax.set_title(f"TRAINING: food found per {LIFE}-step life (population average)", loc="left", fontsize=fs)
        if self.gen:
            ax.legend(loc="upper left", frameon=False, fontsize=fs * 0.7)
            sp = "   ".join(f"{k.upper()}: {1000 * self.sec[k]:.0f} ms/gen" for k in KEYS)
            ax.text(0.01, 0.02, sp, transform=ax.transAxes, fontsize=fs * 0.65, color=INK2)
        else:
            ax.text(0.5, 0.55, "press  Evolve  or  +25 generations", transform=ax.transAxes, ha="center",
                    color=INK2, fontsize=fs * 0.9)

    def _draw_world(self, fs):
        ax = self.axW; ax.cla(); ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.4, 2.6); ax.axis("off")
        t = self.world["lndp"].t; food = self.sched[min(t, LIFE)]
        for row, k in enumerate(KEYS):
            y = 1.75 - row * 1.35; w = self.world[k]; ag = self.ag[k]
            for c in range(NI):
                x0 = 4.0 + c * 2.0
                ax.add_patch(plt.Rectangle((x0, y - 0.35), 1.8, 0.7, fc=SURFACE, ec=INK2, lw=0.8))
                if c == food:
                    ax.text(x0 + 0.45, y, "food", ha="center", va="center", fontsize=fs * 0.75, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#fdf1dc", ec="#c98500", lw=1))
            if ag.last_a is not None and self.dev_i > T_SA:
                arr = int(w.arrived[0])
                ax.scatter([4.0 + arr * 2.0 + 1.3], [y], s=fs * 18, color=COL[k], edgecolors=INK, zorder=4)
            ax.text(0.0, y + 0.15, METHODS[k], fontsize=fs * 0.78, va="center", color=COL[k], fontweight="bold")
            ax.text(0.0, y - 0.22, f"food so far: {int(w.found[0])}", fontsize=fs * 0.75, va="center", color=INK2)
        phase = (f"BEFORE LIFE: development {self.dev_i}/{T_SA}" if self.dev_i <= T_SA
                 else f"DURING LIFE: step {t}/{LIFE}")
        ax.set_title(f"Same world, both current solutions    {phase}", loc="left", fontsize=fs)

    def _draw_switch(self, fs, ps):
        ax = self.axS; self._clean(ax)
        x = np.arange(K_MAX + 1)
        for i, k in enumerate(KEYS):
            rate, count, fmean = self.adapt[k]
            ok = count > 5
            ax.plot(x[ok], rate[ok], color=COL[k], lw=2.4, marker="o", ms=5,
                    label=f"{METHODS[k].split(' (')[0]}: {fmean:.0f} food/life")
        ax.axhline(0.5, color=INK2, lw=0.9, ls=":")
        ax.text(K_MAX, 0.52, "coin flip", fontsize=fs * 0.65, color=INK2, ha="right")
        ax.set_ylim(0, 1.05); ax.set_xlim(-0.3, K_MAX + 0.3)
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(K_MAX)] + [f"{K_MAX}+"])
        ax.set_xlabel("episodes since the food last switched sides (0 = first episode after a switch)",
                      fontsize=fs * 0.72)
        ax.set_ylabel("chance of finding food", fontsize=fs * 0.75); ax.tick_params(labelsize=fs * 0.7)
        ax.legend(loc="lower right", frameon=False, fontsize=fs * 0.7)
        ax.set_title(f"LIFELONG LEARNING: after the food moves (64 test lives, gen {self.gen})",
                     loc="left", fontsize=fs * 0.92)

    def _pos(self, k):
        pos = {}
        nh = NH_L if k == "lndp" else NH_D
        for i in range(NI):
            pos[i] = (1.7, 8.6 - i * 3.6)
        ys = np.linspace(9.4, 1.4, nh) if nh > 3 else np.array([8.6, 5.0, 1.4])
        for j in range(nh):
            pos[NI + j] = (5.0, ys[j])
        for o in range(NO):
            pos[NI + nh + o] = (8.3, 8.6 - o * 3.6)
        return pos, nh

    def _draw_net(self, ax, k, fs, dev):
        ax.cla(); ax.set_xlim(-0.6, 10.6); ax.set_ylim(-1.2, 10.4); ax.axis("off")
        pos, nh = self._pos(k); ag = self.ag[k]
        if dev:
            if k == "ndp":
                fi = min(len(ag.frames) - 1, int(round(self.dev_i * GROW / T_SA)))
                fr = ag.frames[fi]; A, W, v, alive = fr["A"], fr["W"], fr["v"], fr["alive"]
                note = f"growing: step {fi}/{GROW}, {int(alive[NI:NI + NH_D].sum())} hidden cells"
            else:
                fr = ag.frames[min(self.dev_i, len(ag.frames) - 1)]; A, W, v = fr["A"], fr["W"], fr["v"]
                alive = np.ones(N_L, bool); note = f"spontaneous activity: step {self.dev_i}/{T_SA}"
        else:
            A, W, v = ag.A[0], ag.W()[0], ag.v[0]
            alive = ag.alive[0] if k == "ndp" else np.ones(N_L, bool)
            note = "FROZEN: nothing changes during life" if k == "ndp" else "weights and wiring change every step"
        lo, hi = NI, NI + nh

        def curve(i, j):
            (x0, y0), (x1, y1) = pos[i], pos[j]
            if lo <= i < hi and lo <= j < hi:
                tt = np.linspace(0, 1, 30); bow = 1.0 + 0.25 * abs(y1 - y0)
                return x0 - bow * np.sin(np.pi * tt) * (1 if y1 < y0 else -1) * 0.5, y0 + (y1 - y0) * tt
            return [x0, x1], [y0, y1]

        lim = max(0.5, np.abs(W).max())
        for i, j in zip(*np.where(A > 0)):
            if alive[i] and alive[j]:
                xs, ys = curve(i, j)
                ax.plot(xs, ys, color=C_POS if W[i, j] >= 0 else C_NEG, lw=0.4 + 2.2 * abs(W[i, j]) / lim,
                        alpha=0.85, solid_capstyle="round", zorder=1)
        if k == "lndp" and not dev:
            for i, j, kind in self.flash:
                xs, ys = curve(i, j)
                if kind == "add":
                    ax.plot(xs, ys, color=C_ADD, lw=5, alpha=0.35, zorder=0)
                else:
                    ax.plot(xs, ys, color=C_DEL, lw=1.6, ls="--", alpha=0.9, zorder=1)
        s = (fs * 2.3) ** 2
        for n, (x, y) in pos.items():
            if not alive[n]:
                ax.scatter([x], [y], s=s * 0.6, facecolors="none", edgecolors=FAINT, linewidths=0.8, ls=":", zorder=2)
                continue
            ax.scatter([x], [y], s=s, color=plt.cm.RdBu_r(0.5 + 0.5 * np.clip(v[n], -1, 1)),
                       edgecolors=INK2, linewidths=1, zorder=3)
        for i in range(NI):
            ax.text(pos[i][0] - 0.6, pos[i][1], IN_LABELS[i], ha="right", va="center", fontsize=fs * 0.7)
        for o in range(NO):
            ax.text(pos[NI + nh + o][0] + 0.6, pos[NI + nh + o][1], OUT_LABELS[o], ha="left", va="center",
                    fontsize=fs * 0.7)
        if ag.last_a is not None and not dev:
            ax.scatter(*pos[NI + nh + int(ag.last_a[0])], s=s * 2, facecolors="none", edgecolors=INK,
                       linewidths=2, zorder=2)
        ax.text(-0.5, -0.8, note + f"   |   connections: {int((A > 0).sum())}", fontsize=fs * 0.75,
                color=COL[k] if not dev else INK2, fontweight="bold" if (k == "ndp" and not dev) else "normal")
        ax.set_title(METHODS[k], loc="left", fontsize=fs, color=COL[k], fontweight="bold")

    def _draw_life(self, fs):
        t = self.world["lndp"].t
        ax = self.axF; self._clean(ax); self._bands(ax, t)
        for k in KEYS:
            ax.plot(self.rec[k]["found"], color=COL[k], lw=2.2, label=METHODS[k].split(" (")[0])
        ax.set_xlim(0, LIFE); ax.set_ylim(0, LIFE); ax.tick_params(labelsize=fs * 0.65); ax.set_xticklabels([])
        ax.legend(loc="upper left", frameon=False, fontsize=fs * 0.65)
        ax.set_title("THIS LIFE: food found so far (background = food side: lavender left, cream right)",
                     loc="left", fontsize=fs * 0.85)
        ax = self.axD; self._clean(ax); self._bands(ax, t)
        for k in KEYS:
            ax.plot(self.rec[k]["dW"], color=COL[k], lw=2.2)
        ax.set_xlim(0, LIFE); ax.set_ylim(0, max(1, max(max(self.rec[k]["dW"]) for k in KEYS) * 1.2))
        ax.tick_params(labelsize=fs * 0.65); ax.set_xticklabels([])
        ax.set_title("total weight change since the life started", loc="left", fontsize=fs * 0.8)
        ax = self.axC; self._clean(ax); self._bands(ax, t)
        for k in KEYS:
            ax.plot(self.rec[k]["conns"], color=COL[k], lw=2.2)
        ax.set_xlim(0, LIFE); ax.set_ylim(0, max(ALLOW_L.sum(), ALLOW_D.sum()) * 0.75)
        ax.tick_params(labelsize=fs * 0.65); ax.set_xlabel("step in this life", fontsize=fs * 0.75)
        ax.set_title("number of connections", loc="left", fontsize=fs * 0.8)


def main():
    ap = argparse.ArgumentParser(description="Normal NDP vs LNDP")
    ap.add_argument("--size", type=float, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--save", default=None); ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(tuple(a.size) if a.size else ((16, 9.5) if a.save else screen_inches()))
    if a.save:
        t0 = time.perf_counter()
        for g in range(a.gens):
            app.evolve_one()
            if (g + 1) % 50 == 0:
                print(f"gen {g + 1}: " + "  ".join(f"{k} {np.mean(app.hist[k][-10:]):.1f}" for k in KEYS)
                      + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
        app.test_switch(); app.new_life()
        for _ in range(T_SA + 1 + a.steps):
            app.life_step()
        app.draw(); app.fig.savefig(a.save, dpi=110)
        print("saved", a.save)
        for k in KEYS:
            print(k, "after-switch:", np.round(app.adapt[k][0], 2), "food/life", round(app.adapt[k][2], 1))
        return
    try:
        win = app.fig.canvas.manager.window
        if hasattr(win, "showMaximized"):
            win.showMaximized()
        elif hasattr(win, "state"):
            try:
                win.state("zoomed")
            except Exception:
                win.attributes("-zoomed", True)
    except Exception:
        pass
    app.start(); plt.show()


if __name__ == "__main__":
    main()