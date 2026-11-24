"""
Paper 3: Lifelong Neural Developmental Programs (LNDP), as a small live simulation.

THE TASK (the paper's non-stationary foraging, shortened)
    A corridor: [left end] [center] [right end]. The agent starts in the center and
    only sees which cell it is in. Food sits at one end. Moving to an end finishes the
    episode (food found = reward 1) and the agent returns to the center. After each
    episode the food switches sides with probability p_switch.
    To do well the agent must REMEMBER where food was last found and switch when it
    stops being there, which requires learning during its life.
    (Paper: 5-cell corridor, 10-step episodes, p_switch 0.5, 10,000 generations.
     Here: 3 cells and p_switch 0.1 so evolution finds solutions in a few minutes;
     at p_switch 0.5 the food side is a coin flip, so memory can't beat guessing.)

THE NETWORK (shared by all versions)
    3 input nodes (where am I) -> 3 hidden -> 3 output nodes (go left, stay, go right).
    Every node has a 2-number STATE, every connection a 2-number STATE (first = weight).
    Starts random (new random wiring and states every lifetime), or empty, or full.
    Before life: 20 steps of spontaneous activity (noise on the inputs), same rules.
    Shared, evolved rules (as in LNDP, simplified from graph transformer + GRUs):
      node rule        new node state from: its state, activity, reward, type, neighbours
      synapse rule     new connection state from: its state, both nodes' activity and
                       states, the reward, and activity x activity x reward terms
      grow / prune     probability to ADD a connection (from the two node states) and
                       to REMOVE one (from the connection state)

THE THREE VERSIONS (evolved side by side with CMA-ES, population 32)
    LNDP                 rules keep running during life, including grow/prune
    LNDP, no structural  weights change during life, wiring never changes
    NDP (frozen)         rules run only before life (development), then frozen

WHAT YOU SEE
    Top left      evolution: food found per 120-step life
    Top right     the three current solutions living in the SAME world, side by side
    Bottom left   the LNDP agent's network live: connections appear (green ring flash)
                  and disappear (red dashed flash), weights change every step
    Bottom right  food found over this life, and how many connections each network has

Controls: Evolve on/off, Reset, starting network (empty / random half / full),
p_switch slider, Play/Pause life, New life, speed.
Keyboard: space = evolve on/off, p = play/pause life, n = new life.

Run:  python lndp_app.py         |  python lndp_app.py --save shot.png --gens 250
Needs: numpy, matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

NI, NH, NO = 3, 3, 3
N = NI + NH + NO
DH, DEDGE, TICKS, T_SA, LIFE = 2, 2, 2, 20, 120
POP, MAX_GENS = 32, 600
CENTER = NI // 2
ALLOW = np.zeros((N, N))
ALLOW[:NI, NI:NI + NH] = 1; ALLOW[NI:NI + NH, NI:NI + NH] = 1; ALLOW[NI:NI + NH, NI + NH:] = 1
np.fill_diagonal(ALLOW, 0)
TYPE = np.zeros((N, 3)); TYPE[:NI, 0] = 1; TYPE[NI:NI + NH, 1] = 1; TYPE[NI + NH:, 2] = 1
NX = DH + 1 + 1 + 3 + DH
NZ = DEDGE + 2 + 1 + 2 + DH + DH

METHODS = ["LNDP", "LNDP, no structural plasticity", "NDP (frozen after development)"]
KEYS = ["lndp", "nosp", "ndp"]
MODE = {"lndp": (True, True), "nosp": (True, False), "ndp": (False, False)}   # (plastic, structural)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"lndp": "#1baf7a", "nosp": "#6250d6", "ndp": "#888780"}
C_POS, C_NEG, C_ADD, C_DEL = "#1baf7a", "#eb6834", "#1baf7a", "#e34948"
BAND_L, BAND_R = "#eeedfe", "#fdf1dc"
IN_LABELS = ["at left end", "at center", "at right end"]
OUT_LABELS = ["go left", "stay", "go right"]


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


# ---------------------------------------------------------------------------
# LNDP model
# ---------------------------------------------------------------------------
def dims(structural):
    return DH * NX + DH + DEDGE * NZ + DEDGE + 1 + ((2 * DH + 1) + (DEDGE + 1) if structural else 0)


def unpack(th, structural):
    th = np.atleast_2d(th); P = len(th); i = 0

    def take(n, shape):
        nonlocal i
        x = th[:, i:i + n].reshape((P,) + shape); i += n; return x

    out = dict(A=take(DH * NX, (DH, NX)), b=take(DH, (DH,)), U=take(DEDGE * NZ, (DEDGE, NZ)),
               ub=take(DEDGE, (DEDGE,)), lr=take(1, ()))
    if structural:
        out.update(sg=take(2 * DH, (2 * DH,)), sgb=take(1, ()), pr=take(DEDGE, (DEDGE,)), prb=take(1, ()))
    return out


class Life:
    """One lifetime for P agents at once."""

    def __init__(self, p, P, rng, density, p_switch, food_schedule=None):
        self.p, self.P, self.rng, self.ps = p, P, rng, p_switch
        self.h = rng.uniform(-1, 1, (P, N, DH)); self.v = np.zeros((P, N))
        dens = np.clip(rng.normal(density, 0.1, (P, 1, 1)), 0, 1) if 0 < density < 1 else density
        self.A = (rng.random((P, N, N)) < dens) * ALLOW
        self.e = rng.uniform(-1, 1, (P, N, N, DEDGE)) * self.A[..., None]
        self.r = np.zeros(P); self.pos = np.full(P, CENTER)
        self.schedule = food_schedule
        self.food = np.where(rng.random(P) < 0.5, 0, NI - 1) if food_schedule is None else np.full(P, food_schedule[0])
        self.k = np.zeros(P, int); self.found = np.zeros(P); self.t = 0
        self.lr = 0.3 * np.exp(np.clip(p["lr"], -3, 1))
        self.added = np.zeros((P, N, N)); self.removed = np.zeros((P, N, N)); self.last_a = None

    def W(self):
        return self.e[..., 0] * self.A

    def tick(self, obs):
        W = self.W()
        for _ in range(TICKS):
            self.v[:, :NI] = obs
            self.v = np.tanh(np.einsum('pi,pij->pj', self.v, W))
        self.v[:, :NI] = obs

    def update(self, plastic, structural):
        self.added[:] = 0; self.removed[:] = 0
        if not plastic:
            return
        p, P = self.p, self.P
        nb = (self.A + self.A.transpose(0, 2, 1)) > 0
        mh = np.einsum('pij,pjd->pid', nb, self.h) / np.maximum(nb.sum(2, keepdims=True), 1)
        x = np.concatenate([self.h, self.v[..., None], np.repeat(self.r[:, None, None], N, 1),
                            np.repeat(TYPE[None], P, 0), mh], -1)
        self.h = self.h + 0.5 * (np.tanh(np.einsum('pdx,pnx->pnd', p["A"], x) + p["b"][:, None, :]) - self.h)
        vi = np.broadcast_to(self.v[:, :, None], (P, N, N)); vj = np.broadcast_to(self.v[:, None, :], (P, N, N))
        rr = np.broadcast_to(self.r[:, None, None], (P, N, N))
        hi = np.broadcast_to(self.h[:, :, None, :], (P, N, N, DH)); hj = np.broadcast_to(self.h[:, None, :, :], (P, N, N, DH))
        z = np.concatenate([self.e, vi[..., None], vj[..., None], rr[..., None], (vi * vj)[..., None],
                            (rr * vi * vj)[..., None], hi, hj], -1)
        de = np.tanh(np.einsum('pez,pijz->pije', p["U"], z) + p["ub"][:, None, None, :])
        self.e = np.clip(self.e + self.lr[:, None, None, None] * de, -3, 3) * self.A[..., None]
        if structural:
            hh = np.concatenate([hi, hj], -1)
            pg = 1 / (1 + np.exp(-(np.einsum('pk,pijk->pij', p["sg"], hh) + p["sgb"][:, None, None] - 3)))
            pp = 1 / (1 + np.exp(-(np.einsum('pk,pijk->pij', p["pr"], self.e) + p["prb"][:, None, None] - 3)))
            add = (self.rng.random((P, N, N)) < pg) * ALLOW * (1 - self.A)
            rem = (self.rng.random((P, N, N)) < pp) * self.A
            self.A = np.clip(self.A + add - rem, 0, 1)
            self.e = np.where(add[..., None] > 0, 0.1 * self.rng.standard_normal(self.e.shape), self.e) * self.A[..., None]
            self.added, self.removed = add, rem

    def development(self, structural):
        """Spontaneous activity before life: noise on the inputs, no reward, same rules."""
        s = np.zeros((self.P, NI))
        for _ in range(T_SA):
            s = s + 0.1 * (0 - s) + 0.3 * self.rng.standard_normal((self.P, NI)); self.r = np.zeros(self.P)
            self.tick(np.maximum(s, 0)); self.update(True, structural)

    def step(self, plastic, structural):
        self.tick(np.eye(NI)[self.pos])
        out = self.v[:, NI + NH:]
        pr = np.exp(3 * (out - out.max(1, keepdims=True))); pr /= pr.sum(1, keepdims=True)
        a = (pr.cumsum(1) > self.rng.random((self.P, 1))).argmax(1)
        self.v[:, NI + NH:] = np.eye(NO)[a]                    # the chosen action neuron fires
        self.pos = np.clip(self.pos + a - 1, 0, NI - 1); self.k += 1
        got = self.pos == self.food
        self.r = got.astype(float); self.found += got
        reset = got | (self.pos == 0) | (self.pos == NI - 1) | (self.k >= 10)
        self.t += 1
        if self.schedule is not None:
            self.food = np.full(self.P, self.schedule[min(self.t, len(self.schedule) - 1)])
        else:
            sw = reset & (self.rng.random(self.P) < self.ps)
            self.food = np.where(sw, NI - 1 - self.food, self.food)
        self.arrived = self.pos.copy()
        self.pos = np.where(reset, CENTER, self.pos); self.k = np.where(reset, 0, self.k)
        self.update(plastic, structural)
        self.last_a = a
        return a


def fitness(th, key, rng, p_switch, density, lives=2):
    plastic, structural = MODE[key]
    p = unpack(th, key == "lndp"); P = len(np.atleast_2d(th)); tot = np.zeros(P)
    for _ in range(lives):
        L = Life(p, P, rng, density, p_switch)
        L.development(key == "lndp")
        for _ in range(LIFE):
            L.step(plastic, structural)
        tot += L.found
    return tot / lives


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class App:
    def __init__(self):
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
            "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.fig = plt.figure(figsize=(15, 10.5))
        self.fig.canvas.manager.set_window_title("Paper 3: LNDP")
        gs = self.fig.add_gridspec(2, 2, left=0.05, right=0.985, top=0.875, bottom=0.225,
                                   hspace=0.42, wspace=0.16, height_ratios=[1, 1.15])
        self.axE = self.fig.add_subplot(gs[0, 0]); self.axW = self.fig.add_subplot(gs[0, 1])
        self.axN = self.fig.add_subplot(gs[1, 0])
        sub = gs[1, 1].subgridspec(2, 1, hspace=0.55)
        self.axF = self.fig.add_subplot(sub[0]); self.axC = self.fig.add_subplot(sub[1])
        self.title = self.fig.text(0.05, 0.955, "", fontsize=13, va="center")
        self.fig.text(0.05, 0.925, "Task: corridor [left end][center][right end]; food at one end, switches sides "
                      "after an episode with probability p_switch. The agent only sees which cell it is in.",
                      fontsize=9.5, color=INK2, va="center")
        self.density = 0.5
        self.playing = True
        self._controls()
        self.timer = self.fig.canvas.new_timer(interval=150)
        self.timer.add_callback(self._tick)
        self.evolving = False
        self.reset()

    # ---------------- evolution ----------------
    def reset(self):
        if self.evolving:
            self.toggle_evolve()
        self.es = {k: CMAES(np.zeros(dims(k == "lndp")), 0.5, POP, seed=i) for i, k in enumerate(KEYS)}
        self.hist = {k: [] for k in KEYS}
        self.rng = np.random.default_rng(0)
        self.gen = 0
        self.new_life()

    def evolve_one(self):
        if self.gen >= MAX_GENS:
            return
        for k in KEYS:
            X = self.es[k].ask()
            f = fitness(X, k, self.rng, self.s_p.val, self.density)
            self.es[k].tell(f); self.hist[k].append(f.mean())
        self.gen += 1

    # ---------------- watched life ----------------
    def new_life(self):
        seed = int(np.random.default_rng().integers(1 << 30))
        r = np.random.default_rng(seed)
        # one shared world: food side over time (switches with p_switch per step)
        sched = [int(r.integers(2)) * (NI - 1)]
        for _ in range(LIFE):
            sched.append(NI - 1 - sched[-1] if r.random() < self.s_p.val else sched[-1])
        self.sched = np.array(sched)
        self.lives = {}
        for i, k in enumerate(KEYS):
            L = Life(unpack(self.es[k].m, k == "lndp"), 1, np.random.default_rng(seed + 1 + i),
                     self.density, self.s_p.val, food_schedule=self.sched)
            L.development(k == "lndp")
            L.added[:] = 0; L.removed[:] = 0
            self.lives[k] = L
        self.rec = {k: dict(found=[0], conns=[int(self.lives[k].A.sum())]) for k in KEYS}
        self.flash = []
        self.draw()

    def life_step(self):
        if self.lives["lndp"].t >= LIFE:
            self.new_life(); return
        for k in KEYS:
            L = self.lives[k]
            L.step(*MODE[k])
            self.rec[k]["found"].append(L.found[0]); self.rec[k]["conns"].append(int(L.A.sum()))
        L = self.lives["lndp"]
        self.flash = [(i, j, "add") for i, j in zip(*np.where(L.added[0] > 0))] + \
                     [(i, j, "del") for i, j in zip(*np.where(L.removed[0] > 0))]

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig
        self.b_evo = Button(f.add_axes([0.05, 0.10, 0.10, 0.05]), "Evolve: off")
        self.b_rst = Button(f.add_axes([0.16, 0.10, 0.08, 0.05]), "Reset")
        self.b_play = Button(f.add_axes([0.05, 0.035, 0.10, 0.05]), "Pause life")
        self.b_new = Button(f.add_axes([0.16, 0.035, 0.08, 0.05]), "New life")
        self.b_evo.on_clicked(lambda e: self.toggle_evolve())
        self.b_rst.on_clicked(lambda e: self.reset())
        self.b_play.on_clicked(lambda e: self.toggle_play())
        self.b_new.on_clicked(lambda e: self.new_life())
        axr = f.add_axes([0.29, 0.03, 0.17, 0.12])
        axr.set_title("starting network", fontsize=9.5, loc="left", color=INK2)
        self.r_d = RadioButtons(axr, ["empty (no connections)", "random half", "full"], active=1,
                                activecolor=COL["lndp"])
        self.r_d.on_clicked(self._set_density)
        for s in axr.spines.values():
            s.set_visible(False)
        self.s_p = Slider(f.add_axes([0.56, 0.11, 0.17, 0.025]), "p_switch", 0.0, 0.5, valinit=0.1,
                          valstep=0.05, color=COL["lndp"])
        self.s_speed = Slider(f.add_axes([0.56, 0.06, 0.17, 0.025]), "life steps per frame", 1, 10,
                              valinit=1, valstep=1, color=INK2)
        f.text(0.79, 0.15, "Network panel:\n"
               "  aqua = positive weight, orange = negative\n"
               "  line width = size of weight\n"
               "  green glow = connection just grown\n"
               "  red dashed = connection just pruned\n"
               "  node fill = activity (red +, blue −)",
               fontsize=9, color=INK2, va="top", linespacing=1.5)
        f.text(0.49, 0.012, "Keys: space = evolve on/off, p = play/pause life, n = new life",
               fontsize=8.5, color=INK2)
        f.canvas.mpl_connect("key_press_event", self._key)

    def _key(self, e):
        {" ": self.toggle_evolve, "p": self.toggle_play, "n": self.new_life}.get(e.key, lambda: None)()

    def _set_density(self, label):
        self.density = {"empty (no connections)": 0.0, "random half": 0.5, "full": 1.0}[label]
        self.new_life()

    def toggle_evolve(self):
        self.evolving = not self.evolving
        self.b_evo.label.set_text("Evolve: on" if self.evolving else "Evolve: off")
        self.fig.canvas.draw_idle()

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause life" if self.playing else "Play life")
        self.fig.canvas.draw_idle()

    def start(self):
        self.timer.start()

    def _tick(self):
        if self.evolving:
            self.evolve_one()
            if self.gen % 25 == 0:
                self.new_life(); return
        if self.playing:
            for _ in range(int(self.s_speed.val)):
                self.life_step()
        self.draw()

    # ---------------- drawing ----------------
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
        ps = self.s_p.val
        self.title.set_text(f"Paper 3: LNDP — a network that keeps growing and rewiring during its life"
                            f"    generation {self.gen}/{MAX_GENS}")
        # evolution
        ax = self.axE; self._clean(ax)
        for name, k in zip(METHODS, KEYS):
            h = np.array(self.hist[k])
            if len(h):
                sm = np.convolve(h, np.ones(7) / 7, mode="valid") if len(h) >= 7 else h
                ax.plot(np.arange(len(sm)) + len(h) - len(sm), sm, color=COL[k], lw=2.2,
                        label=f"{name} ({dims(k == 'lndp')} numbers)")
        ideal, rnd = LIFE * (1 - ps), LIFE / 3
        ax.axhline(ideal, color=INK2, lw=1, ls=":"); ax.axhline(rnd, color=INK2, lw=1, ls=":")
        xmax = max(60, self.gen)
        ax.text(xmax, ideal + 2, "ideal: always go where food was last", fontsize=8.5, color=INK2, ha="right")
        ax.text(xmax, rnd + 2, "random moves", fontsize=8.5, color=INK2, ha="right")
        ax.set_xlim(0, xmax); ax.set_ylim(0, LIFE + 50); ax.set_yticks(range(0, LIFE + 1, 20))
        ax.set_title(f"Evolution: food found per {LIFE}-step life (population average)", loc="left")
        ax.set_xlabel("generation")
        if self.gen:
            ax.legend(loc="upper left", frameon=False, fontsize=8.8, ncol=1)
        else:
            ax.text(0.5, 0.55, "press  Evolve  to start", transform=ax.transAxes, ha="center", color=INK2)
        # world: three corridors
        ax = self.axW; ax.cla(); ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.3, 3.3); ax.axis("off")
        t = self.lives["lndp"].t
        food = self.sched[min(t, LIFE)]
        for row, (name, k) in enumerate(zip(METHODS, KEYS)):
            y = 2.6 - row * 1.15
            L = self.lives[k]
            for c in range(NI):
                x0 = 4.2 + c * 1.9
                ax.add_patch(plt.Rectangle((x0, y - 0.32), 1.7, 0.64, fc=SURFACE, ec=INK2, lw=0.8))
                if c == food:
                    ax.text(x0 + 0.45, y, "food", ha="center", va="center", fontsize=8.5, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#fdf1dc", ec="#c98500", lw=1))
            if L.last_a is not None:
                arr = int(L.arrived[0])
                hit = arr == (self.sched[t - 1] if t > 0 else food)
                ax.scatter([4.2 + arr * 1.9 + 1.25], [y], s=170, color=COL[k], edgecolors=INK,
                           zorder=4, marker="o")
                ax.text(10.15, y, "✓" if hit and arr != CENTER else ("✗" if arr != CENTER else "·"),
                        ha="right", va="center", fontsize=13, color="#1baf7a" if hit and arr != CENTER else "#e34948")
            ax.text(0.0, y + 0.12, name, fontsize=9.5, va="center", color=COL[k], fontweight="bold")
            ax.text(0.0, y - 0.2, f"food so far: {int(L.found[0])}", fontsize=9, va="center", color=INK2)
        ax.set_title(f"The three current solutions in the same world   step {t}/{LIFE}   (dot = where it moved)",
                     loc="left")
        # network of the LNDP agent
        self._draw_net(self.lives["lndp"])
        # food over life
        ax = self.axF; self._clean(ax); self._bands(ax, t)
        for name, k in zip(METHODS, KEYS):
            ax.plot(self.rec[k]["found"], color=COL[k], lw=2.2, label=name)
        ax.set_xlim(0, LIFE); ax.set_ylim(0, LIFE)
        ax.set_title("This life: food found so far (background = food side: lavender left, cream right)",
                     loc="left", fontsize=10)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        ax = self.axC; self._clean(ax); self._bands(ax, t)
        for name, k in zip(METHODS, KEYS):
            ax.plot(self.rec[k]["conns"], color=COL[k], lw=2.2)
        ax.set_xlim(0, LIFE); ax.set_ylim(-1, ALLOW.sum() + 2)
        ax.set_title(f"Connections in each network (of {int(ALLOW.sum())} allowed)", loc="left", fontsize=10)
        ax.set_xlabel("step in this life")
        self.fig.canvas.draw_idle()

    def _draw_net(self, L):
        ax = self.axN; ax.cla(); ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.7, 10.2); ax.axis("off")
        pos = {}
        for i in range(NI):
            pos[i] = (1.6, 8.4 - i * 3.2)
        for j in range(NH):
            pos[NI + j] = (5.0, 8.4 - j * 3.2)
        for k in range(NO):
            pos[NI + NH + k] = (8.4, 8.4 - k * 3.2)
        W, A = L.W()[0], L.A[0]

        def curve(i, j):
            (x0, y0), (x1, y1) = pos[i], pos[j]
            if NI <= i < NI + NH and NI <= j < NI + NH:          # hidden -> hidden: bow to the left
                tt = np.linspace(0, 1, 30); bow = 1.2 + 0.3 * abs(y1 - y0)
                xs = x0 + (x1 - x0) * tt - bow * np.sin(np.pi * tt) * (1 if y1 < y0 else -1) * 0.5
                ys = y0 + (y1 - y0) * tt
                return xs, ys
            return [x0, x1], [y0, y1]

        for i, j in zip(*np.where(ALLOW > 0)):
            if A[i, j] > 0:
                w = W[i, j]
                xs, ys = curve(i, j)
                ax.plot(xs, ys, color=C_POS if w >= 0 else C_NEG, lw=0.4 + 1.5 * abs(w), alpha=0.85,
                        solid_capstyle="round", zorder=1)
        for i, j, kind in self.flash:
            xs, ys = curve(i, j)
            if kind == "add":
                ax.plot(xs, ys, color=C_ADD, lw=4, alpha=0.35, zorder=0)
            else:
                ax.plot(xs, ys, color=C_DEL, lw=1.5, ls="--", alpha=0.9, zorder=1)
        act = L.v[0]
        for n, (x, y) in pos.items():
            ax.scatter([x], [y], s=520, color=plt.cm.RdBu_r(0.5 + 0.5 * np.clip(act[n], -1, 1)),
                       edgecolors=INK2, linewidths=1, zorder=3)
        for i in range(NI):
            ax.text(pos[i][0] - 0.55, pos[i][1], IN_LABELS[i], ha="right", va="center", fontsize=9)
        for k in range(NO):
            ax.text(pos[NI + NH + k][0] + 0.55, pos[NI + NH + k][1], OUT_LABELS[k], ha="left", va="center",
                    fontsize=9)
        if L.last_a is not None:
            a = int(L.last_a[0])
            ax.scatter(*pos[NI + NH + a], s=1000, facecolors="none", edgecolors=INK, linewidths=2, zorder=2)
        n_add, n_del = int(L.added[0].sum()), int(L.removed[0].sum())
        ax.text(0, -0.55, f"this step: {n_add} connection(s) grown, {n_del} pruned   "
                f"total {int(A.sum())} of {int(ALLOW.sum())}", fontsize=9.5, color=INK2)
        ax.set_title("LNDP agent's network, live (inputs → hidden → outputs; ring = chosen action)",
                     loc="left")


def main():
    ap = argparse.ArgumentParser(description="LNDP toy simulation")
    ap.add_argument("--save", default=None); ap.add_argument("--gens", type=int, default=250)
    ap.add_argument("--density", default="random half", choices=["empty (no connections)", "random half", "full"])
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App()
    if a.save:
        app.r_d.set_active(["empty (no connections)", "random half", "full"].index(a.density))
        for _ in range(a.gens):
            app.evolve_one()
        app.new_life()
        for _ in range(a.steps):
            app.life_step()
        app.draw(); app.fig.savefig(a.save, dpi=110)
        print("Saved", a.save, {k: round(float(np.mean(app.hist[k][-10:])), 1) for k in KEYS},
              {k: int(app.lives[k].found[0]) for k in KEYS})
    else:
        app.start(); plt.show()


if __name__ == "__main__":
    main()