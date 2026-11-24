"""
From CMA-ES scores to building blocks: what one generation changes, and what that does.

One generation, shown left to right:
    (A) the 128 robots' scores, ranked; the best 64 get a pull weight w_i
    (B) the new mean = old mean + weighted pull toward the best 64: how much EACH of the 565
        genome numbers moved, grouped by building block
    (C) the building block you pick, BEFORE and AFTER, number by number
        (a neuron type = 4x4 table + 4 biases; a GRU rule = 12 rows x 14 columns = 168 numbers)
    (D) a network built from the OLD mean vs one built from the NEW mean: same random wirings,
        same starts, so the only difference is the genome. Steps survived per episode.
    (E) history over generations

"Score used by CMA-ES" lets you change what CMA-ES is told:
    real      the CartPole lifetime score (normal SFNN training)
    random    random numbers: CMA-ES learns nothing, the mean just wanders
    reversed  worse = better: CMA-ES pushes the genome toward robots that fall quickly
Watch how the building blocks and the robot built from the mean change in each case.
"Compare" switches (B), (C), (D) between "this one step" and "since generation 0": one step
moves 565 numbers only a little, the changes add up over many generations.

Model: SFNN on CartPole, 4 sensors, 4 hidden, 2 actions, 565-number genome
(same as sfnn_step_by_step.py). One generation = 128 real lives of 8 episodes.

Run:   python cmaes_to_blocks.py
       python cmaes_to_blocks.py --save shot.png --gens 6     (headless)
Needs: numpy, matplotlib (GUI backend such as TkAgg or QtAgg)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons

# ---------------------------------------------------------------------------
# SFNN model (same as the step-by-step app)
# ---------------------------------------------------------------------------
NI, NH, NO, D = 4, 4, 2, 4
N_EP, MAX_STEPS, POP = 8, 200, 128
P_CONN, INIT_STD = 0.5, 0.5
OBS_SCALE = np.array([2.4, 2.0, 0.21, 2.0])
K_IN, K_HID = NI * P_CONN, NH * P_CONN
BLOCKS = [("input neuron type", 20), ("hidden neuron type", 20), ("output neuron type", 20),
          ("rule input→hidden", 168), ("rule hidden→hidden", 168),
          ("rule hidden→output", 168), ("learning rate", 1)]
SHORT = ["input\nneuron", "hidden\nneuron", "output\nneuron", "rule\nin→hid", "rule\nhid→hid",
         "rule\nhid→out", "learn.\nrate"]
DIM = sum(n for _, n in BLOCKS)
EDGES = np.cumsum([0] + [n for _, n in BLOCKS])

SURFACE, INK, INK2, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c7c0", "#e4e3df"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#6250d6"
BLOCK_COL = [GREEN, BLUE, ORANGE, "#0f8f8a", PURPLE, "#d6559a", "#b8860b"]
DIVERGE = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fbe3d6", "#f7f7f5", "#d3e5f7", "#2a78d6"])
CHANGE = LinearSegmentedColormap.from_list("chg", ["#b5179e", "#f3d1ec", "#f7f7f5", "#cdeee0", "#0b8a5f"])
MODES = ["real", "random", "reversed"]
MODE_COL = {"real": BLUE, "random": INK2, "reversed": RED}


def sig(v):
    return 1 / (1 + np.exp(-v))


def unpack(theta):
    theta = np.asarray(theta); lead = theta.shape[:-1]; i = 0; p = {}

    def take(n, shape):
        nonlocal i
        v = theta[..., i:i + n].reshape(lead + shape); i += n; return v

    for t in ("in", "hid", "out"):
        p["W_" + t] = take(D * D, (D, D)); p["b_" + t] = take(D, (D,))
    for s in ("ih", "hh", "ho"):
        p["Wx_" + s] = take(3 * D * (2 * D + 1), (3, D, 2 * D + 1))
        p["U_" + s] = take(3 * D * D, (3, D, D))
        p["bg_" + s] = take(3 * D, (3, D))
    p["lr"] = sig(take(1, ()) - 2.0)
    return p


def block_matrix(theta, k):
    """The genome numbers of block k arranged the way the block uses them."""
    v = theta[EDGES[k]:EDGES[k + 1]]
    if k < 3:                                  # neuron type: 4x4 table | 4 biases  -> 4 x 5
        return np.column_stack([v[:16].reshape(4, 4), v[16:20]])
    if k < 6:                                  # GRU rule: rows = 3 parts x 4 outputs, cols = 9 inputs | 4 weight | bias
        Wx = v[:108].reshape(3, 4, 9); U = v[108:156].reshape(3, 4, 4); b = v[156:168].reshape(3, 4, 1)
        return np.concatenate([Wx, U, b], 2).reshape(12, 14)
    return v.reshape(1, 1)


def cart_step_batch(s, right):
    x, xd, ph, phd = s.T
    F = np.where(right, 10.0, -10.0)
    tau = (F + 0.05 * phd ** 2 * np.sin(ph)) / 1.1
    phdd = (9.8 * np.sin(ph) - np.cos(ph) * tau) / (0.5 * (4 / 3 - 0.1 * np.cos(ph) ** 2 / 1.1))
    xdd = tau - 0.05 * phdd * np.cos(ph) / 1.1
    s2 = np.stack([x + 0.02 * xd, xd + 0.02 * xdd, ph + 0.02 * phd, phd + 0.02 * phdd], 1)
    return s2, (np.abs(s2[:, 0]) > 2.4) | (np.abs(s2[:, 2]) > 0.2095)


def gru_batch(H, M, p, s, pre, post, r, live):
    Wx, U, bg = p["Wx_" + s], p["U_" + s], p["bg_" + s]
    xp = np.einsum('pgek,pjk->pgje', Wx[..., :D], pre)
    xq = np.einsum('pgek,pik->pgie', Wx[..., D:2 * D], post)
    xr = Wx[..., 2 * D] * r[:, None, None] + bg
    X = xp[:, :, None] + xq[:, :, :, None] + xr[:, :, None, None]
    Uh = np.einsum('pgek,pijk->pgije', U, H)
    z = sig(X[:, 0] + Uh[:, 0]); rho = sig(X[:, 1] + Uh[:, 1])
    n = np.tanh(X[:, 2] + rho * Uh[:, 2])
    Hn = H + p["lr"][:, None, None, None] * ((1 - z) * n + z * H - H)
    return np.where(M[..., None] & live[:, None, None, None], Hn, H)


def lives(X, rng):
    """Each row of X = one genome, one robot, one life (own random wiring, own random start weights)."""
    P = len(X); p = unpack(X)
    M1 = rng.random((P, NH, NI)) < P_CONN
    pi, hi = np.nonzero(~M1.any(2)); M1[pi, hi, rng.integers(NI, size=len(pi))] = True
    M2 = (rng.random((P, NH, NH)) < P_CONN) & ~np.eye(NH, dtype=bool)
    M3 = rng.random((P, NO, NH)) < P_CONN
    F1, F2, F3 = M1.astype(float), M2.astype(float), M3.astype(float)
    H1 = INIT_STD * rng.normal(size=(P, NH, NI, D)) * M1[..., None]
    H2 = INIT_STD * rng.normal(size=(P, NH, NH, D)) * M2[..., None]
    H3 = INIT_STD * rng.normal(size=(P, NO, NH, D)) * M3[..., None]
    a_h = np.zeros((P, NH, D)); a_o = np.zeros((P, NO, D)); ret = np.zeros((P, N_EP))
    w_in = p["W_in"].sum(-1)
    for e in range(N_EP):
        s = (rng.random((P, 4)) - 0.5) * 0.1
        live = np.ones(P, dtype=bool); r = np.zeros(P)
        for _ in range(MAX_STEPS):
            a_in = np.tanh((s / OBS_SCALE)[:, :, None] * w_in[:, None, :] + p["b_in"][:, None, :])
            for _ in range(2):
                m_h = (np.einsum('pij,pijd,pjd->pid', F1, H1, a_in) / np.sqrt(K_IN)
                       + np.einsum('pij,pijd,pjd->pid', F2, H2, a_h) / np.sqrt(K_HID))
                a_h = np.where(live[:, None, None],
                               np.tanh(np.einsum('ped,pnd->pne', p["W_hid"], m_h) + p["b_hid"][:, None]), a_h)
                m_o = np.einsum('pij,pijd,pjd->pid', F3, H3, a_h) / np.sqrt(K_HID)
                a_o = np.where(live[:, None, None],
                               np.tanh(np.einsum('ped,pnd->pne', p["W_out"], m_o) + p["b_out"][:, None]), a_o)
            right = a_o[:, 1, 0] > a_o[:, 0, 0]
            H1 = gru_batch(H1, M1, p, "ih", a_in, a_h, r, live)
            H2 = gru_batch(H2, M2, p, "hh", a_h, a_h, r, live)
            H3 = gru_batch(H3, M3, p, "ho", a_h, a_o, r, live)
            s2, fell = cart_step_batch(s, right)
            ret[:, e] += live
            s = np.where(live[:, None], s2, s); live = live & ~fell; r = np.ones(P)
            if not live.any():
                break
    w = np.arange(1, N_EP + 1) / np.arange(1, N_EP + 1).sum()
    return (ret / MAX_STEPS * w).sum(1), ret


class CMAES:
    def __init__(self, n, lam, sigma0, rng):
        self.n, self.lam, self.mu, self.rng = n, lam, lam // 2, rng
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum()
        me = self.me = 1 / np.sum(self.w ** 2)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.zeros(n); self.sigma = sigma0; self.C = np.eye(n)
        self.pc = np.zeros(n); self.ps = np.zeros(n); self.B = np.eye(n); self.D = np.ones(n); self.g = 0

    def ask(self):
        self.Y = (self.rng.standard_normal((self.lam, self.n)) * self.D) @ self.B.T
        return self.m + self.sigma * self.Y

    def tell(self, fit):
        Ys = self.Y[np.argsort(-fit)[:self.mu]]; yw = self.w @ Ys
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * (self.B @ ((self.B.T @ yw) / self.D))
        psn = np.linalg.norm(self.ps)
        hs = float(psn / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Ys.T * self.w) @ Ys)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= np.exp((self.cs / self.ds) * (psn / self.chiN - 1)); self.g += 1
        ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-20))


def test_mean(m, seed, n=16):
    """n lives of the SAME genome m, with fixed seeds, so old and new mean meet identical wirings."""
    _, ret = lives(np.repeat(m[None], n, 0), np.random.default_rng(seed))
    return ret.mean(0)


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
        return 16, 9


class App:
    def __init__(self, size):
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                             "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
                             "axes.labelcolor": INK2})
        self.mode = "real"; self.block = 1; self.since_start = False
        self.fig = plt.figure(figsize=size)
        try:
            self.fig.canvas.manager.set_window_title("CMA-ES scores → genome → building blocks → robot")
        except AttributeError:
            pass
        f = self.fig
        self.axA = f.add_axes([0.045, 0.52, 0.25, 0.28]); self.axB = f.add_axes([0.40, 0.52, 0.27, 0.28])
        self.axD = f.add_axes([0.72, 0.52, 0.265, 0.28])
        self.axC = f.add_axes([0.16, 0.16, 0.51, 0.27]); self.axE = f.add_axes([0.72, 0.16, 0.265, 0.27])
        self.title = f.text(0.045, 0.965, "", va="center", fontweight="bold")
        self.flow = f.text(0.045, 0.905, "", va="center", color=INK2, family="monospace")
        self._controls()
        f.canvas.mpl_connect("resize_event", lambda e: self.draw())
        self.reset()

    def _controls(self):
        f = self.fig; c = "#ecebe7"
        self.btns = []
        for lab, x, w, fn in (("Run 1 generation", 0.045, 0.1, lambda: self.run(1)),
                              ("Run 10 generations", 0.15, 0.11, lambda: self.run(10)),
                              ("Reset", 0.265, 0.06, self.reset),
                              ("Compare: last step", 0.335, 0.15, self.toggle_compare)):
            b = Button(f.add_axes([x, 0.015, w, 0.045]), lab, color=c, hovercolor="#dcdad3")
            b.on_clicked(lambda e, fn=fn: fn()); self.btns.append(b)
        axm = f.add_axes([0.53, 0.0, 0.25, 0.075]); axm.set_title("score used by CMA-ES", fontsize=9, loc="left", color=INK2)
        for s in axm.spines.values():
            s.set_visible(False)
        self.r_mode = RadioButtons(axm, ["real (CartPole score)", "random (no information)", "reversed (worse = better)"],
                                   active=0, activecolor=BLUE)
        self.r_mode.on_clicked(lambda lab: self._set_mode(MODES[[l.get_text() for l in self.r_mode.labels].index(lab)]))
        axb = f.add_axes([0.02, 0.16, 0.12, 0.27]); axb.set_title("block to inspect (C)", fontsize=9, loc="left", color=INK2)
        for s in axb.spines.values():
            s.set_visible(False)
        self.r_block = RadioButtons(axb, [n for n, _ in BLOCKS], active=1, activecolor=PURPLE)
        self.r_block.on_clicked(lambda lab: self._set_block([n for n, _ in BLOCKS].index(lab)))

    def toggle_compare(self):
        self.since_start = not self.since_start
        self.btns[3].label.set_text("Compare: since generation 0" if self.since_start else "Compare: last step")
        self.draw()

    def _set_mode(self, m):
        self.mode = m; self.draw()

    def _set_block(self, k):
        self.block = k; self.draw()

    def reset(self):
        self.rng = np.random.default_rng(1)
        self.es = CMAES(DIM, POP, 0.5, self.rng)
        self.gen = 0; self.last = None; self.hist = []
        self.m0 = self.es.m.copy()
        self.mean_test = [test_mean(self.es.m, 7)]
        self.draw()

    def run(self, n):
        for _ in range(n):
            self.title.set_text(f"running generation {self.gen + 1}: 128 lives ..."); self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            m_old = self.es.m.copy()
            X = self.es.ask()
            real, ret = lives(X, self.rng)
            used = {"real": real, "random": self.rng.random(POP), "reversed": -real}[self.mode]
            order = np.argsort(-used)
            self.es.tell(used)
            self.gen += 1
            after = test_mean(self.es.m, 7)
            self.last = dict(X=X, real=real, used=used, order=order, m_old=m_old, m_new=self.es.m.copy(),
                             before=self.mean_test[-1], after=after, mode=self.mode)
            self.mean_test.append(after)
            self.hist.append((real.mean(), real.max(), (after / MAX_STEPS).mean(), self.mode))
        self.draw()

    # ---------------- drawing ----------------
    def _fs(self):
        w, h = self.fig.get_size_inches()
        return float(np.clip(min(w / 1.75, h) * 1.1, 7.5, 14))

    @staticmethod
    def _clean(ax, grid=True):
        ax.cla()
        if grid:
            ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def draw(self):
        fs = self.fs = self._fs()
        self.title.set_text(f"What one CMA-ES step changes: scores → new mean genome → new building blocks "
                            f"→ new robot      (generation {self.gen})")
        self.title.set_fontsize(fs * 1.2)
        self.flow.set_text("(A) rank the 128 scores  →  (B) new mean = old mean pulled toward the best 64  "
                           "→  (C) every block's numbers change  →  (D) robots built from the new mean behave differently")
        self.flow.set_fontsize(fs * 0.85)
        for b in self.btns:
            b.label.set_fontsize(fs * 0.85)
        for r in (self.r_mode, self.r_block):
            for l in r.labels:
                l.set_fontsize(fs * 0.78)
        self._draw_A(fs); self._draw_B(fs); self._draw_C(fs); self._draw_D(fs); self._draw_E(fs)
        self.fig.canvas.draw_idle()

    def _wait(self, ax, fs, title):
        ax.set_title(title, loc="left", fontsize=fs)
        ax.text(0.5, 0.5, "press  Run 1 generation", ha="center", va="center", transform=ax.transAxes,
                color=INK2, fontsize=fs)

    def _draw_A(self, fs):
        ax = self.axA; self._clean(ax)
        L = self.last
        if L is None:
            self._wait(ax, fs, "(A) Scores of the 128 robots"); return
        used = L["used"][L["order"]]; real = L["real"][L["order"]]
        x = np.arange(POP)
        ax.bar(x[:64], used[:64], color=MODE_COL[L["mode"]], width=1.0, label="best 64 (pull the mean)")
        ax.bar(x[64:], used[64:], color=FAINT, width=1.0, label="worst 64 (ignored)")
        if L["mode"] != "real":
            ax.plot(x, real, color=INK, lw=0.8, ls=":", label="their real CartPole score")
        ax2 = ax.twinx()
        ax2.plot(np.arange(64), self.es.w, color=ORANGE, lw=2, label="pull weight w_i")
        ax2.set_ylim(0, self.es.w[0] * 1.3); ax2.tick_params(labelsize=fs * 0.65, colors=ORANGE)
        ax2.set_ylabel("pull weight w_i", color=ORANGE, fontsize=fs * 0.75)
        for s in ("top",):
            ax2.spines[s].set_visible(False)
        ax.set_xlim(-1, POP); ax.set_xlabel("robots, ranked best → worst by the score CMA-ES is given", fontsize=fs * 0.75)
        ax.set_ylabel("score used", fontsize=fs * 0.75); ax.tick_params(labelsize=fs * 0.65)
        ax.legend(loc="upper right", fontsize=fs * 0.65, frameon=False)
        ax.set_title(f"(A) Scores of the 128 robots, ranked  (mode: {L['mode']})", loc="left", fontsize=fs)

    def _draw_B(self, fs):
        ax = self.axB; self._clean(ax)
        L = self.last
        if L is None:
            self._wait(ax, fs, "(B) How much each genome number moved"); return
        d = L["m_new"] - (self.m0 if self.since_start else L["m_old"])
        for k in range(len(BLOCKS)):
            xs = np.arange(EDGES[k], EDGES[k + 1])
            ax.bar(xs, d[xs], color=BLOCK_COL[k], width=1.0, alpha=1 if k == self.block else 0.45)
        for e in EDGES[1:-1]:
            ax.axvline(e - 0.5, color=INK2, lw=0.6)
        lim = np.abs(d).max() * 1.15
        ax.set_ylim(-lim, lim * 1.45); ax.set_xlim(-1, DIM)
        labels = [(30, "3 neuron\ntypes", INK2, self.block < 3)] + \
            [((EDGES[k] + EDGES[k + 1]) / 2, SHORT[k], BLOCK_COL[k], k == self.block) for k in (3, 4, 5)] + \
            [(DIM - 8, "η", BLOCK_COL[6], self.block == 6)]
        for xm, t, c, bold in labels:
            ax.text(xm, lim * 1.08, t, ha="center", va="bottom", fontsize=fs * 0.62, color=c,
                    fontweight="bold" if bold else "normal")
        ax.set_xlabel("genome number (1 … 565)", fontsize=fs * 0.75)
        ax.set_ylabel("new mean − old mean", fontsize=fs * 0.75); ax.tick_params(labelsize=fs * 0.65)
        ax.set_title(f"(B) How much each of the 565 numbers moved {'since generation 0' if self.since_start else 'this step'}"
                     f"  (total {np.linalg.norm(d):.2f})",
                     loc="left", fontsize=fs)

    def _draw_C(self, fs):
        ax = self.axC; ax.cla(); ax.axis("off")
        k = self.block; name = BLOCKS[k][0]
        L = self.last
        m_old = (self.m0 if self.since_start else L["m_old"]) if L else self.es.m
        m_new = L["m_new"] if L else self.es.m
        A, B = block_matrix(m_old, k), block_matrix(m_new, k)
        span = "generation 0 → now" if self.since_start else "before → after this CMA-ES step"
        ax.set_title(f"(C) Block '{name}' ({BLOCKS[k][1]} numbers): {span}",
                     loc="left", fontsize=fs)
        if k == 6:
            lo, ln = sig(A[0, 0] - 2), sig(B[0, 0] - 2)
            ax.text(0.02, 0.6, f"genome number: {A[0, 0]:+.3f}  →  {B[0, 0]:+.3f}\n"
                    f"learning rate η = σ(number − 2):  {lo:.3f}  →  {ln:.3f}\n\n"
                    f"Every link in every robot of the next generation updates its weight\n"
                    f"with this η (bigger η = weights change faster during life).",
                    fontsize=fs, va="center", family="monospace", transform=ax.transAxes)
            return
        lim = max(np.abs(A).max(), np.abs(B).max())
        dl = max(1e-6, np.abs(B - A).max())
        subs = [ax.inset_axes([0.075, 0.0, 0.27, 0.86]), ax.inset_axes([0.39, 0.0, 0.27, 0.86]),
                ax.inset_axes([0.705, 0.0, 0.27, 0.86])]
        t0 = "GENERATION 0 (start)" if self.since_start else "BEFORE (old mean)"
        for a, M, t, cm, l in ((subs[0], A, t0, DIVERGE, lim), (subs[1], B, "AFTER (new mean)", DIVERGE, lim),
                               (subs[2], B - A, f"CHANGE (scale ±{dl:.2f})", CHANGE, dl)):
            a.imshow(M, cmap=cm, vmin=-l, vmax=l, aspect="auto", interpolation="nearest")
            a.set_title(t, fontsize=fs * 0.8, color=INK2)
            a.tick_params(labelsize=fs * 0.55, length=0)
            if k < 3:
                a.set_xticks([1.5, 4]); a.set_xticklabels(["4×4 table", "bias"])
                a.set_yticks(range(4)); a.set_yticklabels([f"out {i + 1}" for i in range(4)])
                a.axvline(3.5, color=INK, lw=1)
            else:
                a.set_xticks([1.5, 5.5, 8, 10.5, 13])
                a.set_xticklabels(["sender", "receiver", "r", "link's\nweight", "bias"])
                a.set_yticks([1.5, 5.5, 9.5]); a.set_yticklabels(["keep z", "use ρ", "suggest n"])
                for xx in (3.5, 7.5, 8.5, 12.5):
                    a.axvline(xx, color=INK, lw=0.6 if xx != 8.5 else 1.2)
                for yy in (3.5, 7.5):
                    a.axhline(yy, color=INK, lw=1)
            if a is not subs[0]:
                a.set_yticklabels([])
        who = "every hidden neuron" if k == 1 else ("every input neuron" if k == 0 else
              "every output neuron" if k == 2 else "every link of this kind")
        ax.text(0.0, -0.2, f"These numbers are used by {who} in every robot of the next generation.", transform=ax.transAxes,
                fontsize=fs * 0.75, color=INK2)

    def _draw_D(self, fs):
        ax = self.axD; self._clean(ax)
        L = self.last
        x = np.arange(1, N_EP + 1)
        if L is None:
            ax.bar(x, self.mean_test[0], color=FAINT, width=0.6, label="robot from current mean")
            ax.set_title("(D) Robot built from the mean genome", loc="left", fontsize=fs)
        else:
            ref = self.mean_test[0] if self.since_start else L["before"]
            ax.bar(x - 0.2, ref, width=0.4, color=FAINT,
                   label="from generation-0 mean" if self.since_start else "from OLD mean")
            ax.bar(x + 0.2, L["after"], width=0.4, color=MODE_COL[L["mode"]], label="from NEW mean")
            ax.set_title("(D) Same 16 random wirings: " + ("start genome vs now" if self.since_start
                         else "old genome vs new genome"), loc="left", fontsize=fs)
        ax.set_xticks(x); ax.set_xlabel("episode within the life", fontsize=fs * 0.75)
        ax.set_ylabel("steps the pole stayed up (avg of 16 lives)", fontsize=fs * 0.7)
        ax.tick_params(labelsize=fs * 0.65); ax.legend(fontsize=fs * 0.7, frameon=False, loc="upper left")
        ax.set_ylim(0, max(40, np.max(self.mean_test) * 1.3))

    def _draw_E(self, fs):
        ax = self.axE; self._clean(ax)
        ax.set_title("(E) Over generations", loc="left", fontsize=fs)
        if not self.hist:
            ax.text(0.5, 0.5, "no generations yet", ha="center", transform=ax.transAxes, color=INK2, fontsize=fs * 0.9)
            return
        H = self.hist; g = np.arange(1, len(H) + 1)
        ax.plot(g, [h[0] for h in H], color=INK2, lw=1.4, label="population average (real score)")
        ax.plot(g, [h[2] for h in H], color=BLUE, lw=2.4, label="robot from the mean (real score)")
        for i, h in enumerate(H):
            ax.axvspan(i + 0.5, i + 1.5, color=MODE_COL[h[3]], alpha=0.07, lw=0)
        ax.set_xlim(0.5, max(5, len(H)) + 0.5); ax.set_ylim(0, None)
        ax.set_xlabel("generation  (background colour = score mode used)", fontsize=fs * 0.72)
        ax.tick_params(labelsize=fs * 0.65); ax.legend(fontsize=fs * 0.68, frameon=False, loc="upper left")


def main():
    ap = argparse.ArgumentParser(description="CMA-ES scores -> genome -> building blocks -> robot")
    ap.add_argument("--size", type=float, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--save", default=None); ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--mode", default="real", choices=MODES)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(tuple(a.size) if a.size else ((16, 9) if a.save else screen_inches()))
    if a.save:
        app.r_mode.set_active(MODES.index(a.mode))
        app.run(a.gens); app.fig.savefig(a.save, dpi=110); print("saved", a.save)
        for h in app.hist:
            print(f"pop avg {h[0]:.3f}  best {h[1]:.3f}  mean-robot {h[2]:.3f}  ({h[3]})")
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
    plt.show()


if __name__ == "__main__":
    main()