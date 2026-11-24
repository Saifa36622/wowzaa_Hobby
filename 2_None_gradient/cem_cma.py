"""
CEM vs. CMA-ES, side by side, on the same narrow tilted valley.

Interactive desktop app (matplotlib window with buttons).

    Next generation   advance both methods by one generation
    Run 20            animate 20 generations
    Reset             start over from the same starting point
    Keyboard:         space = next generation, r = reset

What you see (top row, one panel per method):
    gray dots      all candidates sampled this generation
    blue dots      winners (top 5 of 10)
    dashed line    trail of the center (mean) over generations
    ellipse        search region = 2 x spread (CEM) or 2 x sigma * shape C (CMA-ES)
    purple arrow   CMA-ES memory path p_c (direction it has been moving recently)
    open circle    the true optimum

Bottom row: loss at the center and size of the search over generations.

Run:   python cem_vs_cmaes_app.py            (random seed each run)
       python cem_vs_cmaes_app.py --seed 3   (repeatable)
Needs: numpy, matplotlib (with a GUI backend such as TkAgg or QtAgg)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# The problem: a narrow valley tilted by 30 degrees (30x steeper across than along)
# ---------------------------------------------------------------------------
THETA = np.radians(30)
CT, ST = np.cos(THETA), np.sin(THETA)
OPT = np.array([3.0, 2.0])          # true optimum
START = np.array([-3.9, -1.4])      # same start for both methods
LAM, MU = 10, 5                     # population size, number of winners


def f(x):
    d = np.asarray(x) - OPT
    u = CT * d[..., 0] + ST * d[..., 1]          # along the valley
    v = -ST * d[..., 0] + CT * d[..., 1]         # across the valley
    return u * u + 30 * v * v


# ---------------------------------------------------------------------------
# CEM: center + spread per knob, refit to the winners every generation
# ---------------------------------------------------------------------------
class CEM:
    def __init__(self, rng, std0=0.5):
        self.rng = rng
        self.m = START.copy()
        self.sd = np.full(2, std0)
        self.trail = [self.m.copy()]
        self.X = np.empty((0, 2))
        self.order = np.arange(0)

    def step(self):
        X = self.m + self.sd * self.rng.standard_normal((LAM, 2))     # 1. sample
        F = f(X)                                                      # 2. score
        self.order = np.argsort(F)
        elites = X[self.order[:MU]]                                   # 3. keep best
        self.m = elites.mean(0)                                       # 4. refit center
        self.sd = np.maximum(elites.std(0), 1e-3)                     #    refit spread
        self.X = X
        self.trail.append(self.m.copy())

    def ellipse(self):
        """(width, height, angle in degrees) of the 2-spread ellipse."""
        return 4 * self.sd[0], 4 * self.sd[1], 0.0

    def readout(self):
        return f"spread x {self.sd[0]:.3f}, y {self.sd[1]:.3f}"

    def size(self):
        return self.sd.max()


# ---------------------------------------------------------------------------
# CMA-ES: center m, step size sigma, shape C, memory paths p_c and p_sigma
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, rng, sigma0=0.5):
        self.rng = rng
        n = 2
        self.n = n
        w = np.log(MU + 0.5) - np.log(np.arange(1, MU + 1))
        self.w = w / w.sum()                                  # ranked weights
        me = 1 / np.sum(self.w ** 2)
        self.me = me
        # automatic rates (standard defaults, computed from n)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        # state
        self.m = START.copy()
        self.sigma = sigma0
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.g = 0
        self.trail = [self.m.copy()]
        self.X = np.empty((0, 2))
        self.order = np.arange(0)

    def step(self):
        ev, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(ev, 1e-20))
        Y = (self.rng.standard_normal((LAM, self.n)) * D) @ B.T       # directions shaped by C
        X = self.m + self.sigma * Y                                   # 1. sample
        F = f(X)                                                      # 2. score
        self.order = np.argsort(F)
        Ys = Y[self.order[:MU]]
        yw = self.w @ Ys                                              # 3. weighted winners
        self.m = self.m + self.sigma * yw                             #    move center
        # 4. step-size memory path p_sigma
        ps_in = B @ ((B.T @ yw) / D)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * ps_in
        hs = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1)))
              < (1.4 + 2 / (self.n + 1)) * self.chiN)
        # 5. shape memory path p_c
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        # 6. shape update: keep old + stretch along path + stretch toward winners
        rank_mu = (Ys * self.w[:, None]).T @ Ys
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = (self.C + self.C.T) / 2
        # 7. step-size update: grow if path longer than random, shrink if shorter
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.g += 1
        self.X = X
        self.trail.append(self.m.copy())

    def axes(self):
        ev, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(ev, 1e-20))
        return D[::-1], B[:, ::-1]               # largest first

    def ellipse(self):
        D, B = self.axes()
        angle = np.degrees(np.arctan2(B[1, 0], B[0, 0]))
        return 4 * self.sigma * D[0], 4 * self.sigma * D[1], angle

    def path_ratio(self):
        return np.linalg.norm(self.ps) / self.chiN

    def stretch(self):
        D, _ = self.axes()
        return D[0] / D[1]

    def readout(self):
        return (f"σ {self.sigma:.3f}, path/random {self.path_ratio():.2f}, "
                f"stretch {self.stretch():.1f}×")

    def size(self):
        return self.sigma * self.axes()[0][0]


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CEM, C_CMA = "#eb6834", "#1baf7a"
C_CAND, C_WIN, C_PATH = "#888780", "#2a78d6", "#6250d6"


class App:
    def __init__(self, seed=None):
        self.seed = seed
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK2,
            "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.fig = plt.figure(figsize=(13, 9.5))
        self.fig.canvas.manager.set_window_title("CEM vs. CMA-ES")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.6, 1], left=0.06, right=0.98,
                                   top=0.87, bottom=0.17, hspace=0.36, wspace=0.18)
        self.axA = self.fig.add_subplot(gs[0, 0])
        self.axB = self.fig.add_subplot(gs[0, 1])
        self.axL = self.fig.add_subplot(gs[1, 0])
        self.axS = self.fig.add_subplot(gs[1, 1])
        self.gen_text = self.fig.text(0.06, 0.965, "", fontsize=13, va="center")
        # close-up of CEM's crowd, which becomes too small to see in the main panel
        self.zoom = self.axA.inset_axes([0.60, 0.03, 0.37, 0.37])
        self.fig.legend(handles=[Line2D([], [], color=C_CEM, lw=2, label="CEM"),
                                 Line2D([], [], color=C_CMA, lw=2, label="CMA-ES")],
                        loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.98, 0.985))

        # contours (drawn once)
        g = np.linspace(-6, 6, 300)
        GX, GY = np.meshgrid(g, g)
        Z = f(np.stack([GX, GY], -1))
        for ax in (self.axA, self.axB):
            ax.contour(GX, GY, Z, levels=[0.3, 2, 6, 14, 28, 50, 85],
                       colors=GRID, linewidths=0.9)
            ax.plot(*OPT, "o", ms=9, mfc="none", mec=INK, mew=1.2, zorder=6)
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

        # buttons
        bx = [0.06, 0.20, 0.30]
        self.b_next = Button(self.fig.add_axes([bx[0], 0.035, 0.13, 0.05]), "Next generation")
        self.b_run = Button(self.fig.add_axes([bx[1], 0.035, 0.09, 0.05]), "Run 20")
        self.b_reset = Button(self.fig.add_axes([bx[2], 0.035, 0.09, 0.05]), "Reset")
        self.b_next.on_clicked(lambda e: self.step())
        self.b_run.on_clicked(lambda e: self.run(20))
        self.b_reset.on_clicked(lambda e: self.reset())
        self.fig.text(0.41, 0.06,
                      "gray dot = candidate     blue dot = winner (top 5 of 10)     open circle = optimum\n"
                      "ellipse = 2× search spread     purple arrow = CMA-ES memory path p_c\n"
                      "keys: space = next generation, r = reset",
                      fontsize=9, color=INK2, va="center", linespacing=1.6)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.timer = self.fig.canvas.new_timer(interval=160)
        self.timer.add_callback(self._tick)
        self.remaining = 0
        self.dynamic = []
        self.reset()

    # ---- controls --------------------------------------------------------
    def reset(self):
        self.remaining = 0
        self.timer.stop()
        ss = np.random.SeedSequence(self.seed)
        sa, sb = ss.spawn(2)
        self.cem = CEM(np.random.default_rng(sa))
        self.cma = CMAES(np.random.default_rng(sb))
        self.gen = 0
        self.hist = dict(fA=[f(START)], fB=[f(START)],
                         sA=[self.cem.size()], sB=[self.cma.size()])
        self.draw()

    def step(self):
        self.cem.step()
        self.cma.step()
        self.gen += 1
        self.hist["fA"].append(f(self.cem.m)); self.hist["fB"].append(f(self.cma.m))
        self.hist["sA"].append(self.cem.size()); self.hist["sB"].append(self.cma.size())
        self.draw()

    def run(self, k):
        if self.remaining > 0:
            return
        self.remaining = k
        self.timer.start()

    def _tick(self):
        if self.remaining <= 0:
            self.timer.stop()
            return
        self.step()
        self.remaining -= 1
        if self.remaining <= 0:
            self.timer.stop()

    def on_key(self, event):
        if event.key == " ":
            self.step()
        elif event.key == "r":
            self.reset()

    # ---- drawing ---------------------------------------------------------
    def _panel(self, ax, opt, color, title):
        tr = np.array(opt.trail)
        self.dynamic += ax.plot(tr[:, 0], tr[:, 1], "--", color=color, lw=1.2, zorder=3)
        if len(opt.X):
            win = opt.order[:MU]
            lose = opt.order[MU:]
            self.dynamic.append(ax.scatter(*opt.X[lose].T, s=26, c=C_CAND, zorder=4,
                                           edgecolors=SURFACE, linewidths=1))
            self.dynamic.append(ax.scatter(*opt.X[win].T, s=30, c=C_WIN, zorder=5,
                                           edgecolors=SURFACE, linewidths=1))
        w, h, ang = opt.ellipse()
        e = Ellipse(opt.m, w, h, angle=ang, fill=False, ec=color, lw=1.8, zorder=6)
        ax.add_patch(e); self.dynamic.append(e)
        self.dynamic += ax.plot(*opt.m, "o", ms=8, color=color, zorder=7)
        ax.set_title(f"{title}    loss at center {self._fmt(f(opt.m))}\n{opt.readout()}",
                     loc="left", fontsize=10.5)

    def _zoom(self):
        """Close-up of CEM's crowd, re-centered and re-scaled every generation."""
        z, o = self.zoom, self.cem
        z.cla()
        z.set_facecolor(SURFACE)
        r = max(4 * o.sd.max(), 0.004)
        z.set_xlim(o.m[0] - r, o.m[0] + r); z.set_ylim(o.m[1] - r, o.m[1] + r)
        z.set_aspect("equal"); z.set_xticks([]); z.set_yticks([])
        for sp in z.spines.values():
            sp.set_color(INK2); sp.set_linewidth(0.8)
        if len(o.X):
            z.scatter(*o.X[o.order[MU:]].T, s=18, c=C_CAND, edgecolors=SURFACE, linewidths=0.8)
            z.scatter(*o.X[o.order[:MU]].T, s=22, c=C_WIN, edgecolors=SURFACE, linewidths=0.8)
        w, h, ang = o.ellipse()
        z.add_patch(Ellipse(o.m, w, h, angle=ang, fill=False, ec=C_CEM, lw=1.5))
        z.plot(*o.m, "+", ms=9, mew=1.5, color=C_CEM)
        z.set_title(f"zoom: box width {2 * r:.3g}", fontsize=8.5, color=INK2, pad=3)

    @staticmethod
    def _fmt(v):
        return f"{v:.1e}" if v < 1e-3 else (f"{v:.3f}" if v < 10 else f"{v:.1f}")

    def draw(self):
        for a in self.dynamic:
            a.remove()
        self.dynamic = []
        self._panel(self.axA, self.cem, C_CEM, "CEM")
        self._panel(self.axB, self.cma, C_CMA, "CMA-ES")
        self._zoom()
        # memory path arrow for CMA-ES
        L = self.cma.sigma * np.linalg.norm(self.cma.pc)
        if L > 0.1:
            vec = self.cma.sigma * self.cma.pc * min(1.0, 3.4 / L)
            arr = self.axB.annotate("", xy=self.cma.m + vec, xytext=self.cma.m,
                                    arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=2),
                                    zorder=8)
            self.dynamic.append(arr)

        g = np.arange(len(self.hist["fA"]))
        for ax, keyA, keyB, title, ylab in [
            (self.axL, "fA", "fB", "Loss at the center", "loss (log scale)"),
            (self.axS, "sA", "sB", "Size of the search (largest ellipse radius)", "size (log scale)"),
        ]:
            ax.cla()
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.plot(g, self.hist[keyA], color=C_CEM, lw=2, label="CEM")
            ax.plot(g, self.hist[keyB], color=C_CMA, lw=2, label="CMA-ES")
            ax.set_yscale("log")
            ax.set_xlim(0, max(20, len(g)))
            ax.set_title(title, loc="left")
            ax.set_xlabel("generation"); ax.set_ylabel(ylab)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.gen_text.set_text(f"CEM vs. CMA-ES on the same tilted valley    generation {self.gen}")
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    p.add_argument("--save", type=str, default=None,
                   help="run --gens generations headlessly and save a PNG instead of opening a window")
    p.add_argument("--gens", type=int, default=20)
    a = p.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(seed=a.seed)
    if a.save:
        for _ in range(a.gens):
            app.step()
        app.fig.savefig(a.save, dpi=130)
        print(f"Saved {a.save}  (CEM loss {f(app.cem.m):.3g}, CMA-ES loss {f(app.cma.m):.3g})")
    else:
        plt.show()


if __name__ == "__main__":
    main()