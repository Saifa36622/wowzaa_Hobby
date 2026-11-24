"""
Normal network vs SFNN: what happens when you change the number of inputs or actions?

Left   NORMAL network: every link has its OWN weight (each link a different colour).
       More sensors or actions = more weights to store and train.
Middle SFNN: links and neurons come in only 3 TYPES each (3 colours), and every member of a
       type uses the SAME numbers from the genome. More sensors or actions = more neurons,
       but the genome stays the same size.
Right  Number of stored numbers vs number of sensors, and a zoom into ONE SFNN neuron
       (this is where the "4 x 4 = 16" comes from: it is inside a neuron, not a limit).

"Train at this size" pretends the networks were trained at the current size. Then move the
sliders: the normal network shows in red the links that have NO trained weight (it must be
retrained); SFNN just reuses the same genome.

Controls
    sliders: sensors (inputs), hidden neurons, actions (outputs), values per neuron (D)
    presets: CartPole (4 in, 2 out), Acrobot (6 in, 3 out), MountainCar (2 in, 3 out), big robot
    click any neuron in either network to see what it uses
Run:   python sfnn_vs_normal_scaling.py
       python sfnn_vs_normal_scaling.py --save shot.png        (headless screenshot)
Needs: numpy, matplotlib (GUI backend such as TkAgg or QtAgg)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.widgets import Button, Slider

SURFACE, INK, INK2, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c7c0", "#e4e3df"
GREEN, BLUE, ORANGE, RED, PURPLE = "#1baf7a", "#2a78d6", "#eb6834", "#e34948", "#6250d6"
TEAL, PINK, GOLD = "#0f8f8a", "#d6559a", "#b8860b"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fbe3d6", "#f7f7f5", "#d3e5f7", "#2a78d6"])
TYPE_COL = {"in": GREEN, "hid": BLUE, "out": ORANGE}
LINK_COL = {"ih": TEAL, "hh": PURPLE, "ho": PINK}
TYPE_NAME = {"in": "input", "hid": "hidden", "out": "output"}
PRESETS = [("CartPole\n4 in, 2 out", 4, 2), ("Acrobot\n6 in, 3 out", 6, 3),
           ("MountainCar\n2 in, 3 out", 2, 3), ("big robot\n30 in, 12 out", 30, 12)]


def normal_count(I, H, O):
    return I * H + H + H * O + O                          # weights + biases, one hidden layer


def sfnn_blocks(D):
    neuron = D * D + D
    rule = 3 * (D * (2 * D + 1) + D * D + D)
    return [("input neuron type", neuron, GREEN), ("hidden neuron type", neuron, BLUE),
            ("output neuron type", neuron, ORANGE), ("rule input→hidden", rule, TEAL),
            ("rule hidden→hidden", rule, PURPLE), ("rule hidden→output", rule, PINK),
            ("learning rate", 1, GOLD)]


def sfnn_count(D):
    return sum(n for _, n, _ in sfnn_blocks(D))


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
        self.I, self.H, self.O, self.D = 4, 8, 2, 4
        self.trained = None                       # (I, O) the networks were "trained" at
        self.sel = ("sfnn", "hid", 0)
        self.fig = plt.figure(figsize=size)
        try:
            self.fig.canvas.manager.set_window_title("Normal network vs SFNN: scaling inputs and actions")
        except AttributeError:
            pass
        f = self.fig
        self.axN = f.add_axes([0.025, 0.43, 0.31, 0.40]); self.axS = f.add_axes([0.355, 0.43, 0.31, 0.40])
        self.axNs = f.add_axes([0.025, 0.215, 0.31, 0.15]); self.axSs = f.add_axes([0.355, 0.29, 0.31, 0.05])
        self.axSt = f.add_axes([0.355, 0.20, 0.31, 0.075])
        self.axC = f.add_axes([0.725, 0.565, 0.255, 0.265]); self.axZ = f.add_axes([0.695, 0.20, 0.295, 0.29])
        self.title = f.text(0.025, 0.955, "", va="center", fontweight="bold")
        self.sub = f.text(0.025, 0.925, "", va="center", color=INK2)
        self._controls()
        f.canvas.mpl_connect("pick_event", self._pick)
        f.canvas.mpl_connect("resize_event", lambda e: self.draw())
        self.draw()

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig; c = "#ecebe7"
        self.sl = {}
        for k, (lab, lo, hi, v, y) in {"I": ("sensors (inputs)", 1, 40, self.I, 0.135),
                                        "H": ("hidden neurons", 1, 48, self.H, 0.1),
                                        "O": ("actions (outputs)", 1, 20, self.O, 0.065),
                                        "D": ("values per neuron D", 1, 8, self.D, 0.03)}.items():
            s = Slider(f.add_axes([0.12, y, 0.25, 0.022]), lab, lo, hi, valinit=v, valstep=1,
                       color=BLUE if k != "D" else GOLD)
            s.on_changed(lambda v, k=k: self._set(k, int(v)))
            self.sl[k] = s
        self.btns = []
        for i, (lab, I, O) in enumerate(PRESETS):
            b = Button(f.add_axes([0.43 + i * 0.083, 0.07, 0.078, 0.075]), lab, color=c, hovercolor="#dcdad3")
            b.on_clicked(lambda e, I=I, O=O: self._preset(I, O))
            self.btns.append(b)
        b = Button(f.add_axes([0.775, 0.07, 0.1, 0.075]), "Train at\nthis size", color="#fff4cc", hovercolor="#f5e6a8")
        b.on_clicked(lambda e: self._train()); self.btns.append(b)
        b = Button(f.add_axes([0.88, 0.07, 0.1, 0.075]), "Forget\ntraining", color=c, hovercolor="#dcdad3")
        b.on_clicked(lambda e: self._forget()); self.btns.append(b)
        self.hint = f.text(0.43, 0.025, "", color=INK2, va="bottom")

    def _set(self, k, v):
        setattr(self, k, v)
        if self.sel[1] == "in" and self.sel[2] >= self.I or self.sel[1] == "out" and self.sel[2] >= self.O \
                or self.sel[1] == "hid" and self.sel[2] >= self.H:
            self.sel = (self.sel[0], self.sel[1], 0)
        self.draw()

    def _preset(self, I, O):
        for k, v in (("I", I), ("O", O)):
            self.sl[k].eventson = False; self.sl[k].set_val(v); self.sl[k].eventson = True
        self.I, self.O = I, O; self._set("I", I)

    def _train(self):
        self.trained = (self.I, self.O); self.draw()

    def _forget(self):
        self.trained = None; self.draw()

    def _pick(self, e):
        for (panel, layer), art in self.node_artists.items():
            if e.artist is art:
                self.sel = (panel, layer, int(e.ind[0])); self.draw(); return

    # ---------------- helpers ----------------
    def _fs(self):
        w, h = self.fig.get_size_inches()
        return float(np.clip(min(w / 1.75, h) * 1.1, 7.5, 14))

    @staticmethod
    def _blank(ax):
        ax.cla(); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def _pos(self, n, x):
        ys = np.linspace(0.9, 0.1, n) if n > 1 else np.array([0.5])
        return np.column_stack([np.full(n, x), ys])

    def _node_size(self):
        m = max(self.I, self.H, self.O)
        return float(np.clip(5200 / m, 18, 330)) * (self.fs / 10) ** 2

    # ---------------- drawing ----------------
    def draw(self):
        fs = self.fs = self._fs()
        I, H, O, D = self.I, self.H, self.O, self.D
        nN, nS = normal_count(I, H, O), sfnn_count(D)
        self.title.set_text("Normal network vs SFNN: what has to be stored when the task gets more sensors or actions?")
        self.title.set_fontsize(fs * 1.25)
        self.sub.set_text(f"Now: {I} sensors, {H} hidden, {O} actions.   Normal network stores {nN:,} numbers.   "
                          f"SFNN genome stores {nS:,} numbers (depends only on D = {D}).   Click any neuron.")
        self.sub.set_fontsize(fs)
        for b in self.btns:
            b.label.set_fontsize(fs * 0.82)
        for s in self.sl.values():
            s.label.set_fontsize(fs * 0.85); s.valtext.set_fontsize(fs * 0.85)
        if self.trained:
            tI, tO = self.trained
            self.hint.set_text(f"trained at {tI} sensors, {tO} actions: move the sliders and compare")
        else:
            self.hint.set_text("tip: press 'Train at this size', then change sensors or actions")
        self.hint.set_fontsize(fs * 0.85)
        self.node_artists = {}
        self._draw_normal(fs); self._draw_sfnn(fs); self._draw_normal_store(fs)
        self._draw_genome(fs); self._draw_chart(fs); self._draw_zoom(fs)
        self.fig.canvas.draw_idle()

    def _layers(self, ax, panel, fs, colors):
        I, H, O = self.I, self.H, self.O
        P = {"in": self._pos(I, 0.06), "hid": self._pos(H, 0.5), "out": self._pos(O, 0.94)}
        s = self._node_size()
        for layer, pts in P.items():
            ec = colors[layer]
            sel = self.sel if self.sel[0] == panel else None
            lw = np.full(len(pts), 1.5)
            if sel and sel[1] == layer:
                lw[sel[2]] = 4
            art = ax.scatter(pts[:, 0], pts[:, 1], s=s, c=[SURFACE] * len(pts) if panel == "normal" else [ec] * len(pts),
                             edgecolors=[ec] * len(pts) if not isinstance(ec, list) else ec, linewidths=lw,
                             zorder=4, picker=6)
            self.node_artists[(panel, layer)] = art
            if sel and sel[1] == layer:
                ax.scatter(*pts[sel[2]], s=s * 2.2, facecolors="none", edgecolors=INK, linewidths=2, zorder=5)
        return P

    def _draw_normal(self, fs):
        ax = self.axN; self._blank(ax); ax.set_xlim(-0.04, 1.04); ax.set_ylim(0.0, 1.06)
        I, H, O = self.I, self.H, self.O
        P = self._layers(ax, "normal", fs, {"in": INK2, "hid": INK2, "out": INK2})
        segs, cols, miss = [], [], []
        tI, tO = self.trained if self.trained else (I, O)
        k = 0
        sel = self.sel if self.sel[0] == "normal" else None
        for a, b, la, lb in (("in", "hid", I, H), ("hid", "out", H, O)):
            for i in range(la):
                for j in range(lb):
                    seg = [P[a][i], P[b][j]]
                    new = (a == "in" and i >= tI) or (b == "out" and j >= tO)
                    if new:
                        miss.append(seg)
                    else:
                        segs.append(seg)
                        hi = sel is not None and ((sel[1] == a and sel[2] == i) or (sel[1] == b and sel[2] == j))
                        col = hsv_to_rgb([(k * 0.618) % 1, 0.65, 0.85])
                        cols.append(list(col) + [0.95 if hi or sel is None else 0.18])
                    k += 1
        lw = float(np.clip(30 / np.sqrt(max(1, len(segs) + len(miss))), 0.3, 1.8))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=1))
        if miss:
            ax.add_collection(LineCollection(miss, colors=RED, linewidths=lw * 1.3, linestyles="--", zorder=2))
        n = normal_count(I, H, O)
        ax.set_title(f"NORMAL network: every link has its own weight\n({I}×{H} + {H}×{O} weights + {H + O} biases "
                     f"= {n:,} numbers)", loc="left", fontsize=fs * 1.02)
        for x, t in ((0.06, "sensors"), (0.5, "hidden"), (0.94, "actions")):
            ax.text(x, 1.0, t, ha="center", fontsize=fs * 0.85, color=INK2)
        if self.trained and miss:
            ax.text(0.5, 0.02, f"{len(miss)} red links have NO trained weight → must retrain",
                    ha="center", color=RED, fontsize=fs * 0.95, fontweight="bold")
        elif self.trained and (I < tI or O < tO):
            ax.text(0.5, 0.02, f"trained for {tI} sensors/{tO} actions: some trained weights now have nothing to "
                    f"connect to", ha="center", color=RED, fontsize=fs * 0.85)
        elif self.trained:
            ax.text(0.5, 0.02, "same size as trained: works", ha="center", color=GREEN, fontsize=fs * 0.95)

    def _draw_sfnn(self, fs):
        ax = self.axS; self._blank(ax); ax.set_xlim(-0.04, 1.04); ax.set_ylim(0.0, 1.06)
        I, H, O = self.I, self.H, self.O
        P = self._layers(ax, "sfnn", fs, TYPE_COL)
        rng = np.random.default_rng(I * 1000 + H * 31 + O)
        sel = self.sel if self.sel[0] == "sfnn" else None
        for a, b, la, lb, t in (("in", "hid", I, H, "ih"), ("hid", "out", H, O, "ho")):
            M = rng.random((la, lb)) < 0.5
            segs = [[P[a][i], P[b][j]] for i in range(la) for j in range(lb) if M[i, j]]
            alpha = 0.8 if sel is None or sel[1] in (a, b) else 0.25
            lw = float(np.clip(30 / np.sqrt(max(1, I * H + H * O)), 0.3, 1.8))
            ax.add_collection(LineCollection(segs, colors=LINK_COL[t], linewidths=lw, alpha=alpha, zorder=1))
        ax.set_title(f"SFNN: 3 neuron types + 3 link types, numbers SHARED\n(random half of the links; hidden→hidden links exist "
                     f"too, not drawn)", loc="left", fontsize=fs * 1.02)
        for x, t in ((0.06, "sensors"), (0.5, "hidden"), (0.94, "actions")):
            ax.text(x, 1.0, t, ha="center", fontsize=fs * 0.85, color=INK2)
        if self.trained:
            tI, tO = self.trained
            msg = "same genome works: just add/remove neurons" if (I, O) != (tI, tO) else "same size as trained: works"
            ax.text(0.5, 0.02, msg, ha="center", color=GREEN, fontsize=fs * 0.95, fontweight="bold")

    def _draw_normal_store(self, fs):
        ax = self.axNs; self._blank(ax)
        I, H, O = self.I, self.H, self.O
        tI, tO = self.trained if self.trained else (I, O)
        rng = np.random.default_rng(7)
        W1 = rng.normal(size=(I, H)); W2 = rng.normal(size=(H, O))
        a1 = ax.inset_axes([0.0, 0.0, 0.62, 0.6]); a2 = ax.inset_axes([0.7, 0.0, 0.3, 0.6])
        for a, W, ttl in ((a1, W1, f"sensor→hidden weights ({I}×{H})"),
                          (a2, W2.T, f"hidden→action ({O}×{H})")):
            a.imshow(W, cmap=DIVERGE, vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
            a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=fs * 0.8, color=INK2)
        if I > tI:
            a1.add_patch(Rectangle((-0.5, tI - 0.5), H, I - tI, fc=RED, alpha=0.45, ec=RED, hatch="//"))
        if O > tO:
            a2.add_patch(Rectangle((-0.5, tO - 0.5), H, O - tO, fc=RED, alpha=0.45, ec=RED, hatch="//"))
        sel = self.sel if self.sel[0] == "normal" else None
        if sel:
            l, i = sel[1], sel[2]
            if l == "in":
                a1.add_patch(Rectangle((-0.5, i - 0.5), H, 1, fill=False, ec=INK, lw=2))
            elif l == "out":
                a2.add_patch(Rectangle((-0.5, i - 0.5), H, 1, fill=False, ec=INK, lw=2))
            else:
                a1.add_patch(Rectangle((i - 0.5, -0.5), 1, I, fill=False, ec=INK, lw=2))
                a2.add_patch(Rectangle((i - 0.5, -0.5), 1, O, fill=False, ec=INK, lw=2))
        ax.text(0, 1.0, "What the normal network stores: every cell is one weight that belongs to ONE link",
                fontsize=fs * 0.88, va="top", transform=ax.transAxes)

    def _draw_genome(self, fs):
        ax = self.axSs; self._blank(ax)
        blocks = sfnn_blocks(self.D); total = sum(n for _, n, _ in blocks)
        x = 0
        sel = self.sel if self.sel[0] == "sfnn" else None
        hi = {"in": 0, "hid": 1, "out": 2}.get(sel[1]) if sel else None
        for k, (name, n, c) in enumerate(blocks):
            w = max(n, total * 0.006)
            ax.add_patch(Rectangle((x, 0), w, 1, fc=c, ec=SURFACE, lw=1, alpha=1 if hi in (None, k) else 0.3))
            if hi == k:
                ax.add_patch(Rectangle((x, 0), w, 1, fill=False, ec=INK, lw=2.5))
            x += w
        ax.set_xlim(0, x); ax.set_ylim(0, 1)
        ax.text(0, 1.15, f"What SFNN stores: the genome, {total:,} numbers in 7 blocks "
                f"(same at any size)", fontsize=fs * 0.88, transform=ax.transAxes)
        ax = self.axSt; self._blank(ax)
        lines = [f"■ {name}: {n}" for name, n, _ in blocks]
        for k, ((name, n, c), t) in enumerate(zip(blocks, lines)):
            col, row = k // 4, k % 4
            ax.text(col * 0.5, 0.95 - row * 0.25, t, color=c, fontsize=fs * 0.8, va="top", transform=ax.transAxes)

    def _draw_chart(self, fs):
        ax = self.axC; ax.cla(); ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        xs = np.arange(1, 101)
        ax.plot(xs, [normal_count(x, self.H, self.O) for x in xs], color=INK2, lw=2.2,
                label=f"normal network ({self.H} hidden, {self.O} actions)")
        ax.plot(xs, [normal_count(x, 4 * self.H, self.O) for x in xs], color=INK2, lw=1.2, ls=":",
                label=f"normal network, {4 * self.H} hidden")
        ax.plot(xs, np.full(len(xs), sfnn_count(self.D)), color=BLUE, lw=2.4, label=f"SFNN genome (D = {self.D})")
        ax.scatter([self.I], [normal_count(self.I, self.H, self.O)], color=INK, zorder=5, s=40)
        ax.scatter([self.I], [sfnn_count(self.D)], color=BLUE, zorder=5, s=40)
        ax.axvline(self.I, color=FAINT, lw=1)
        ax.set_yscale("log"); ax.set_xlim(1, 100)
        ax.set_xlabel("number of sensors", fontsize=fs * 0.85); ax.set_ylabel("numbers stored (log scale)", fontsize=fs * 0.85)
        ax.tick_params(labelsize=fs * 0.75)
        ax.legend(frameon=False, fontsize=fs * 0.75, loc="lower right")
        ax.set_title("Numbers to store vs number of sensors", loc="left", fontsize=fs * 0.95)

    def _draw_zoom(self, fs):
        ax = self.axZ; self._blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        D = self.D
        panel, layer, idx = self.sel
        if panel == "normal":
            I, H, O = self.I, self.H, self.O
            own = {"in": H, "hid": I + O + 1, "out": H + 1}[layer]
            ax.text(0, 0.95, f"NORMAL {TYPE_NAME[layer]} neuron {idx + 1}", fontsize=fs, fontweight="bold", va="top")
            ax.text(0, 0.8, f"It owns {own} numbers that no other neuron uses\n(its links' weights"
                    f"{'' if layer == 'in' else ' and its bias'}; outlined in the tables).\n\n"
                    f"Add one more sensor → {H} brand-new weights.\nAdd one more action → {H + 1} brand-new "
                    f"numbers.\nNone of them are trained yet.", fontsize=fs * 0.9, va="top", linespacing=1.4)
            return
        n_type = {"in": self.I, "hid": self.H, "out": self.O}[layer]
        c = TYPE_COL[layer]
        ax.text(0, 0.98, f"Inside ONE SFNN {TYPE_NAME[layer]} neuron (click others to compare)",
                fontsize=fs, fontweight="bold", va="top")
        rng = np.random.default_rng({"in": 1, "hid": 2, "out": 3}[layer] * 10 + D)
        W = rng.normal(size=(D, D)); b = rng.normal(size=D)
        vin = np.tanh(rng.normal(size=D))
        if layer == "in":                         # an input neuron gets ONE sensor value, copied D times
            vin = np.full(D, np.tanh(rng.normal()))
        vout = np.tanh(W @ vin + b)
        cell = min(0.5 / (D + 3), 0.075); y0 = 0.42
        # incoming values
        def column(x, v, lab):
            for q in range(D):
                ax.add_patch(Rectangle((x, y0 + (D / 2 - q - 1) * cell), cell, cell * 0.92,
                                       fc=DIVERGE(0.5 + 0.5 * v[q]), ec=FAINT, lw=0.6))
            ax.text(x + cell / 2, y0 + D / 2 * cell + 0.03, lab, ha="center", fontsize=fs * 0.75, color=INK2)
        column(0.04, vin, "1 sensor value\n(copied \u00d7" + str(D) + ")" if layer == "in" else f"{D} values in")
        xg = 0.25
        for r in range(D):
            for q in range(D):
                ax.add_patch(Rectangle((xg + q * cell, y0 + (D / 2 - r - 1) * cell), cell, cell * 0.92,
                                       fc=DIVERGE(0.5 + 0.5 * np.tanh(W[r, q])), ec=c, lw=0.8))
        ax.text(xg + D * cell / 2, y0 + D / 2 * cell + 0.03, f"{D}×{D} table = {D * D}", ha="center",
                fontsize=fs * 0.75, color=c)
        xb = xg + D * cell + 0.05
        column(xb, np.tanh(b), f"+{D} biases")
        xo = xb + cell + 0.17
        column(xo, vout, f"{D} values out")
        for x1, x2 in ((0.04 + cell + 0.01, xg - 0.01), (xb + cell + 0.01, xo - 0.01)):
            ax.add_patch(FancyArrowPatch((x1, y0), (x2, y0), arrowstyle="-|>", mutation_scale=fs * 1.2, color=INK2))
        ax.text(0, 0.17, f"This table ({D}×{D}) mixes the neuron's OWN {D} values. It has nothing to do\n"
                f"with how many sensors or actions there are. Its {D * D + D} numbers are the genome's\n"
                f"'{TYPE_NAME[layer]} neuron type' block, and ALL {n_type} {TYPE_NAME[layer]} neurons use the same ones.",
                fontsize=fs * 0.82, va="top", linespacing=1.4)


def main():
    ap = argparse.ArgumentParser(description="Normal network vs SFNN scaling")
    ap.add_argument("--size", type=float, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(tuple(a.size) if a.size else ((16, 9) if a.save else screen_inches()))
    if a.save:
        app._train(); app._preset(6, 3); app.draw()
        app.fig.savefig(a.save, dpi=110); print("saved", a.save); return
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