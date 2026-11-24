"""
Frozen vs. plastic network: watch the weights change in real time.

Two agents repeatedly choose to go LEFT or RIGHT for food.
Both start with the same weights, already tuned for "food is on the left".

    Frozen agent   weights never change.
    Plastic agent  after every choice, a local reward-modulated rule updates
                   only the connection it just used:

                       delta_w(chosen) = eta * (reward - 0.5)

                   food found   (reward 1) -> strengthen that connection
                   no food      (reward 0) -> weaken it

Each agent is a tiny network: one input neuron ("hungry", always 1) connected
to two output neurons ("go left", "go right"). The probability of going right
is sigmoid(beta * (w_right - w_left)).

What you see
    Top row      the two networks, live
                   line thickness = size of the weight, color = sign
                   (aqua = positive, orange = negative)
                   bars = probability of choosing each side
                   ring  = side chosen this step (with food or empty)
    Middle left  weight history: solid = plastic agent, dashed = frozen agent
    Middle right success rate over the last 30 steps
    Shaded bands show which side the food is on.

Controls
    Play / Pause, Step (one step), Move food, Reset
    eta slider     learning rate of the rule (0 = plastic agent becomes frozen)
    speed slider   steps per animation frame
    Keyboard       space = play/pause, m = move food, s = step, r = reset

Run:   python plasticity_foraging_app.py
       python plasticity_foraging_app.py --seed 3        (repeatable)
       python plasticity_foraging_app.py --save out.png  (headless screenshot)
Needs: numpy, matplotlib (with a GUI backend such as TkAgg or QtAgg)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, Slider

# ---------------------------------------------------------------------------
# Settings (same as the in-chat demo)
# ---------------------------------------------------------------------------
W_LEFT0, W_RIGHT0 = 1.2, 0.0      # starting weights: tuned for "food on the left"
BETA = 3.0                        # how decisively the agent follows its weights
W_CLIP = 3.0                      # weights are kept inside [-3, 3]
WIN = 30                          # window for the success rate
HIST = 300                        # how many steps the plots show

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_POS, C_NEG = "#1baf7a", "#eb6834"          # positive / negative weight
C_WL, C_WR = "#2a78d6", "#e87ba4"            # w_left / w_right series
C_FROZEN, C_PLASTIC = "#888780", "#1baf7a"
BAND_L, BAND_R = "#eeedfe", "#fdf1dc"        # food on left / right
C_HIT, C_MISS = "#1baf7a", "#e34948"


# ---------------------------------------------------------------------------
# The agent: two weights, a choice rule, and (optionally) a plasticity rule
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, plastic):
        self.plastic = plastic
        self.w = {"L": W_LEFT0, "R": W_RIGHT0}
        self.results = []                       # last WIN rewards
        self.last = None                        # (choice, reward, delta_w)

    def p_right(self):
        return 1 / (1 + np.exp(-BETA * (self.w["R"] - self.w["L"])))

    def act(self, rng, food, eta):
        choice = "R" if rng.random() < self.p_right() else "L"
        reward = 1 if choice == food else 0
        dw = 0.0
        if self.plastic:                                     # the local rule
            dw = eta * (reward - 0.5)
            self.w[choice] = float(np.clip(self.w[choice] + dw, -W_CLIP, W_CLIP))
        self.results.append(reward)
        if len(self.results) > WIN:
            self.results.pop(0)
        self.last = (choice, reward, dw)

    def success(self):
        return np.mean(self.results) if self.results else np.nan


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, seed=None):
        self.seed = seed
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK2,
            "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title("Frozen vs. plastic network")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.15, 1], left=0.06, right=0.98,
                                   top=0.91, bottom=0.2, hspace=0.32, wspace=0.16)
        self.axF = self.fig.add_subplot(gs[0, 0])
        self.axP = self.fig.add_subplot(gs[0, 1])
        self.axW = self.fig.add_subplot(gs[1, 0])
        self.axS = self.fig.add_subplot(gs[1, 1])
        self.title = self.fig.text(0.06, 0.965, "", fontsize=13, va="center")

        self.net_F = self._build_network(self.axF)
        self.net_P = self._build_network(self.axP)
        self._build_history_axes()
        self._build_controls()

        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self._tick)
        self.playing = False
        self.reset()

    # ---- network panel ---------------------------------------------------
    def _build_network(self, ax):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect("equal"); ax.axis("off")
        nodes = {"in": (1.6, 5.0), "L": (5.6, 7.6), "R": (5.6, 2.4)}
        net = {"nodes": nodes}
        # edges (drawn under nodes)
        net["edge"] = {}
        net["wlabel"] = {}
        for k in ("L", "R"):
            (x0, y0), (x1, y1) = nodes["in"], nodes[k]
            line, = ax.plot([x0, x1], [y0, y1], color=C_POS, lw=3, solid_capstyle="round", zorder=1)
            net["edge"][k] = line
            # label sits clear of the line: above the upper edge, below the lower edge
            ly = 8.5 if k == "L" else 1.5
            net["wlabel"][k] = ax.text(2.7, ly, "", ha="center", va="center", fontsize=11,
                                       fontweight="bold", zorder=5)
        # nodes
        for k, (x, y) in nodes.items():
            ax.add_patch(Circle((x, y), 0.62, fc=SURFACE, ec=INK2, lw=1.2, zorder=3))
        ax.text(*nodes["in"], "1", ha="center", va="center", fontsize=11, zorder=4)
        ax.text(nodes["in"][0] - 0.85, nodes["in"][1], "input\n(hungry)", ha="right", va="center",
                fontsize=9, color=INK2)
        ax.text(*nodes["L"], "L", ha="center", va="center", fontsize=11, zorder=4)
        ax.text(*nodes["R"], "R", ha="center", va="center", fontsize=11, zorder=4)
        # choice ring (moves to the chosen node)
        net["ring"] = Circle(nodes["L"], 0.9, fill=False, lw=2.5, ec=C_HIT, zorder=2)
        ax.add_patch(net["ring"])
        # probability bars
        net["bar"], net["ptext"] = {}, {}
        for k in ("L", "R"):
            x, y = nodes[k]
            ax.add_patch(FancyBboxPatch((6.6, y - 0.32), 2.4, 0.64, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=GRID, ec="none", zorder=1))
            bar = FancyBboxPatch((6.6, y - 0.32), 1.2, 0.64, boxstyle="round,pad=0,rounding_size=0.12",
                                 fc=C_WL if k == "L" else C_WR, ec="none", zorder=2)
            ax.add_patch(bar)
            net["bar"][k] = bar
            net["ptext"][k] = ax.text(6.6, y + 0.62, "", fontsize=9, color=INK2, va="bottom")
        # food marker
        net["food"] = ax.text(9.35, 7.6, "food", ha="left", va="center", fontsize=10,
                              color=INK, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.3", fc="#fdf1dc", ec="#c98500", lw=1))
        net["status"] = ax.text(0.2, 0.3, "", fontsize=10, color=INK2, va="bottom")
        return net

    def _draw_network(self, ax, net, agent, title):
        for k in ("L", "R"):
            w = agent.w[k]
            if abs(w) < 0.005:
                w = 0.0                       # avoid showing "-0.00"
            col = C_POS if w >= 0 else C_NEG
            net["edge"][k].set_color(col)
            net["edge"][k].set_linewidth(0.6 + 3.2 * abs(w))
            net["edge"][k].set_alpha(0.3 if abs(w) < 0.05 else 1.0)
            net["wlabel"][k].set_text(f"w_{'left' if k == 'L' else 'right'} = {w:+.2f}")
            net["wlabel"][k].set_color(col)
        pR = agent.p_right()
        for k, p in (("L", 1 - pR), ("R", pR)):
            net["bar"][k].set_width(max(2.4 * p, 0.001))
            net["ptext"][k].set_text(f"P(go {'left' if k == 'L' else 'right'}) = {p:.0%}")
        fy = net["nodes"][self.food][1]
        net["food"].set_position((9.35, fy))
        if agent.last:
            choice, reward, dw = agent.last
            net["ring"].center = net["nodes"][choice]
            net["ring"].set_edgecolor(C_HIT if reward else C_MISS)
            net["ring"].set_visible(True)
            side = "left" if choice == "L" else "right"
            msg = f"last step: went {side}, {'found food' if reward else 'empty'}"
            if agent.plastic:
                msg += f"  →  Δw_{side} = {dw:+.3f}"
            net["status"].set_text(msg)
        else:
            net["ring"].set_visible(False)
            net["status"].set_text("press Play or Step")
        s = agent.success()
        s_txt = "–" if np.isnan(s) else f"{s:.0%}"
        ax.set_title(f"{title}\nsuccess over last {WIN} steps: {s_txt}", loc="left", fontsize=11)

    # ---- history plots ---------------------------------------------------
    def _build_history_axes(self):
        ax = self.axW
        self.lines = {
            ("P", "L"): ax.plot([], [], color=C_WL, lw=2.2, label="w_left (plastic)")[0],
            ("P", "R"): ax.plot([], [], color=C_WR, lw=2.2, label="w_right (plastic)")[0],
            ("F", "L"): ax.plot([], [], color=C_WL, lw=1.5, ls="--", label="w_left (frozen)")[0],
            ("F", "R"): ax.plot([], [], color=C_WR, lw=1.5, ls="--", label="w_right (frozen)")[0],
        }
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-W_CLIP - 0.2, W_CLIP + 0.2)
        ax.set_title("Weights over time (solid = plastic, dashed = frozen)", loc="left")
        ax.set_xlabel("step"); ax.set_ylabel("weight")
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9)
        ax = self.axS
        self.s_lines = {
            "F": ax.plot([], [], color=C_FROZEN, lw=2.2, label="frozen")[0],
            "P": ax.plot([], [], color=C_PLASTIC, lw=2.2, label="plastic")[0],
        }
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"Success rate (last {WIN} steps)", loc="left")
        ax.set_xlabel("step")
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        for a in (self.axW, self.axS):
            a.grid(True, color=GRID, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        self.bands = []
        self.fig.text(0.06, 0.118,
                      "Shaded background:  lavender = food on left,  cream = food on right",
                      fontsize=9, color=INK2)

    def _draw_history(self):
        t = np.array(self.h["t"])
        if len(t) == 0:
            t0, t1 = 0, HIST
        else:
            t0, t1 = max(0, t[-1] - HIST + 1), max(HIST, t[-1] + 1)
        for (who, k), line in self.lines.items():
            line.set_data(t, self.h[f"{who}{k}"])
        for who, line in self.s_lines.items():
            line.set_data(t, self.h[f"s{who}"])
        for b in self.bands:
            b.remove()
        self.bands = []
        # food-side bands
        segs, start = [], 0
        foods = self.h["food"]
        for i in range(1, len(foods) + 1):
            if i == len(foods) or foods[i] != foods[start]:
                segs.append((t[start] - 0.5, t[i - 1] + 0.5, foods[start]))
                start = i
        for ax in (self.axW, self.axS):
            ax.set_xlim(t0, t1)
            for a, b, side in segs:
                if b >= t0:
                    self.bands.append(ax.axvspan(a, b, color=BAND_L if side == "L" else BAND_R,
                                                 zorder=0, lw=0))

    # ---- controls --------------------------------------------------------
    def _build_controls(self):
        y = 0.045
        self.b_play = Button(self.fig.add_axes([0.06, y, 0.08, 0.05]), "Play")
        self.b_step = Button(self.fig.add_axes([0.15, y, 0.07, 0.05]), "Step")
        self.b_food = Button(self.fig.add_axes([0.23, y, 0.11, 0.05]), "Move food")
        self.b_reset = Button(self.fig.add_axes([0.35, y, 0.07, 0.05]), "Reset")
        self.b_play.on_clicked(lambda e: self.toggle_play())
        self.b_step.on_clicked(lambda e: self.step())
        self.b_food.on_clicked(lambda e: self.move_food())
        self.b_reset.on_clicked(lambda e: self.reset())
        self.s_eta = Slider(self.fig.add_axes([0.53, 0.085, 0.25, 0.025]), "rule learning rate η",
                            0.0, 0.5, valinit=0.15, valstep=0.01, color=C_PLASTIC)
        self.s_speed = Slider(self.fig.add_axes([0.53, 0.04, 0.25, 0.025]), "steps per frame",
                              1, 10, valinit=1, valstep=1, color=C_FROZEN)
        self.s_eta.on_changed(lambda v: self.draw())
        self.fig.text(0.81, 0.062, "keys: space = play/pause\nm = move food, s = step, r = reset",
                      fontsize=9, color=INK2, va="center", linespacing=1.6)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def move_food(self):
        self.food = "R" if self.food == "L" else "L"
        self.draw()

    def on_key(self, e):
        {" ": self.toggle_play, "m": self.move_food, "s": self.step,
         "r": self.reset}.get(e.key, lambda: None)()

    # ---- simulation ------------------------------------------------------
    def reset(self):
        if self.playing:
            self.toggle_play()
        self.rng = np.random.default_rng(self.seed)
        self.food = "L"
        self.frozen = Agent(plastic=False)
        self.plastic = Agent(plastic=True)
        self.t = 0
        self.h = {k: [] for k in ("t", "FL", "FR", "PL", "PR", "sF", "sP", "food")}
        self.draw()

    def _advance(self):
        eta = self.s_eta.val
        self.frozen.act(self.rng, self.food, eta)
        self.plastic.act(self.rng, self.food, eta)
        self.t += 1
        h = self.h
        h["t"].append(self.t); h["food"].append(self.food)
        h["FL"].append(self.frozen.w["L"]); h["FR"].append(self.frozen.w["R"])
        h["PL"].append(self.plastic.w["L"]); h["PR"].append(self.plastic.w["R"])
        h["sF"].append(self.frozen.success()); h["sP"].append(self.plastic.success())
        if len(h["t"]) > HIST:
            for k in h:
                h[k].pop(0)

    def step(self):
        self._advance()
        self.draw()

    def _tick(self):
        for _ in range(int(self.s_speed.val)):
            self._advance()
        self.draw()

    def draw(self):
        self._draw_network(self.axF, self.net_F, self.frozen, "Frozen agent (weights fixed)")
        self._draw_network(self.axP, self.net_P, self.plastic,
                           f"Plastic agent (rule: Δw = η·(reward − 0.5), η = {self.s_eta.val:.2f})")
        self._draw_history()
        side = "LEFT" if self.food == "L" else "RIGHT"
        self.title.set_text(f"Frozen vs. plastic network    step {self.t}    food is on the {side}")
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description="Frozen vs. plastic network, live weights")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", type=str, default=None,
                   help="simulate (80 steps, food moves at step 40) and save a PNG instead of opening a window")
    a = p.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(seed=a.seed)
    if a.save:
        for i in range(120):
            if i == 50:
                app.move_food()
            app._advance()
        app.draw()
        app.fig.savefig(a.save, dpi=120)
        print(f"Saved {a.save}  frozen weights {app.frozen.w}  plastic weights "
              f"{ {k: round(v, 2) for k, v in app.plastic.w.items()} }")
    else:
        plt.show()


if __name__ == "__main__":
    main()