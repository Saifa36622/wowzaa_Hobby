"""
Spontaneous activity (practice before birth), visualized in real time.

THE ROBOT (kept simple on purpose)
    A light-seeking robot. Each trial, a light appears on its LEFT or RIGHT.
    It has 8 light sensors: 4 face left, 4 face right. But the wires are SHUFFLED,
    so the network is never told which input neuron is which sensor.
    It must turn toward the light. Reward = 1 if it turns the right way.

THE NETWORK (same for both agents)
    8 input neurons -> 2 hidden neurons (A, B) -> 2 output neurons (turn L, turn R)
    Two local rules run all the time:
      1. Wiring rule (input -> hidden), no reward needed:
         the hidden neuron that responds most to the current input pattern
         moves its weights a little toward that pattern ("fire together, wire together",
         with competition so A and B specialize on different patterns).
      2. Reward rule (hidden -> output):
         after a turn, strengthen that turn's connections if the reward beat what
         the agent expected, weaken them if it was worse.

TWO AGENTS, SAME BIRTH WIRING
    With practice:    before entering the world, its input neurons are driven by
                      OU noise for T steps. Only rule 1 can do anything (no reward).
    Without practice: enters the world straight away.
    Then both face the SAME sequence of lights and sensor readings.

THE PRACTICE NOISE (Ornstein-Uhlenbeck, one per input neuron)
    next = s + alpha * (mu - s) + sigma * kick          activity = max(next, 0)
    kick is shared inside a "practice group" with correlation rho.
    If the practice groups match the real sensor sides, neurons that will later
    see the same light already "fire together" during practice, so rule 1 can
    pre-wire hidden A and B to the two sides before the robot sees any light.

Controls
    Play / Pause, Step, Restart (same robot), New robot (new shuffle + birth wiring),
    Compare 100 lifetimes (averages over many robots with the current settings).
    Sliders for alpha, sigma, mu, rho, practice length and animation speed.
    Practice grouping: matches the real sensors, or wrong (each group mixes sides).
    Keyboard: space = play/pause, s = step, r = restart, n = new robot.

Run:   python spontaneous_activity_app.py
       python spontaneous_activity_app.py --save shot.png   (headless screenshot)
Needs: numpy, matplotlib (with a GUI backend such as TkAgg or QtAgg)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons, Slider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NS = 8                 # input neurons (sensors)
N_TRIALS = 200         # trials in the world per lifetime
ETA_H = 0.02           # wiring rule learning rate (input -> hidden)
ETA_V = 0.5            # reward rule learning rate (hidden -> output)
BETA = 4.0             # how decisively the agent follows its outputs
LANE_LEN = 120         # steps shown in the input-activity panel
WIN = 20               # moving-average window for success

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_LEFT, C_RIGHT = "#2a78d6", "#e87ba4"          # sensors facing left / right
C_PRAC, C_NOPR = "#1baf7a", "#888780"           # with / without practice
C_POS, C_NEG = "#1baf7a", "#eb6834"             # weight sign (hidden -> output)
C_HIT, C_MISS = "#1baf7a", "#e34948"
BAND = "#eeedfe"                                 # practice-phase shading


# ---------------------------------------------------------------------------
# Simulation pieces (shared by the live app and the 100-lifetime comparison)
# ---------------------------------------------------------------------------
def new_robot(seed):
    r = np.random.default_rng(seed)
    side = r.permutation(np.array([0] * 4 + [1] * 4))   # 0 = faces left, 1 = faces right
    W0 = r.uniform(0.5, 1.5, (2, NS))
    W0 /= W0.sum(1, keepdims=True)                        # input -> hidden, rows sum to 1
    V0 = r.normal(0, 0.1, (2, 2))                         # hidden -> output
    return side, W0, V0


def practice_groups(side, grouping):
    if grouping == "match":
        return side.copy()
    g = np.zeros(NS, int)                                 # wrong: each group = 2 left + 2 right
    for s in (0, 1):
        idx = np.where(side == s)[0]
        g[idx[:2]], g[idx[2:]] = 0, 1
    return g


def ou_step(s, rng, alpha, sigma, mu, rho, groups):
    kick = np.sqrt(rho) * rng.standard_normal(2)[groups] + np.sqrt(1 - rho) * rng.standard_normal(NS)
    s = s + alpha * (mu - s) + sigma * kick
    return s, np.maximum(s, 0.0)


def sensor_reading(rng, side, light):
    """Sensors facing the light read ~1.0, the others ~0.1, plus noise."""
    return np.maximum(np.where(side == light, 1.0, 0.1) + 0.25 * rng.standard_normal(NS), 0.0)


class Agent:
    def __init__(self, W0, V0, rng):
        self.W, self.V = W0.copy(), V0.copy()
        self.rbar = 0.5                      # expected reward (running average)
        self.rng = rng
        self.last = None                     # (action, reward, light)
        self.successes = []

    def wiring_rule(self, x):
        """Competitive Hebbian: the most responsive hidden neuron moves toward the input pattern."""
        total = x.sum()
        if total < 0.3:                      # too quiet to learn from
            return None
        j = int(np.argmax(self.W @ x))
        self.W[j] += ETA_H * (x / total - self.W[j])
        return j

    def act(self, x, light):
        h = self.W @ x
        z = self.V @ h
        p_right = 1 / (1 + np.exp(-BETA * (z[1] - z[0])))
        a = int(self.rng.random() < p_right)
        reward = float(a == light)
        self.V[a] += ETA_V * (reward - self.rbar) * h        # reward rule
        np.clip(self.V, -3, 3, out=self.V)
        self.rbar += 0.05 * (reward - self.rbar)
        self.wiring_rule(x)                                   # wiring rule keeps running
        self.last = (a, reward, light)
        self.successes.append(reward)
        return p_right

    def selectivity(self, side):
        """0 = both hidden neurons mix both sides, 1 = each owns one side."""
        d = self.W[:, side == 0].sum(1) - self.W[:, side == 1].sum(1)
        return abs(d[0] - d[1]) / 2


def run_lifetime(seed, T, alpha, sigma, mu, rho, grouping):
    side, W0, V0 = new_robot(seed)
    P = Agent(W0, V0, np.random.default_rng(seed + 11))
    N = Agent(W0, V0, np.random.default_rng(seed + 22))
    noise = np.random.default_rng(seed + 33)
    g = practice_groups(side, grouping)
    s = np.full(NS, mu)
    for _ in range(T):
        s, x = ou_step(s, noise, alpha, sigma, mu, rho, g)
        P.wiring_rule(x)
    world = np.random.default_rng(seed + 44)
    for _ in range(N_TRIALS):
        light = int(world.integers(2))
        x = sensor_reading(world, side, light)
        P.act(x, light)
        N.act(x, light)
    return np.array(P.successes), np.array(N.successes)


def moving_avg(v, w=WIN):
    v = np.asarray(v, float)
    if len(v) == 0:
        return v
    c = np.cumsum(np.insert(v, 0, 0))
    out = np.empty(len(v))
    for i in range(len(v)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, seed=0):
        self.robot_seed = seed
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK2,
            "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.fig = plt.figure(figsize=(15, 10.5))
        self.fig.canvas.manager.set_window_title("Spontaneous activity: practice before birth")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.25, 1], left=0.04, right=0.985,
                                   top=0.92, bottom=0.31, hspace=0.38, wspace=0.12)
        self.axIn = self.fig.add_subplot(gs[0, 0])
        self.axP = self.fig.add_subplot(gs[0, 1])
        self.axN = self.fig.add_subplot(gs[0, 2])
        self.axSel = self.fig.add_subplot(gs[1, 0])
        self.axSuc = self.fig.add_subplot(gs[1, 1:])
        self.title = self.fig.text(0.04, 0.965, "", fontsize=13, va="center")

        self._build_lanes()
        self.netP = self._build_net(self.axP)
        self.netN = self._build_net(self.axN)
        self._build_plots()
        self._build_controls()

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)
        self.playing = False
        self.batch = None
        self.restart()

    # ---------------- panels ----------------
    def _build_lanes(self):
        ax = self.axIn
        ax.set_xlim(0, LANE_LEN - 1)
        ax.set_ylim(-0.6, NS * 2.2 - 0.2)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.lanes, self.lane_labels = [], []
        for i in range(NS):
            base = (NS - 1 - i) * 2.2
            ax.axhline(base, color=GRID, lw=0.6)
            ln, = ax.plot([], [], lw=1.6)
            self.lanes.append(ln)
            self.lane_labels.append(ax.text(-2, base + 0.5, f"in {i+1}", ha="right", va="center", fontsize=9))
        self.lane_buf = np.zeros((NS, 0))
        ax.text(0, -0.55, "color = which side the sensor really faces (the network is not told)",
                fontsize=8.5, color=INK2, va="top")

    def _build_net(self, ax):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10.4)
        ax.set_aspect("equal"); ax.axis("off")
        ys = np.linspace(9.4, 0.9, NS)
        pos = {"in": [(1.2, y) for y in ys], "hid": [(5.0, 7.0), (5.0, 3.3)], "out": [(8.4, 7.0), (8.4, 3.3)]}
        net = {"pos": pos, "e_in": [[None] * NS for _ in range(2)], "e_out": [[None] * 2 for _ in range(2)]}
        for j in range(2):
            for i in range(NS):
                (x0, y0), (x1, y1) = pos["in"][i], pos["hid"][j]
                net["e_in"][j][i], = ax.plot([x0, x1], [y0, y1], lw=1, solid_capstyle="round", zorder=1)
        for k in range(2):
            for j in range(2):
                (x0, y0), (x1, y1) = pos["hid"][j], pos["out"][k]
                net["e_out"][k][j], = ax.plot([x0, x1], [y0, y1], lw=1, solid_capstyle="round", zorder=1)
        net["in_nodes"] = []
        for i, (x, y) in enumerate(pos["in"]):
            c = Circle((x, y), 0.36, ec=SURFACE, lw=1.2, zorder=3)
            ax.add_patch(c); net["in_nodes"].append(c)
        net["hid_nodes"] = []
        for j, (x, y) in enumerate(pos["hid"]):
            c = Circle((x, y), 0.62, ec=INK2, lw=1, zorder=3)
            ax.add_patch(c); net["hid_nodes"].append(c)
            ax.text(x, y, "AB"[j], ha="center", va="center", fontsize=11, zorder=4, color=SURFACE,
                    fontweight="bold")
        for k, (x, y) in enumerate(pos["out"]):
            ax.add_patch(Circle((x, y), 0.62, fc=SURFACE, ec=INK2, lw=1, zorder=3))
            ax.text(x, y, "L" if k == 0 else "R", ha="center", va="center", fontsize=11, zorder=4)
            ax.text(x, y - 0.95, "turn left" if k == 0 else "turn right", ha="center", va="top",
                    fontsize=8.5, color=INK2)
        net["ring"] = Circle(pos["out"][0], 0.85, fill=False, lw=2.4, zorder=2)
        ax.add_patch(net["ring"])
        net["light"] = ax.text(9.35, 7.0, "light", ha="left", va="center", fontsize=9.5, fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.25", fc="#fdf1dc", ec="#c98500", lw=1))
        net["status"] = ax.text(0.2, -0.2, "", fontsize=9.5, color=INK2, va="top")
        return net

    def _build_plots(self):
        for a in (self.axSel, self.axSuc):
            a.grid(True, color=GRID, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        ax = self.axSel
        self.sel_P, = ax.plot([], [], color=C_PRAC, lw=2.2, label="with practice")
        self.sel_N, = ax.plot([], [], color=C_NOPR, lw=2.2, label="without practice")
        ax.set_ylim(-0.03, 1.0)
        ax.set_title("Wiring quality: do A and B own different sides?", loc="left")
        ax.set_xlabel("time since birth (practice steps, then world trials)")
        ax.set_ylabel("selectivity (0 = mixed, ~0.8 = sorted)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        self.sel_band = None
        ax = self.axSuc
        self.suc_P, = ax.plot([], [], color=C_PRAC, lw=2.4, label="with practice (this robot)")
        self.suc_N, = ax.plot([], [], color=C_NOPR, lw=2.4, label="without practice (this robot)")
        self.bat_P, = ax.plot([], [], color=C_PRAC, lw=1.4, ls="--", label="with practice (avg of 100 robots)")
        self.bat_N, = ax.plot([], [], color=C_NOPR, lw=1.4, ls="--", label="without practice (avg of 100 robots)")
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.text(N_TRIALS - 1, 0.515, "guessing", ha="right", fontsize=8.5, color=INK2)
        ax.set_xlim(0, N_TRIALS); ax.set_ylim(0.2, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"Success in the world (last {WIN} trials)", loc="left")
        ax.set_xlabel("trial in the world")
        ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
        self.batch_text = ax.text(0, 1.11, "", transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")

    def _build_controls(self):
        f = self.fig
        x0, w = 0.10, 0.27
        mk = lambda y, label, lo, hi, v, step: Slider(f.add_axes([x0, y, w, 0.022]), label, lo, hi,
                                                     valinit=v, valstep=step, color=C_PRAC)
        self.s_alpha = mk(0.225, "pull-back α", 0.01, 0.5, 0.05, 0.01)
        self.s_sigma = mk(0.190, "noise size σ", 0.0, 0.6, 0.2, 0.01)
        self.s_mu = mk(0.155, "set point μ", -1.5, 1.5, -0.3, 0.05)
        self.s_rho = mk(0.120, "group correlation ρ", 0.0, 1.0, 0.8, 0.05)
        self.s_T = mk(0.085, "practice length", 0, 600, 300, 10)
        self.s_speed = Slider(f.add_axes([x0, 0.050, w, 0.022]), "steps per frame", 1, 20,
                              valinit=4, valstep=1, color=C_NOPR)
        f.text(x0, 0.015, "α, σ, μ, ρ and grouping act live; practice length applies on Restart.",
               fontsize=8.5, color=INK2)
        axr = f.add_axes([0.45, 0.13, 0.22, 0.11])
        axr.set_title("practice grouping", fontsize=9.5, loc="left", color=INK2)
        self.r_group = RadioButtons(axr, ["matches the real sensor sides", "wrong (each group mixes sides)"],
                                    active=0, activecolor=C_PRAC)
        for s in axr.spines.values():
            s.set_visible(False)
        y = 0.055
        self.b_play = Button(f.add_axes([0.45, y, 0.06, 0.045]), "Play")
        self.b_step = Button(f.add_axes([0.515, y, 0.05, 0.045]), "Step")
        self.b_restart = Button(f.add_axes([0.57, y, 0.065, 0.045]), "Restart")
        self.b_new = Button(f.add_axes([0.64, y, 0.075, 0.045]), "New robot")
        self.b_batch = Button(f.add_axes([0.72, y, 0.13, 0.045]), "Compare 100 lifetimes")
        self.b_play.on_clicked(lambda e: self.toggle_play())
        self.b_step.on_clicked(lambda e: self.step())
        self.b_restart.on_clicked(lambda e: self.restart())
        self.b_new.on_clicked(lambda e: self.new_robot())
        self.b_batch.on_clicked(lambda e: self.compare_batch())
        f.text(0.70, 0.235,
               "Practice noise, one per input neuron:\n"
               "   next = s + α·(μ − s) + σ·kick\n"
               "   activity = max(next, 0)\n"
               "   kicks are shared inside a practice group\n"
               "   with correlation ρ",
               fontsize=9, color=INK2, va="top", linespacing=1.5)
        f.text(0.45, 0.015, "Keys: space = play/pause, s = step, r = restart, n = new robot",
               fontsize=8.5, color=INK2)
        f.canvas.mpl_connect("key_press_event", self.on_key)

    # ---------------- control actions ----------------
    def grouping(self):
        return "match" if self.r_group.value_selected.startswith("matches") else "wrong"

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")
        (self.timer.start if self.playing else self.timer.stop)()
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        {" ": self.toggle_play, "s": self.step, "r": self.restart, "n": self.new_robot}.get(e.key, lambda: None)()

    def new_robot(self):
        self.robot_seed += 1
        self.batch = None
        self.restart()

    def restart(self):
        if self.playing:
            self.toggle_play()
        self.side, W0, V0 = new_robot(self.robot_seed)
        self.P = Agent(W0, V0, np.random.default_rng(self.robot_seed + 11))
        self.N = Agent(W0, V0, np.random.default_rng(self.robot_seed + 22))
        self.noise = np.random.default_rng(self.robot_seed + 33)
        self.world = np.random.default_rng(self.robot_seed + 44)
        self.T = int(self.s_T.val)
        self.s = np.full(NS, self.s_mu.val)
        self.t, self.trial = 0, 0
        self.phase = "practice" if self.T > 0 else "world"
        self.lane_buf = np.zeros((NS, 0))
        self.selP = [(0, self.P.selectivity(self.side))]
        self.selN = []
        self.last_x = np.zeros(NS)
        self.draw()

    def compare_batch(self):
        args = (int(self.s_T.val), self.s_alpha.val, self.s_sigma.val, self.s_mu.val,
                self.s_rho.val, self.grouping())
        P, N = [], []
        for k in range(100):
            p, n = run_lifetime(1000 + k, *args)
            P.append(p); N.append(n)
        P, N = np.array(P), np.array(N)
        self.batch = (moving_avg(P.mean(0)), moving_avg(N.mean(0)),
                      P[:, :50].mean(), N[:, :50].mean(), P[:, 50:].mean(), N[:, 50:].mean(), args)
        self.draw()

    # ---------------- simulation ----------------
    def _advance(self):
        if self.phase == "practice":
            g = practice_groups(self.side, self.grouping())
            self.s, x = ou_step(self.s, self.noise, self.s_alpha.val, self.s_sigma.val,
                                self.s_mu.val, self.s_rho.val, g)
            self.P.wiring_rule(x)
            self.last_x = x
            self.t += 1
            self.selP.append((self.t, self.P.selectivity(self.side)))
            self._push_lane(x)
            if self.t >= self.T:
                self.phase = "world"
        elif self.phase == "world":
            light = int(self.world.integers(2))
            x = sensor_reading(self.world, self.side, light)
            self.P.act(x, light)
            self.N.act(x, light)
            self.last_x = x
            self.t += 1
            self.trial += 1
            self.selP.append((self.t, self.P.selectivity(self.side)))
            self.selN.append((self.t, self.N.selectivity(self.side)))
            self._push_lane(x)
            if self.trial >= N_TRIALS:
                self.phase = "done"

    def _push_lane(self, x):
        self.lane_buf = np.concatenate([self.lane_buf, x[:, None]], axis=1)[:, -LANE_LEN:]

    def step(self):
        if self.phase != "done":
            self._advance()
        self.draw()

    def _tick(self):
        for _ in range(int(self.s_speed.val)):
            if self.phase == "done":
                break
            self._advance()
        if self.phase == "done" and self.playing:
            self.toggle_play()
        self.draw()

    # ---------------- drawing ----------------
    def _draw_net(self, ax, net, agent, title, idle):
        side = self.side
        for i in range(NS):
            net["in_nodes"][i].set_facecolor(C_LEFT if side[i] == 0 else C_RIGHT)
            net["in_nodes"][i].set_alpha(0.35 + 0.65 * min(1.0, self.last_x[i]) if not idle else 0.35)
        for j in range(2):
            share_left = agent.W[j, side == 0].sum()
            col = np.array(to_rgb(C_LEFT)) * share_left + np.array(to_rgb(C_RIGHT)) * (1 - share_left)
            net["hid_nodes"][j].set_facecolor(col)
            for i in range(NS):
                e = net["e_in"][j][i]
                e.set_color(C_LEFT if side[i] == 0 else C_RIGHT)
                e.set_linewidth(0.3 + 16 * agent.W[j, i])
                e.set_alpha(0.85)
        for k in range(2):
            for j in range(2):
                v = agent.V[k, j]
                e = net["e_out"][k][j]
                e.set_color(C_POS if v >= 0 else C_NEG)
                e.set_linewidth(0.5 + 1.8 * abs(v))
                e.set_alpha(0.3 if abs(v) < 0.05 else 1.0)
        in_world = self.phase in ("world", "done") and agent.last is not None
        net["light"].set_visible(in_world)
        net["ring"].set_visible(in_world)
        sel = agent.selectivity(side)
        if in_world:
            a, r, light = agent.last
            net["light"].set_position((9.35, net["pos"]["out"][light][1]))
            net["ring"].center = net["pos"]["out"][a]
            net["ring"].set_edgecolor(C_HIT if r else C_MISS)
            sr = np.mean(agent.successes[-WIN:])
            net["status"].set_text(f"trial {self.trial}: light {'left' if light == 0 else 'right'}, "
                                   f"turned {'left' if a == 0 else 'right'} {'(reward 1)' if r else '(reward 0)'}\n"
                                   f"success last {WIN}: {sr:.0%}    wiring selectivity: {sel:.2f}")
        elif idle:
            net["status"].set_text("not born yet: waiting while the other agent practices\n"
                                   f"wiring selectivity: {sel:.2f}")
        else:
            net["status"].set_text(f"practice step {self.t}/{self.T}: no light, no reward,\n"
                                   f"only the wiring rule runs    wiring selectivity: {sel:.2f}")
        ax.set_title(title, loc="left", fontsize=11)

    def draw(self):
        # input lanes
        buf = self.lane_buf
        n = buf.shape[1]
        xs = np.arange(LANE_LEN - n, LANE_LEN)
        for i in range(NS):
            base = (NS - 1 - i) * 2.2
            col = C_LEFT if self.side[i] == 0 else C_RIGHT
            self.lanes[i].set_data(xs, base + np.clip(buf[i], 0, 1.9))
            self.lanes[i].set_color(col)
            self.lane_labels[i].set_color(col)
        src = ("practice noise (OU), no light" if self.phase == "practice"
               else "real light sensors")
        self.axIn.set_title(f"Input neuron activity: {src}", loc="left")

        self._draw_net(self.axP, self.netP, self.P, "With practice", idle=False)
        self._draw_net(self.axN, self.netN, self.N, "Without practice", idle=self.phase == "practice")

        # selectivity
        if self.selP:
            t, v = zip(*self.selP); self.sel_P.set_data(t, v)
        if self.selN:
            t, v = zip(*self.selN); self.sel_N.set_data(t, v)
        else:
            self.sel_N.set_data([], [])
        if self.sel_band is not None:
            self.sel_band.remove()
        self.sel_band = self.axSel.axvspan(0, self.T, color=BAND, lw=0, zorder=0) if self.T > 0 else None
        self.axSel.set_xlim(0, self.T + N_TRIALS)

        # success
        sp, sn = moving_avg(self.P.successes), moving_avg(self.N.successes)
        self.suc_P.set_data(np.arange(1, len(sp) + 1), sp)
        self.suc_N.set_data(np.arange(1, len(sn) + 1), sn)
        if self.batch:
            bp, bn, p50, n50, pl, nl, args = self.batch
            x = np.arange(1, N_TRIALS + 1)
            self.bat_P.set_data(x, bp); self.bat_N.set_data(x, bn)
            self.batch_text.set_text(f"Average of 100 robots (dashed):  trials 1-50: with practice {p50:.0%}, "
                                     f"without {n50:.0%}    trials 51-200: {pl:.0%} vs {nl:.0%}")
        else:
            self.bat_P.set_data([], []); self.bat_N.set_data([], [])
            self.batch_text.set_text("")

        phase_txt = {"practice": f"PRACTICE (before birth)  step {self.t}/{self.T}",
                     "world": f"IN THE WORLD  trial {self.trial}/{N_TRIALS}",
                     "done": "LIFETIME OVER"}[self.phase]
        self.title.set_text(f"Spontaneous activity: practice before birth    robot #{self.robot_seed}    {phase_txt}")
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description="Spontaneous activity, visualized")
    p.add_argument("--seed", type=int, default=0, help="robot number to start with")
    p.add_argument("--save", type=str, default=None, help="simulate headlessly and save a PNG")
    p.add_argument("--steps", type=int, default=360, help="steps to simulate with --save")
    a = p.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(seed=a.seed)
    if a.save:
        for _ in range(a.steps):
            app._advance()
        app.compare_batch()
        app.fig.savefig(a.save, dpi=110)
        print(f"Saved {a.save}: phase {app.phase}, trial {app.trial}, "
              f"selectivity P {app.P.selectivity(app.side):.2f} N {app.N.selectivity(app.side):.2f}")
    else:
        plt.show()


if __name__ == "__main__":
    main()