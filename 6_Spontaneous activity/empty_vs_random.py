"""
Empty vs. random wiring at birth, both with spontaneous activity (practice before birth).

Same robot, same practice noise, same lights, same sensor readings, same dice for
choosing turns. The ONLY difference between the two agents is the input -> hidden
wiring they are born with:
    Random wiring:  every input connected to A and B with a random strength (as in compare.py)
    Empty wiring:   pick one
                      all zeros  - no connections at all
                      all equal  - every connection the same strength (1/8), a "blank slate"
Hidden -> output weights start the same (small, random) for both.

What to look for
    The wiring rule is competitive: each step, only the hidden neuron that responds MOST
    to the input learns (it "wins" the step). A and B must end up winning on DIFFERENT
    patterns to split the sensors between them ("symmetry breaking").
    Random wiring:  small random differences break the tie from the start.
    All equal:      the first step is a tie (A wins), but that win moves A away from B,
                    so the next pattern may suit B better. The tie breaks itself and the
                    result matches random wiring.
    All zeros:      A wins the tie, gets positive weights, and from then on responds more
                    than B (still zero) to EVERY input. B never wins, never learns, and A
                    averages over both sides -> the robot stays at guessing.

Controls
    Play / Pause, Step, Restart (same robot), New robot, Compare 100 lifetimes.
    Empty wiring type (all zeros / all equal), practice length, animation speed.
    Keyboard: space = play/pause, s = step, r = restart, n = new robot.

Run:   python empty_vs_random.py
       python empty_vs_random.py --save shot.png   (headless screenshot)
Needs: compare.py in the same folder, numpy, matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider

from compare import (NS, N_TRIALS, LANE_LEN, WIN, SURFACE, INK, INK2, GRID, C_LEFT, C_RIGHT,
                     C_POS, C_NEG, C_HIT, C_MISS, BAND, new_robot, practice_groups, ou_step,
                     sensor_reading, Agent, moving_avg)

C_RAND, C_EMPTY = "#1baf7a", "#eb6834"          # random / empty birth wiring
EMPTY_GREY = "#d3d1c7"                           # hidden neuron with no connections
ALPHA, SIGMA, MU, RHO = 0.05, 0.2, -0.3, 0.8     # practice noise: compare.py defaults,
GROUPING = "match"                               # grouped by the real sensor sides
EMPTY_KINDS = {"all zeros (no connections)": "zeros", "all equal (1/8 each)": "equal"}


# ---------------------------------------------------------------------------
# Simulation pieces
# ---------------------------------------------------------------------------
def empty_wiring(kind):
    return np.zeros((2, NS)) if kind == "zeros" else np.full((2, NS), 1 / NS)


class Tracked(Agent):
    """Agent that also counts how often each hidden neuron wins the wiring rule."""

    def __init__(self, W0, V0, rng):
        super().__init__(W0, V0, rng)
        self.wins = [0, 0]

    def wiring_rule(self, x):
        j = super().wiring_rule(x)
        if j is not None:
            self.wins[j] += 1
        return j


def make_agents(seed, kind):
    """Same robot, same hidden -> output weights, same turn dice; only W0 differs."""
    side, W0, V0 = new_robot(seed)
    R = Tracked(W0, V0, np.random.default_rng(seed + 11))
    E = Tracked(empty_wiring(kind), V0, np.random.default_rng(seed + 11))
    return side, R, E


def run_lifetime(seed, T, kind):
    side, R, E = make_agents(seed, kind)
    noise = np.random.default_rng(seed + 33)
    g = practice_groups(side, GROUPING)
    s = np.full(NS, MU)
    for _ in range(T):
        s, x = ou_step(s, noise, ALPHA, SIGMA, MU, RHO, g)
        R.wiring_rule(x); E.wiring_rule(x)
    world = np.random.default_rng(seed + 44)
    for _ in range(N_TRIALS):
        light = int(world.integers(2))
        x = sensor_reading(world, side, light)
        R.act(x, light); E.act(x, light)
    return R, E, side


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
        self.fig.canvas.manager.set_window_title("Spontaneous activity: empty vs. random birth wiring")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.25, 1], left=0.04, right=0.985,
                                   top=0.92, bottom=0.31, hspace=0.38, wspace=0.12)
        self.axIn = self.fig.add_subplot(gs[0, 0])
        self.axR = self.fig.add_subplot(gs[0, 1])
        self.axE = self.fig.add_subplot(gs[0, 2])
        self.axSel = self.fig.add_subplot(gs[1, 0])
        self.axSuc = self.fig.add_subplot(gs[1, 1:])
        self.title = self.fig.text(0.04, 0.965, "", fontsize=13, va="center")

        self._build_lanes()
        self.netR = self._build_net(self.axR)
        self.netE = self._build_net(self.axE)
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
        ax.text(0, -0.55, "same input for both agents; color = side the sensor really faces",
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
        net["hid_nodes"], net["wins"] = [], []
        for j, (x, y) in enumerate(pos["hid"]):
            c = Circle((x, y), 0.62, ec=INK2, lw=1, zorder=3)
            ax.add_patch(c); net["hid_nodes"].append(c)
            ax.text(x, y, "AB"[j], ha="center", va="center", fontsize=11, zorder=4, color=SURFACE,
                    fontweight="bold")
            net["wins"].append(ax.text(x, y + 0.85, "", ha="center", va="bottom", fontsize=8.5,
                                       color=INK2, zorder=4))
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
        self.sel_R, = ax.plot([], [], color=C_RAND, lw=2.2, label="random wiring")
        self.sel_E, = ax.plot([], [], color=C_EMPTY, lw=2.2, label="empty wiring")
        ax.set_ylim(-0.03, 1.0)
        ax.set_title("Wiring quality: do A and B own different sides?", loc="left")
        ax.set_xlabel("time since birth (practice steps, then world trials)")
        ax.set_ylabel("selectivity (0 = mixed, ~0.8 = sorted)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        self.sel_band = None
        ax = self.axSuc
        self.suc_R, = ax.plot([], [], color=C_RAND, lw=2.4, label="random wiring (this robot)")
        self.suc_E, = ax.plot([], [], color=C_EMPTY, lw=2.4, label="empty wiring (this robot)")
        self.bat_R, = ax.plot([], [], color=C_RAND, lw=1.4, ls="--", label="random wiring (avg of 100 robots)")
        self.bat_E, = ax.plot([], [], color=C_EMPTY, lw=1.4, ls="--", label="empty wiring (avg of 100 robots)")
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
        self.s_T = Slider(f.add_axes([x0, 0.190, w, 0.022]), "practice length", 0, 600,
                          valinit=300, valstep=10, color=C_RAND)
        self.s_speed = Slider(f.add_axes([x0, 0.150, w, 0.022]), "steps per frame", 1, 20,
                              valinit=4, valstep=1, color="#888780")
        f.text(x0, 0.105, f"Practice noise (same for both): α = {ALPHA}, σ = {SIGMA}, μ = {MU}, ρ = {RHO},\n"
                          "grouped by the real sensor sides. Practice length applies on Restart.",
               fontsize=8.5, color=INK2, va="top", linespacing=1.5)
        axr = f.add_axes([0.45, 0.13, 0.22, 0.11])
        axr.set_title("empty wiring =", fontsize=9.5, loc="left", color=INK2)
        self.r_kind = RadioButtons(axr, list(EMPTY_KINDS), active=0, activecolor=C_EMPTY)
        self.r_kind.on_clicked(lambda _: (setattr(self, "batch", None), self.restart()))
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
               "The wiring rule is competitive: only the hidden\n"
               "neuron that responds MOST to the input learns.\n"
               "Random: small differences break the A/B tie.\n"
               "All equal: A wins step 1, which makes A and B\n"
               "   differ, so the tie breaks itself.\n"
               "All zeros: once A has any weight it beats B (0)\n"
               "   on every input, so B never learns.",
               fontsize=9, color=INK2, va="top", linespacing=1.5)
        f.text(0.45, 0.015, "Keys: space = play/pause, s = step, r = restart, n = new robot",
               fontsize=8.5, color=INK2)
        f.canvas.mpl_connect("key_press_event", self.on_key)

    # ---------------- control actions ----------------
    def kind(self):
        return EMPTY_KINDS[self.r_kind.value_selected]

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
        self.side, self.R, self.E = make_agents(self.robot_seed, self.kind())
        self.noise = np.random.default_rng(self.robot_seed + 33)
        self.world = np.random.default_rng(self.robot_seed + 44)
        self.groups = practice_groups(self.side, GROUPING)
        self.T = int(self.s_T.val)
        self.s = np.full(NS, MU)
        self.t, self.trial = 0, 0
        self.phase = "practice" if self.T > 0 else "world"
        self.lane_buf = np.zeros((NS, 0))
        self.selR = [(0, self.R.selectivity(self.side))]
        self.selE = [(0, self.E.selectivity(self.side))]
        self.last_x = np.zeros(NS)
        self.draw()

    def compare_batch(self):
        T, kind = int(self.s_T.val), self.kind()
        R, E, selR, selE = [], [], [], []
        for k in range(100):
            r, e, side = run_lifetime(1000 + k, T, kind)
            R.append(r.successes); E.append(e.successes)
            selR.append(r.selectivity(side)); selE.append(e.selectivity(side))
        R, E = np.array(R), np.array(E)
        self.batch = (moving_avg(R.mean(0)), moving_avg(E.mean(0)),
                      R[:, :50].mean(), E[:, :50].mean(), R[:, 50:].mean(), E[:, 50:].mean(),
                      np.mean(selR), np.mean(selE))
        self.draw()

    # ---------------- simulation ----------------
    def _advance(self):
        if self.phase == "practice":
            self.s, x = ou_step(self.s, self.noise, ALPHA, SIGMA, MU, RHO, self.groups)
            self.R.wiring_rule(x); self.E.wiring_rule(x)
        elif self.phase == "world":
            light = int(self.world.integers(2))
            x = sensor_reading(self.world, self.side, light)
            self.R.act(x, light); self.E.act(x, light)
            self.trial += 1
        else:
            return
        self.last_x = x
        self.t += 1
        self.selR.append((self.t, self.R.selectivity(self.side)))
        self.selE.append((self.t, self.E.selectivity(self.side)))
        self.lane_buf = np.concatenate([self.lane_buf, x[:, None]], axis=1)[:, -LANE_LEN:]
        if self.phase == "practice" and self.t >= self.T:
            self.phase = "world"
        elif self.phase == "world" and self.trial >= N_TRIALS:
            self.phase = "done"

    def step(self):
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
    def _draw_net(self, ax, net, agent, title, color):
        side = self.side
        for i in range(NS):
            net["in_nodes"][i].set_facecolor(C_LEFT if side[i] == 0 else C_RIGHT)
            net["in_nodes"][i].set_alpha(0.35 + 0.65 * min(1.0, self.last_x[i]))
        for j in range(2):
            total = agent.W[j].sum()
            if total < 1e-9:
                col = to_rgb(EMPTY_GREY)                      # no connections at all
            else:
                share_left = agent.W[j, side == 0].sum() / total
                col = np.array(to_rgb(C_LEFT)) * share_left + np.array(to_rgb(C_RIGHT)) * (1 - share_left)
            net["hid_nodes"][j].set_facecolor(col)
            net["wins"][j].set_text(f"won {agent.wins[j]} steps")
            for i in range(NS):
                e = net["e_in"][j][i]
                e.set_color(C_LEFT if side[i] == 0 else C_RIGHT)
                e.set_linewidth(0.3 + 16 * agent.W[j, i])
                e.set_alpha(0.85 if agent.W[j, i] > 1e-6 else 0.15)
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
        else:
            net["status"].set_text(f"practice step {self.t}/{self.T}: no light, no reward,\n"
                                   f"only the wiring rule runs    wiring selectivity: {sel:.2f}")
        ax.set_title(title, loc="left", fontsize=11, color=color)

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
        src = "practice noise (OU), no light" if self.phase == "practice" else "real light sensors"
        self.axIn.set_title(f"Input neuron activity: {src}", loc="left")

        empty_name = "all zeros" if self.kind() == "zeros" else "all equal"
        self._draw_net(self.axR, self.netR, self.R, "Born with RANDOM wiring", C_RAND)
        self._draw_net(self.axE, self.netE, self.E, f"Born with EMPTY wiring ({empty_name})", C_EMPTY)

        # selectivity
        t, v = zip(*self.selR); self.sel_R.set_data(t, v)
        t, v = zip(*self.selE); self.sel_E.set_data(t, v)
        if self.sel_band is not None:
            self.sel_band.remove()
        self.sel_band = self.axSel.axvspan(0, self.T, color=BAND, lw=0, zorder=0) if self.T > 0 else None
        self.axSel.set_xlim(0, self.T + N_TRIALS)

        # success
        sr, se = moving_avg(self.R.successes), moving_avg(self.E.successes)
        self.suc_R.set_data(np.arange(1, len(sr) + 1), sr)
        self.suc_E.set_data(np.arange(1, len(se) + 1), se)
        if self.batch:
            br, be, r50, e50, rl, el, sR, sE = self.batch
            x = np.arange(1, N_TRIALS + 1)
            self.bat_R.set_data(x, br); self.bat_E.set_data(x, be)
            self.batch_text.set_text(f"Average of 100 robots (dashed):  trials 1-50: random {r50:.0%}, "
                                     f"empty {e50:.0%}    trials 51-200: {rl:.0%} vs {el:.0%}    "
                                     f"final selectivity: {sR:.2f} vs {sE:.2f}")
        else:
            self.bat_R.set_data([], []); self.bat_E.set_data([], [])
            self.batch_text.set_text("")

        phase_txt = {"practice": f"PRACTICE (before birth)  step {self.t}/{self.T}",
                     "world": f"IN THE WORLD  trial {self.trial}/{N_TRIALS}",
                     "done": "LIFETIME OVER"}[self.phase]
        self.title.set_text(f"Empty vs. random birth wiring    robot #{self.robot_seed}    {phase_txt}")
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description="Empty vs. random birth wiring, visualized")
    p.add_argument("--seed", type=int, default=0, help="robot number to start with")
    p.add_argument("--save", type=str, default=None, help="simulate headlessly and save a PNG")
    p.add_argument("--steps", type=int, default=360, help="steps to simulate with --save")
    p.add_argument("--empty", choices=["zeros", "equal"], default="zeros", help="empty wiring type")
    a = p.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(seed=a.seed)
    if a.empty == "equal":
        app.r_kind.set_active(1)                 # triggers restart()
    if a.save:
        for _ in range(a.steps):
            app._advance()
        app.compare_batch()
        app.fig.savefig(a.save, dpi=110)
        print(f"Saved {a.save}: phase {app.phase}, trial {app.trial}, "
              f"selectivity random {app.R.selectivity(app.side):.2f} empty {app.E.selectivity(app.side):.2f}, "
              f"wins random {app.R.wins} empty {app.E.wins}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
