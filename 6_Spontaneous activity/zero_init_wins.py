"""
When is starting from ZERO better than starting from random?

empty_vs_random.py showed all-zero wiring failing: zero breaks the competition
between hidden neurons A and B, because nothing makes them different. Zero is only
harmful when the units of a layer must become different from each other and nothing
else breaks the tie. When the tie is already broken (or there is no tie to break),
zero is often the BETTER start. Two experiments, same rules for both starts:

ROW 1: linear regression with more knobs than data (200 knobs, 50 data points)
    Gradient descent from zero vs. from random knobs. Both fit the training data
    perfectly, but they end at DIFFERENT answers:
      - The data can only "see" 50 directions out of 200. Gradient descent never
        moves the knobs in the other 150 directions.
      - Start at zero -> those 150 directions stay at zero -> the smallest answer that
        fits the data (the "minimum-norm" solution), which generalizes best.
      - Start random -> those 150 directions keep their random values forever ->
        junk that hurts every prediction on new data.
    Refs: Zhang et al. 2017 "Understanding deep learning requires rethinking
    generalization" (sec. 5); Gunasekar et al. 2018 "Characterizing implicit bias in
    terms of optimization geometry"; Hastie et al. 2022 "Surprises in high-dimensional
    ridgeless least squares interpolation".

ROW 2: the robot from compare.py, OUTPUT layer (hidden -> turn L / turn R)
    Both agents: same robot, same random input wiring, same 300 practice steps, same
    lights. Only the hidden -> output weights V at birth differ: zero vs. random
    (std 1.0). The tie between turn-left and turn-right is broken by the REWARD, not
    by the start, so zero costs nothing - and it starts unbiased (50/50), while a
    random start begins with a strong opinion that must first be unlearned.
    Refs: Andrychowicz et al. 2021 "What matters in on-policy reinforcement learning?"
    (shrink the last policy layer ~100x so the first policy is near 50/50); and the
    same "start the last layer at zero" idea in Goyal et al. 2017 (zero-init last BN
    gamma), Zhang et al. 2019 "Fixup", Hu et al. 2021 "LoRA" (B = 0),
    Zhang et al. 2023 "ControlNet" (zero convolutions).

Run:   python zero_init_wins.py
Needs: compare.py in the same folder, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from compare import (NS, N_TRIALS, C_LEFT, C_RIGHT, new_robot, practice_groups, ou_step,
                     sensor_reading, Agent, moving_avg)

N_EXAMPLES = 50
FRAMES = N_TRIALS                       # 200 frames: 1 robot trial or 1 GD step each
GD_PER_FRAME = 1

# ---------------------------------------------------------------------------
# Row 1: overparameterized linear regression
# ---------------------------------------------------------------------------
N_KNOBS, N_DATA, N_TEST = 200, 50, 500


def regression(seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N_DATA, N_KNOBS)) / np.sqrt(N_KNOBS)
    true = r.standard_normal(N_KNOBS)
    y = X @ true
    Xt = r.standard_normal((N_TEST, N_KNOBS)) / np.sqrt(N_KNOBS)
    yt = Xt @ true
    lr = 1 / (2 * np.linalg.eigvalsh(X.T @ X / N_DATA).max())
    P_row = np.linalg.pinv(X) @ X                   # projects onto the 50 directions the data sees
    out = {"y": y, "yt": yt}
    for name, th0 in [("zero", np.zeros(N_KNOBS)), ("random", r.standard_normal(N_KNOBS))]:
        th, h = th0.copy(), dict(train=[], test=[], pred=[], hidden=[])
        for f in range(FRAMES + 1):
            h["train"].append(np.mean((X @ th - y) ** 2))
            h["test"].append(np.mean((Xt @ th - yt) ** 2))
            h["pred"].append(Xt @ th)
            h["hidden"].append(np.linalg.norm(th - P_row @ th))   # part the data can't see
            for _ in range(GD_PER_FRAME):
                th -= lr * 2 * X.T @ (X @ th - y) / N_DATA
        out[name] = h
    return out


# ---------------------------------------------------------------------------
# Row 2: the robot, zero vs. random hidden -> output weights
# ---------------------------------------------------------------------------
V_STD = 1.0
PRACTICE = 300


def p_right(agent, side, light):
    """Chance of turning right for a clean light on one side."""
    x = np.where(side == light, 1.0, 0.1)
    z = agent.V @ (agent.W @ x)
    return 1 / (1 + np.exp(-4.0 * (z[1] - z[0])))


def robot(seed):
    side, W0, _ = new_robot(1000 + seed)
    V_rand = np.random.default_rng(5000 + seed).normal(0, V_STD, (2, 2))
    out = {"side": side}
    for name, V0 in [("zero", np.zeros((2, 2))), ("random", V_rand)]:
        A = Agent(W0, V0, np.random.default_rng(seed + 11))
        noise = np.random.default_rng(seed + 33)
        g, s = practice_groups(side, "match"), np.full(NS, -0.3)
        for _ in range(PRACTICE):                    # same practice for both (V is not used)
            s, x = ou_step(s, noise, 0.05, 0.2, -0.3, 0.8, g)
            A.wiring_rule(x)
        world = np.random.default_rng(seed + 44)
        pr = [(p_right(A, side, 0), p_right(A, side, 1))]
        for _ in range(N_TRIALS):
            light = int(world.integers(2))
            A.act(sensor_reading(world, side, light), light)
            pr.append((p_right(A, side, 0), p_right(A, side, 1)))
        out[name] = dict(success=np.array(A.successes), pr=np.array(pr), V0=V0)
    return out


print("Running", N_EXAMPLES, "regression problems and", N_EXAMPLES, "robots ...")
REG = [regression(s) for s in range(N_EXAMPLES)]
ROB = [robot(s) for s in range(N_EXAMPLES)]
avg_test = {n: np.mean([r[n]["test"] for r in REG], 0) for n in ("zero", "random")}
avg_succ = {n: moving_avg(np.mean([r[n]["success"] for r in ROB], 0)) for n in ("zero", "random")}
print(f"Regression, final test error (avg): zero {avg_test['zero'][-1]:.3f}, "
      f"random {avg_test['random'][-1]:.3f}   (predicting 0 everywhere scores ~1.0)")
for n in ("zero", "random"):
    S = np.array([r[n]["success"] for r in ROB])
    print(f"Robot, {n:>6} output start: success trials 1-20 {S[:, :20].mean():.0%}, "
          f"1-50 {S[:, :50].mean():.0%}, 51-200 {S[:, 50:].mean():.0%}")


# ---------------------------------------------------------------------------
# Interactive view (same style and controls as the other scripts)
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"zero": "#eb6834", "random": "#1baf7a"}          # same colors as empty_vs_random.py
LABEL = {"zero": "start at zero", "random": "start random"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False,
})
box = dict(fc=SURFACE, ec="none", alpha=0.85)

fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.87, bottom=0.17, wspace=0.25, hspace=0.55)
axA1, axA2, axA3 = (fig.add_subplot(gs[0, i]) for i in range(3))
axB1, axB2, axB3 = (fig.add_subplot(gs[1, i]) for i in range(3))
title = fig.suptitle("", fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.915, f"1. Linear regression, {N_KNOBS} knobs but only {N_DATA} data points: "
         "both starts fit the data, zero generalizes better", fontsize=11.5, color=INK)
fig.text(0.02, axB1.get_position().y1 + 0.045, f"2. Robot output layer (hidden -> turn): zero vs. random (std {V_STD}) start, "
         "same input wiring + practice. Zero starts unbiased", fontsize=11.5, color=INK)

# A1. train + test error over gradient descent steps
axA1.set_yscale("log"); axA1.set_xlim(0, FRAMES * GD_PER_FRAME)
axA1.set_title("Error vs. gradient descent steps", loc="left")
axA1.set_xlabel("Gradient descent step"); axA1.set_ylabel("Mean squared error (log scale)")
axA1.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
axA1.text(FRAMES * GD_PER_FRAME * 0.98, 1.15, "predict 0 everywhere", color=INK2, fontsize=8.5, ha="right")
A_test, A_train, A_dot = {}, {}, {}
for n in ("zero", "random"):
    axA1.plot(np.arange(FRAMES + 1) * GD_PER_FRAME, avg_test[n], color=COL[n], lw=1.2, ls="--", alpha=0.7)
    A_test[n], = axA1.plot([], [], color=COL[n], lw=2.2, label=f"{LABEL[n]}: test error (new data)")
    A_train[n], = axA1.plot([], [], color=COL[n], lw=1.2, ls=":", label=f"{LABEL[n]}: train error")
    A_dot[n], = axA1.plot([], [], "o", color=COL[n])
axA1.plot([], [], color=INK2, lw=1.2, ls="--", label=f"test error, avg of {N_EXAMPLES} problems")
legA1 = axA1.legend(loc="lower left", fontsize=8.5)
infoA1 = axA1.text(0.98, 0.96, "", transform=axA1.transAxes, ha="right", va="top", bbox=box)

# A2, A3. result on NEW data: prediction vs. answer
A_pts, A_diag, A_info = {}, {}, {}
for ax, n in [(axA2, "zero"), (axA3, "random")]:
    A_diag[n], = ax.plot([], [], color=INK2, lw=1, ls=":", label="perfect fit")
    A_pts[n] = ax.scatter([], [], s=12, color=COL[n], alpha=0.55, edgecolor="none", label="new data point")
    ax.set_title(f"{LABEL[n].capitalize()}: prediction vs. answer on NEW data", loc="left", color=COL[n])
    ax.set_xlabel("Answer"); ax.set_ylabel("Prediction")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8.5)
    A_info[n] = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", bbox=box)

# B1. success in the world
axB1.set_xlim(0, N_TRIALS); axB1.set_ylim(0.2, 1.03)
axB1.axhline(0.5, color=INK2, lw=0.8, ls=":")
axB1.text(N_TRIALS - 1, 0.515, "guessing", ha="right", fontsize=8.5, color=INK2)
axB1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axB1.set_title("Success in the world (last 20 trials)", loc="left")
axB1.set_xlabel("Trial in the world")
B_succ, B_dot = {}, {}
for n in ("zero", "random"):
    axB1.plot(np.arange(1, N_TRIALS + 1), avg_succ[n], color=COL[n], lw=1.2, ls="--", alpha=0.7)
    B_succ[n], = axB1.plot([], [], color=COL[n], lw=2.2, label=f"{LABEL[n]} (this robot)")
    B_dot[n], = axB1.plot([], [], "o", color=COL[n])
axB1.plot([], [], color=INK2, lw=1.2, ls="--", label=f"avg of {N_EXAMPLES} robots")
axB1.legend(loc="center right", fontsize=8.5)
infoB1 = axB1.text(0.02, 0.04, "", transform=axB1.transAxes, bbox=box)

# B2, B3. how the decision changes: chance of turning right for each light side
B_lines, B_info = {}, {}
for ax, n in [(axB2, "zero"), (axB3, "random")]:
    ax.set_xlim(0, N_TRIALS); ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"{LABEL[n].capitalize()}: chance of turning right", loc="left", color=COL[n])
    ax.set_xlabel("Trial in the world")
    B_lines[n] = (ax.plot([], [], color=C_RIGHT, lw=2.2, label="light on the RIGHT (goal 100%)")[0],
                  ax.plot([], [], color=C_LEFT, lw=2.2, label="light on the LEFT (goal 0%)")[0])
    ax.legend(loc="center right", fontsize=8.5)
    B_info[n] = ax.text(0.98, 0.30, "", transform=ax.transAxes, ha="right", va="top", bbox=box, fontsize=9)

example = 0


def set_example(e):
    global example
    example = e
    reg = REG[e]
    yt = reg["yt"]
    allp = np.concatenate([reg[n]["pred"][-1] for n in ("zero", "random")] + [yt])
    lim = (-np.abs(allp).max() * 1.05, np.abs(allp).max() * 1.05)
    for ax, n in [(axA2, "zero"), (axA3, "random")]:
        ax.set_xlim(lim); ax.set_ylim(lim); A_diag[n].set_data(lim, lim)
    hi = max(max(reg[n]["test"]) for n in ("zero", "random"))
    axA1.set_ylim(1e-4, hi * 3)            # train error keeps falling to ~0 below this
    title.set_text(f"When does starting from ZERO beat starting random?    example #{e}  "
                   f"(regression problem #{e}, robot #{1000 + e})")
    update(int(slider.val))


def update(k):
    it = k * GD_PER_FRAME
    reg, rob = REG[example], ROB[example]
    for j, n in enumerate(("zero", "random")):
        h = reg[n]
        xs = np.arange(k + 1) * GD_PER_FRAME
        A_test[n].set_data(xs, h["test"][:k + 1]); A_train[n].set_data(xs, h["train"][:k + 1])
        A_dot[n].set_data([it], [h["test"][k]])
        A_pts[n].set_offsets(np.column_stack([reg["yt"], h["pred"][k]]))
        A_info[n].set_text(f"GD step {it}/{FRAMES * GD_PER_FRAME}\n"
                           f"train error {h['train'][k]:.3g}   test error {h['test'][k]:.3g}\n"
                           f"knobs the data can't see: size {h['hidden'][k]:.2f}")
        legA1.get_texts()[2 * j].set_text(f"{LABEL[n]}: test error {h['test'][k]:.3g}")

        r = rob[n]
        sm = moving_avg(r["success"][:k])
        B_succ[n].set_data(np.arange(1, k + 1), sm)
        B_dot[n].set_data([k] if k else [], [sm[-1]] if k else [])
        pl, pr_ = r["pr"][:k + 1, 0], r["pr"][:k + 1, 1]
        B_lines[n][0].set_data(np.arange(k + 1), pr_)
        B_lines[n][1].set_data(np.arange(k + 1), pl)
        V0 = r["V0"]
        B_info[n].set_text(f"trial {k}/{N_TRIALS}   right light {pr_[-1]:.0%}, left light {pl[-1]:.0%}\n"
                           f"start V = [[{V0[0, 0]:+.2f}, {V0[0, 1]:+.2f}], [{V0[1, 0]:+.2f}, {V0[1, 1]:+.2f}]]"
                           f"  (rows: turn L, turn R)")
    sz = rob["zero"]["success"][:k]; sr = rob["random"]["success"][:k]
    infoA1.set_text(f"GD step {it}/{FRAMES * GD_PER_FRAME}")
    infoB1.set_text(f"trial {k}/{N_TRIALS}   successes so far: "
                    f"zero {int(sz.sum())}, random {int(sr.sum())}")
    fig.canvas.draw_idle()


# ---- controls: example picker, step slider, buttons ----
last = FRAMES
ex_slider = Slider(fig.add_axes([0.10, 0.105, 0.80, 0.025]), "example #", 0, N_EXAMPLES - 1,
                   valinit=0, valstep=1, color="#9b9a96")
slider = Slider(fig.add_axes([0.10, 0.065, 0.80, 0.025]), "step", 0, last, valinit=0, valstep=1)
slider.valtext.set_text("trial 0 / GD 0")
ex_slider.on_changed(lambda val: set_example(int(val)))
slider.on_changed(lambda val: (slider.valtext.set_text(f"trial {int(val)} / GD {int(val) * GD_PER_FRAME}"),
                               update(int(val))))


def go_to(k):
    slider.set_val(min(max(k, 0), last))   # triggers update()


btn_start = Button(fig.add_axes([0.30, 0.012, 0.08, 0.04]), "|< Start")
btn_prev = Button(fig.add_axes([0.39, 0.012, 0.08, 0.04]), "< Prev")
btn_play = Button(fig.add_axes([0.48, 0.012, 0.08, 0.04]), "Play")
btn_next = Button(fig.add_axes([0.57, 0.012, 0.08, 0.04]), "Next >")
btn_end = Button(fig.add_axes([0.66, 0.012, 0.08, 0.04]), "End >|")

btn_start.on_clicked(lambda e: go_to(0))
btn_prev.on_clicked(lambda e: go_to(int(slider.val) - 1))
btn_next.on_clicked(lambda e: go_to(int(slider.val) + 1))
btn_end.on_clicked(lambda e: go_to(last))

# play/pause: a timer advances the slider one step at a time
timer = fig.canvas.new_timer(interval=60)
playing = False


def tick():
    if int(slider.val) >= last:
        toggle_play(None)                  # reached the end -> pause
    else:
        go_to(int(slider.val) + 1)


def toggle_play(event):
    global playing
    playing = not playing
    if playing:
        if int(slider.val) >= last:
            go_to(0)                       # restart from the beginning
        timer.start()
    else:
        timer.stop()
    btn_play.label.set_text("Pause" if playing else "Play")
    fig.canvas.draw_idle()


timer.add_callback(tick)
btn_play.on_clicked(toggle_play)

set_example(0)
plt.show()
