"""
Why do neural networks use gradients instead of CEM / CMA-ES?

Experiment: the SAME smooth problem with more and more knobs, and the SAME
compute budget (number of loss evaluations) for every method.

    Problem: linear regression with n knobs  (a stand-in for "a network with n weights")
        loss(theta) = mean( (X @ theta - y)^2 )
    It is smooth and has one valley, like the region a neural network trains in.

    We report the RELATIVE loss: final loss / starting loss.
        1.0  = no progress at all
        0.0  = solved perfectly

Part 2 measures the core reason: how well does a RANDOM direction point downhill
when there are n knobs?

Run:  python why_gradients_scale.py
Needs: numpy, matplotlib
Outputs: why_gradients_scale.png, tables in the console,
         and an interactive window (slider + buttons) to scrub through training.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

BUDGET = 2000                       # loss evaluations allowed per method
SIZES = [3, 10, 100, 1000]          # number of knobs


def make_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    m = 2 * n + 20                              # number of data points (answers)
    X = rng.standard_normal((m, n)) / np.sqrt(n)
    true = rng.standard_normal(n)
    y = X @ true
    loss = lambda th: np.mean((X @ th - y) ** 2)
    grad = lambda th: 2 * X.T @ (X @ th - y) / m
    L = 2 * np.linalg.eigvalsh(X.T @ X / m).max()   # steepest curvature
    return loss, grad, np.zeros(n), L, X, y, true


# ---------------------------------------------------------------- optimizers
# Each returns a history: loss and knobs after every update, with the number
# of loss evaluations used so far (entry 0 = the start, 0 evaluations).
def new_history(loss, th0):
    return dict(evals=[0], loss=[loss(th0)], theta=[th0.copy()])


def log(h, evals, loss, th):
    h["evals"].append(evals); h["loss"].append(loss(th)); h["theta"].append(th.copy())


def gd(loss, grad, th0, L):
    th, lr = th0.copy(), 1.0 / L                 # a safe, standard step size
    h = new_history(loss, th0)
    for k in range(BUDGET // 2):                 # 1 loss + 1 gradient per step
        th -= lr * grad(th)
        log(h, 2 * (k + 1), loss, th)
    return h


def cem(loss, th0, pop=20, n_elite=5, alpha=0.7, seed=0):
    rng = np.random.default_rng(seed)
    mean, std = th0.copy(), np.ones(th0.size)
    h = new_history(loss, th0)
    for g in range(BUDGET // pop):
        Xs = mean + std * rng.standard_normal((pop, th0.size))
        F = np.array([loss(x) for x in Xs])
        el = Xs[np.argsort(F)[:n_elite]]
        mean = alpha * el.mean(0) + (1 - alpha) * mean
        std = np.maximum(alpha * el.std(0) + (1 - alpha) * std, 1e-3)
        log(h, pop * (g + 1), loss, mean)
    return h


def cmaes(loss, th0, sigma0=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = th0.size
    pop = 4 + int(3 * np.log(n)); mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    mean, sigma, C = th0.copy(), sigma0, np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    h = new_history(loss, th0)
    for g in range(BUDGET // pop):
        ev, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(ev, 1e-20))
        Yd = (rng.standard_normal((pop, n)) * D) @ B.T
        F = np.array([loss(mean + sigma * y) for y in Yd])
        Ys = Yd[np.argsort(F)[:mu]]; yw = w @ Ys
        mean = mean + sigma * yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / D))
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chiN
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) \
            + cmu * (Ys * w[:, None]).T @ Ys
        C = (C + C.T) / 2
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        log(h, pop * (g + 1), loss, mean)
    return h


# ------------------------------------------------------- part 1: scale it up
print(f"Same smooth problem, same budget ({BUDGET} loss evaluations).")
print("Relative loss = final / start   (1.0 = no progress, 0.0 = solved)\n")
print(f"{'knobs':>6} {'Gradient':>10} {'CEM':>10} {'CMA-ES':>10} {'CMA-ES time':>12}")
rel = {"Gradient descent": [], "CEM": [], "CMA-ES": []}
runs, problems = {}, {}              # per size: histories, and (X, y, true, start loss)
for n in SIZES:
    loss, grad, th0, L, X, y, true = make_problem(n)
    l0 = loss(th0)
    runs[n] = {"Gradient descent": gd(loss, grad, th0, L), "CEM": cem(loss, th0)}
    t = time.time(); runs[n]["CMA-ES"] = cmaes(loss, th0); dt = time.time() - t
    problems[n] = (X, y, true, l0)
    for name in rel:
        rel[name].append(runs[n][name]["loss"][-1] / l0)
    print(f"{n:>6} {rel['Gradient descent'][-1]:>10.2e} {rel['CEM'][-1]:>10.2e} "
          f"{rel['CMA-ES'][-1]:>10.2e} {dt:>10.1f} s")

print("\nCMA-ES must store an n x n covariance matrix (8 bytes per number):")
for n, label in [(1_000, "small MLP"), (100_000, "small CNN"),
                 (10_000_000, "mid-size network"), (1_000_000_000, "1B-parameter model")]:
    gb = 8 * n * n / 1e9
    print(f"  {n:>13,} knobs ({label:<18}) -> {gb:>16,.3f} GB")


# ------------------------------------- part 2: how good is a random direction?
print("\nHow well does a RANDOM direction line up with the true downhill direction?")
print("(1.0 = perfectly downhill, 0.0 = sideways, useless)")
rng = np.random.default_rng(1)
dims = [3, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
align = []
for n in dims:
    g = rng.standard_normal(n); g /= np.linalg.norm(g)
    cos = [abs(g @ (d := rng.standard_normal(n)) / np.linalg.norm(d)) for _ in range(200)]
    align.append(np.mean(cos))
    print(f"  {n:>9,} knobs: average alignment {align[-1]:.4f}   (theory ~ {np.sqrt(2/np.pi/n):.4f})")


# ------------------------------------------------------------------- plot
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Gradient descent": "#2a78d6", "CEM": "#eb6834", "CMA-ES": "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False,
})
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axs[0]
FLOOR = 1e-16                                   # computer precision limit
for name, v in rel.items():
    v = np.maximum(v, FLOOR)                    # log axis can't show exactly 0
    ax.plot(SIZES, v, color=COL[name], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    dy = {"Gradient descent": 10, "CEM": -10, "CMA-ES": 10}[name]
    ax.annotate(name, (SIZES[-1], v[-1]), xytext=(8, dy), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
ax.text(SIZES[0], 1.6, "no progress", color=INK2, fontsize=9)
ax.axhline(FLOOR, color=INK2, linewidth=1, linestyle=":")
ax.text(SIZES[0], FLOOR * 3, "solved (computer precision limit)", color=INK2, fontsize=9)
ax.set_ylim(FLOOR / 10, 30)
ax.set_xlim(2, 6000)
ax.set_title(f"Relative loss after {BUDGET} evaluations, as knobs increase", loc="left")
ax.set_xlabel("Number of knobs (log scale)"); ax.set_ylabel("Final / starting loss (log scale)")
ax.legend(loc="center left")

ax = axs[1]
ax.plot(dims, align, color=COL["CEM"], linewidth=2, marker="o", markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.5)
for n, a in zip(dims, align):
    if n in (3, 1_000, 1_000_000):
        ax.annotate(f"{a:.3f}", (n, a), xytext=(6, 6), textcoords="offset points",
                    color=INK2, fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("How downhill is a random guess? (gradient is always 1.0)", loc="left")
ax.set_xlabel("Number of knobs (log scale)"); ax.set_ylabel("Alignment with downhill (log scale)")

fig.tight_layout()
fig.savefig("why_gradients_scale.png", dpi=150)
print("\nSaved why_gradients_scale.png")

plt.close("all")   # PNG is saved; only the interactive window below is shown


# ---------------------------------------------------------------------------
# Interactive view: scrub through training with a slider and buttons
#    The time axis is LOSS EVALUATIONS, so at every frame all three methods
#    have used the same compute.
# ---------------------------------------------------------------------------
from matplotlib.widgets import Slider, Button

frame_evals = np.arange(0, BUDGET + 1, 2)      # one frame per GD step
last = len(frame_evals) - 1


def upto(h, evals):
    """Index of the last update finished within `evals` loss evaluations."""
    return np.searchsorted(h["evals"], evals, side="right") - 1


# top row: the two loss charts; bottom row: one prediction panel per method
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 6, left=0.05, right=0.98, top=0.92, bottom=0.19,
                      wspace=0.6, hspace=0.32)
ax1, ax3 = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
pred_ax = {n: fig.add_subplot(gs[1, 2 * j:2 * j + 2]) for j, n in enumerate(rel)}
title = fig.suptitle("", fontsize=13, x=0.02, ha="left")
box = dict(fc=SURFACE, ec="none", alpha=0.85)

# 1. relative loss going down over compute
ax1.set_yscale("log")
ax1.set_xlim(0, BUDGET); ax1.set_ylim(FLOOR / 10, 30)
ax1.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
ax1.axhline(FLOOR, color=INK2, linewidth=1, linestyle=":")
ax1.text(BUDGET * 0.98, 1.6, "no progress", color=INK2, fontsize=9, ha="right")
ax1.text(BUDGET * 0.98, FLOOR * 3, "solved (precision limit)", color=INK2, fontsize=9, ha="right")
ax1.set_title("Relative loss vs. loss evaluations", loc="left")
ax1.set_xlabel("Loss evaluations used"); ax1.set_ylabel("Current / starting loss (log scale)")
loss_line = {n: ax1.plot([], [], color=COL[n], linewidth=2, label=n)[0] for n in rel}
loss_dot = {n: ax1.plot([], [], "o", color=COL[n])[0] for n in rel}
leg1 = ax1.legend(loc="center right")
info1 = ax1.text(0.98, 0.36, "", transform=ax1.transAxes, ha="right", va="top", bbox=box)

# 2. result of training, one panel per method: prediction vs. answer
#    (perfect fit = every dot on the diagonal; no progress = flat line at 0)
diag, pred_pts, info2 = {}, {}, {}
for n, ax in pred_ax.items():
    diag[n], = ax.plot([], [], color=INK2, linewidth=1, linestyle=":", label="perfect fit")
    pred_pts[n] = ax.scatter([], [], s=14, color=COL[n], alpha=0.6, edgecolor="none",
                             label="data point")
    ax.set_title(f"{n}: prediction vs. answer", loc="left", color=COL[n])
    ax.set_xlabel("Answer y"); ax.set_ylabel("Prediction X @ theta")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    info2[n] = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", bbox=box)

# 3. relative loss for every size at the current budget (the main result, growing)
scale_line = {n: ax3.plot([], [], color=COL[n], linewidth=2, marker="o", markersize=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, label=n)[0] for n in rel}
picked = ax3.axvline(SIZES[0], color=INK2, linewidth=1, linestyle="--")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlim(2, 2000); ax3.set_ylim(FLOOR / 10, 30)
ax3.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
ax3.axhline(FLOOR, color=INK2, linewidth=1, linestyle=":")
ax3.set_title("Relative loss at this budget, as knobs increase", loc="left")
ax3.set_xlabel("Number of knobs (log scale)  (dashed = shown below)")
ax3.set_ylabel("Current / starting loss (log scale)")
leg3 = ax3.legend(loc="center left", fontsize=9)
info3 = ax3.text(0.02, 0.36, "", transform=ax3.transAxes, va="top", bbox=box)

size = SIZES[0]


def set_size(i):
    global size
    size = SIZES[i]
    X, y, true, l0 = problems[size]
    lo, hi = y.min(), y.max(); pad = 0.15 * (hi - lo)
    lim = (lo - pad, hi + pad)
    for n, ax in pred_ax.items():
        ax.set_xlim(lim); ax.set_ylim(lim); diag[n].set_data(lim, lim)
    picked.set_xdata([size, size])
    pops = {n: runs[size][n]["evals"][1] for n in rel}
    title.set_text(f"{size} knobs, {X.shape[0]} data points, same start (all zeros) and "
                   f"budget for all three   (evals per update: GD {pops['Gradient descent']}, "
                   f"CEM {pops['CEM']}, CMA-ES {pops['CMA-ES']})")
    update(int(slider.val))


def update(k):
    ev = frame_evals[k]
    X, y, true, l0 = problems[size]
    for j, n in enumerate(rel):
        h = runs[size][n]
        i = upto(h, ev)
        r = np.maximum(np.array(h["loss"][:i + 1]) / l0, FLOOR)
        loss_line[n].set_data(h["evals"][:i + 1], r)
        loss_dot[n].set_data([h["evals"][i]], [r[-1]])
        th = h["theta"][i]
        pred = X @ th
        pred_pts[n].set_offsets(np.column_stack([y, pred]))
        err = np.linalg.norm(th - true) / np.linalg.norm(true)
        r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
        info2[n].set_text(f"evals {ev}/{BUDGET}   ({size} knobs)\n"
                          f"knobs off by {err:.0%}   R² = {r2:.3f}")
        scale = [max(runs[s][n]["loss"][upto(runs[s][n], ev)] / problems[s][3], FLOOR)
                 for s in SIZES]
        scale_line[n].set_data(SIZES, scale)
        leg1.get_texts()[j].set_text(f"{n}: {r[-1]:.2g}")
        leg3.get_texts()[j].set_text(f"{n}: {scale[-1]:.2g} at {SIZES[-1]} knobs")
    txt = f"evals {ev}/{BUDGET}"
    info1.set_text(f"{txt}   ({size} knobs)")
    info3.set_text(txt)
    fig.canvas.draw_idle()


# ---- controls: size picker, step slider, buttons ----
size_slider = Slider(fig.add_axes([0.10, 0.105, 0.80, 0.025]), "knobs", 0, len(SIZES) - 1,
                     valinit=0, valstep=1, color="#9b9a96")
size_slider.valtext.set_text(f"{SIZES[0]}")
slider = Slider(fig.add_axes([0.10, 0.065, 0.80, 0.025]), "step", 0, last,
                valinit=0, valstep=1)
slider.valtext.set_text(f"{frame_evals[0]} evals")
size_slider.on_changed(lambda val: (size_slider.valtext.set_text(f"{SIZES[int(val)]}"),
                                    set_size(int(val))))
slider.on_changed(lambda val: (slider.valtext.set_text(f"{frame_evals[int(val)]} evals"),
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
timer = fig.canvas.new_timer(interval=30)
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

set_size(0)
plt.show()
