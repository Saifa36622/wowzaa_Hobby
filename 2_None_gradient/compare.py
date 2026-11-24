"""
Gradient descent vs. CEM vs. CMA-ES on the same problem.

THE SYSTEM (robotics flavor: system identification)
    A robot joint vibrates after a disturbance. A sensor records its angle:
        angle(t) = A * sin(w * t + phi) + noise
    We know the measured angles (the "answers"), but not the three knobs
        theta = [A, w, phi] = [amplitude, frequency, phase]
    Loss = mean squared error between the model and the measurements.

    Why this problem: the loss is bumpy in the frequency w, so it has
    LOCAL MINIMA. Start at the wrong frequency and a single walker can get
    stuck fitting a "wrong-frequency" wave. That is exactly where the three
    methods differ.

FAIRNESS
    Every method starts each run from the SAME random initial guess (one per
    seed). We repeat over many seeds and report the median and spread.
    We also count "loss evaluations" (compute budget), because a population
    method evaluates many candidates per iteration while gradient descent
    evaluates one point (plus one gradient, counted as 1 extra evaluation,
    since a backward pass costs roughly as much as a forward pass).

Run:  python compare_optimizers.py
Needs: numpy, matplotlib
Outputs: comparison.png, internals.png, a step-by-step log in the console,
         and an interactive window (slider + buttons) to scrub through training.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. The system: data and loss
# ---------------------------------------------------------------------------
TRUE_THETA = np.array([1.5, 2.0, 0.5])            # A, w, phi (unknown to the optimizers)
NAMES = ["A (amplitude)", "w (frequency)", "phi (phase)"]

rng_data = np.random.default_rng(0)
T = np.linspace(0, 6, 60)                        # 60 sensor samples over 6 s
Y = TRUE_THETA[0] * np.sin(TRUE_THETA[1] * T + TRUE_THETA[2]) \
    + 0.1 * rng_data.standard_normal(T.size)     # measured angles = the answers


def model(theta, t=T):
    A, w, phi = theta
    return A * np.sin(w * t + phi)


def loss(theta):
    """How wrong is this guess? One number, lower is better."""
    return np.mean((model(theta) - Y) ** 2)


def gradient(theta):
    """Slope of the loss for each knob: mean of 2 * error * (effect of knob)."""
    A, w, phi = theta
    err = model(theta) - Y
    s, c = np.sin(w * T + phi), np.cos(w * T + phi)
    dA = 2 * np.mean(err * s)              # effect of A on output:   sin(...)
    dw = 2 * np.mean(err * A * c * T)      # effect of w on output:   A cos(...) t
    dphi = 2 * np.mean(err * A * c)        # effect of phi on output: A cos(...)
    return np.array([dA, dw, dphi])


# The level a "correct" fit reaches is about the noise level (0.1^2 = 0.01).
SOLVED = 0.02


# ---------------------------------------------------------------------------
# 2. The three optimizers. Each returns a history dict with the same fields
#    plus its own internal parameters, logged at EVERY iteration.
# ---------------------------------------------------------------------------
def run_gradient_descent(theta0, iters, lr=0.02):
    """
    One walker that feels the slope.
    Important parameters:
        lr  learning rate (step size). Too big -> diverge, too small -> slow.
    Logged internals: the gradient vector and its size (norm).
    """
    theta = theta0.copy()
    h = dict(loss=[], evals=[], theta=[], grad_norm=[], grad=[])
    evals = 0
    for _ in range(iters):
        g = gradient(theta)
        evals += 2                                  # 1 loss + 1 gradient
        theta = theta - lr * g                      # step downhill
        h["loss"].append(loss(theta)); h["evals"].append(evals)
        h["theta"].append(theta.copy())
        h["grad"].append(g.copy()); h["grad_norm"].append(np.linalg.norm(g))
    return h


def run_cem(theta0, iters, pop=20, n_elite=5, std0=1.0, alpha=0.7,
            min_std=1e-3, seed=0):
    """
    A crowd of scouts: sample, keep the best, regroup around them.
    Important parameters:
        pop      how many guesses per iteration
        n_elite  how many of the best guesses to keep
        std0     initial spread of the crowd (one per knob)
        alpha    smoothing: how much of the new mean/std to accept
                 (alpha = 1 is "refit from scratch", which can collapse early)
        min_std  floor on the spread so the crowd never fully freezes
    Logged internals: mean (best guess), per-knob std (crowd spread).
    """
    rng = np.random.default_rng(seed)
    mean = theta0.copy()
    std = np.full(3, std0)
    h = dict(loss=[], evals=[], theta=[], std=[], best=[])
    evals, best = 0, np.inf
    for _ in range(iters):
        X = mean + std * rng.standard_normal((pop, 3))  # 1. scatter the crowd
        F = np.array([loss(x) for x in X])              # 2. score everyone
        evals += pop
        elites = X[np.argsort(F)[:n_elite]]             # 3. keep the best
        mean = alpha * elites.mean(0) + (1 - alpha) * mean          # 4. regroup
        std = alpha * elites.std(0) + (1 - alpha) * std
        std = np.maximum(std, min_std)
        best = min(best, F.min())
        h["loss"].append(loss(mean)); h["evals"].append(evals)
        h["theta"].append(mean.copy()); h["std"].append(std.copy())
        h["best"].append(best)
    return h


def run_cmaes(theta0, iters, pop=20, sigma0=1.0, seed=0):
    """
    CEM with memory and a learned crowd shape (standard CMA-ES, written out).
    Important parameters:
        pop     (lambda) guesses per iteration
        mu      how many best guesses update the mean (pop // 2)
        sigma0  initial overall step size
    Logged internals:
        sigma   overall step size (grows when moves agree, shrinks when they cancel)
        axes    lengths of the search ellipse = sigma * sqrt(eigenvalues of C)
                -> very different lengths = the crowd has stretched along a
                   direction that matters (it learned the landscape's shape)
    """
    rng = np.random.default_rng(seed)
    n = 3
    mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    # standard default learning rates for the distribution
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    mean, sigma = theta0.copy(), sigma0
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    h = dict(loss=[], evals=[], theta=[], sigma=[], axes=[], best=[])
    evals, best = 0, np.inf
    for g in range(iters):
        evals_g, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals_g, 1e-20))
        Z = rng.standard_normal((pop, n))
        Yd = (Z * D) @ B.T                         # directions shaped by C
        X = mean + sigma * Yd                      # 1. scatter the (tilted) crowd
        F = np.array([loss(x) for x in X])         # 2. score everyone
        evals += pop
        order = np.argsort(F)
        Ysel = Yd[order[:mu]]
        yw = w @ Ysel                              # weighted step of the best
        mean = mean + sigma * yw                   # 3. move the center
        # 4a. step-size control (memory of past moves)
        invsqrtC = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) \
            < (1.4 + 2 / (n + 1)) * chiN
        # 4b. shape control (learn tilt / stretch of the crowd)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        rank_mu = (Ysel * w[:, None]).T @ Ysel
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc)
             + (1 - hsig) * cc * (2 - cc) * C) + cmu * rank_mu
        C = (C + C.T) / 2
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        best = min(best, F[order[0]])
        h["loss"].append(loss(mean)); h["evals"].append(evals)
        h["theta"].append(mean.copy()); h["sigma"].append(sigma)
        h["axes"].append(sigma * np.sqrt(np.sort(np.linalg.eigvalsh(C))[::-1]))
        h["best"].append(best)
    return h


# ---------------------------------------------------------------------------
# 3. Run all three from the same starts, many seeds
# ---------------------------------------------------------------------------
GD_ITERS = 600          # GD takes small cheap steps -> more iterations
POP_ITERS = 60          # population methods: 60 iterations x 20 guesses = 1200 evals
N_SEEDS = 20

rng_init = np.random.default_rng(42)
starts = np.column_stack([
    rng_init.uniform(0.2, 3.0, N_SEEDS),       # A
    rng_init.uniform(0.5, 5.0, N_SEEDS),       # w   (true value is 2.0)
    rng_init.uniform(-np.pi, np.pi, N_SEEDS),  # phi
])

results = {"Gradient descent": [], "CEM": [], "CMA-ES": []}
for s, th0 in enumerate(starts):
    results["Gradient descent"].append(run_gradient_descent(th0, GD_ITERS))
    results["CEM"].append(run_cem(th0, POP_ITERS, seed=s))
    results["CMA-ES"].append(run_cmaes(th0, POP_ITERS, seed=s))


# ---------------------------------------------------------------------------
# 4. Step-by-step console log for one seed (shows each method's internals)
# ---------------------------------------------------------------------------
LOG_SEED = 0
th0 = starts[LOG_SEED]
print(f"\nTrue theta  [A, w, phi] = {TRUE_THETA}")
print(f"Start theta [A, w, phi] = {np.round(th0, 3)}   (seed {LOG_SEED}, same for all)\n")

gd = results["Gradient descent"][LOG_SEED]
print("GRADIENT DESCENT  (lr = 0.02)")
print(f"{'iter':>5} {'evals':>6} {'loss':>9} {'A':>7} {'w':>7} {'phi':>7} {'|grad|':>9}")
for i in list(range(0, 10)) + list(range(19, GD_ITERS, 60)) + [GD_ITERS - 1]:
    t = gd["theta"][i]
    print(f"{i+1:>5} {gd['evals'][i]:>6} {gd['loss'][i]:>9.4f} "
          f"{t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f} {gd['grad_norm'][i]:>9.4f}")

cem = results["CEM"][LOG_SEED]
print("\nCEM  (pop = 20, elites = 5, alpha = 0.7)")
print(f"{'iter':>5} {'evals':>6} {'loss':>9} {'A':>7} {'w':>7} {'phi':>7} "
      f"{'std_A':>7} {'std_w':>7} {'std_phi':>7}")
for i in list(range(0, 10)) + list(range(14, POP_ITERS, 10)) + [POP_ITERS - 1]:
    t, sd = cem["theta"][i], cem["std"][i]
    print(f"{i+1:>5} {cem['evals'][i]:>6} {cem['loss'][i]:>9.4f} "
          f"{t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f} {sd[0]:>7.3f} {sd[1]:>7.3f} {sd[2]:>7.3f}")

cma = results["CMA-ES"][LOG_SEED]
print("\nCMA-ES  (pop = 20, mu = 10, sigma0 = 1.0)")
print(f"{'iter':>5} {'evals':>6} {'loss':>9} {'A':>7} {'w':>7} {'phi':>7} "
      f"{'sigma':>7} {'axis1':>8} {'axis3':>8} {'stretch':>8}")
for i in list(range(0, 10)) + list(range(14, POP_ITERS, 10)) + [POP_ITERS - 1]:
    t, ax = cma["theta"][i], cma["axes"][i]
    print(f"{i+1:>5} {cma['evals'][i]:>6} {cma['loss'][i]:>9.4f} "
          f"{t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f} {cma['sigma'][i]:>7.3f} "
          f"{ax[0]:>8.4f} {ax[2]:>8.4f} {ax[0]/ax[2]:>8.1f}")

print("\nSUMMARY over", N_SEEDS, "random starts")
print(f"{'method':<18} {'solved':>8} {'median final loss':>18} {'evals used':>11}")
for name, runs in results.items():
    final = np.array([r["loss"][-1] for r in runs])
    print(f"{name:<18} {np.sum(final < SOLVED):>5}/{N_SEEDS} {np.median(final):>18.4f} "
          f"{runs[0]['evals'][-1]:>11}")
print(f"(solved = final loss < {SOLVED}, i.e. found the true frequency)")
print("Note: A < 0 with phi shifted by pi is the SAME wave (-A sin(x) = A sin(x + pi)),")
print("      so the problem has two equally good answers. Both count as solved.\n")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Gradient descent": "#2a78d6", "CEM": "#eb6834", "CMA-ES": "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False,
})


def band(ax, x, runs, key, name):
    M = np.array([r[key] for r in runs])
    med, lo, hi = np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)
    ax.fill_between(x, lo, hi, color=COL[name], alpha=0.15, linewidth=0)
    ax.plot(x, med, color=COL[name], linewidth=2, label=name)
    return med


fig, axs = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Same problem, same starts: gradient descent vs. CEM vs. CMA-ES",
             fontsize=13, x=0.02, ha="left")

# (a) loss vs iteration
ax = axs[0, 0]
for name, runs in results.items():
    x = np.arange(1, len(runs[0]["loss"]) + 1)
    med = band(ax, x, runs, "loss", name)
    ax.annotate(name, (x[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
ax.set_yscale("log"); ax.set_xscale("log")
ax.axhline(SOLVED, color=INK2, linewidth=1, linestyle=":")
ax.set_title("Loss at each iteration (median of 20 starts, band = middle 50%)", loc="left")
ax.set_xlabel("Iteration (log scale)"); ax.set_ylabel("Loss (log scale)")
ax.text(1.0, SOLVED * 1.15, "solved", color=INK2, fontsize=9)

# (b) loss vs evaluations
ax = axs[0, 1]
for name, runs in results.items():
    band(ax, np.array(runs[0]["evals"]), runs, "loss", name)
ax.set_yscale("log"); ax.set_xscale("log")
ax.axhline(SOLVED, color=INK2, linewidth=1, linestyle=":")
ax.text(1.5, SOLVED * 1.15, "solved", color=INK2, fontsize=9)
ax.set_title("Same thing, measured in compute: loss vs. loss evaluations", loc="left")
ax.set_xlabel("Loss evaluations used (log scale)"); ax.set_ylabel("Loss (log scale)")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))

# (c) final loss per start
ax = axs[1, 0]
jit = np.random.default_rng(1)
for k, (name, runs) in enumerate(results.items()):
    final = np.array([r["loss"][-1] for r in runs])
    ax.scatter(k + jit.uniform(-0.12, 0.12, final.size), final, s=46,
               color=COL[name], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(k, final.max() * 1.6, f"{np.sum(final < SOLVED)}/{N_SEEDS} solved",
            ha="center", color=INK, fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(results.keys())
ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 4)
ax.axhline(SOLVED, color=INK2, linewidth=1, linestyle=":")
ax.set_title("Final loss for each of the 20 starts (dots above the line = stuck)", loc="left")
ax.set_ylabel("Final loss (log scale)")

# (d) data + fits for the start where gradient descent did worst
worst = int(np.argmax([r["loss"][-1] for r in results["Gradient descent"]]))
ax = axs[1, 1]
tt = np.linspace(0, 6, 400)
ax.scatter(T, Y, s=18, color=INK2, zorder=3, label="Measured angle (the answers)")
for name, runs in results.items():
    th = runs[worst]["theta"][-1]
    ax.plot(tt, model(th, tt), color=COL[name], linewidth=2,
            label=f"{name}: w = {th[1]:.2f}")
ax.set_title(f"Fitted curves for start #{worst} (start w = {starts[worst,1]:.2f}, true w = 2.00)",
             loc="left")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Joint angle (rad)")
ax.set_ylim(-1.9, 3.0)
ax.legend(loc="upper center", ncol=2, fontsize=9)

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("comparison.png", dpi=150)

# ---- internals of each method, one start ----
s = LOG_SEED
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"What each method tracks internally (start #{s}, same for all three)",
             fontsize=13, x=0.02, ha="left")

ax = axs[0, 0]
ax.plot(np.arange(1, GD_ITERS + 1), gd["grad_norm"], color=COL["Gradient descent"], linewidth=2)
ax.set_yscale("log")
ax.set_title("Gradient descent: size of the gradient |grad| (slope felt)", loc="left")
ax.set_xlabel("Iteration"); ax.set_ylabel("|grad| (log scale)")

ax = axs[0, 1]
labels = ["std of A", "std of w", "std of phi"]
styles = ["-", "--", ":"]
for k in range(3):
    ax.plot(np.arange(1, POP_ITERS + 1), np.array(cem["std"])[:, k], color=COL["CEM"],
            linestyle=styles[k], linewidth=2, label=labels[k])
ax.set_yscale("log")
ax.set_title("CEM: spread of the crowd for each knob", loc="left")
ax.set_xlabel("Iteration"); ax.set_ylabel("Std (log scale)"); ax.legend()

ax = axs[1, 0]
A = np.array(cma["axes"])
ax.plot(np.arange(1, POP_ITERS + 1), cma["sigma"], color=INK2, linewidth=2,
        label="sigma (overall step size)")
for k, st in enumerate(styles):
    ax.plot(np.arange(1, POP_ITERS + 1), A[:, k], color=COL["CMA-ES"], linestyle=st,
            linewidth=2, label=f"ellipse axis {k+1}")
ax.set_yscale("log")
ax.set_title("CMA-ES: step size and the 3 axes of its search ellipse", loc="left")
ax.set_xlabel("Iteration"); ax.set_ylabel("Length (log scale)"); ax.legend()

ax = axs[1, 1]
for name, h in [("Gradient descent", gd), ("CEM", cem), ("CMA-ES", cma)]:
    w_traj = np.array(h["theta"])[:, 1]
    ax.plot(np.arange(1, len(w_traj) + 1), w_traj, color=COL[name], linewidth=2, label=name)
ax.axhline(TRUE_THETA[1], color=INK2, linewidth=1, linestyle=":")
ax.text(1.1, TRUE_THETA[1] + 0.08, "true w = 2.0", color=INK2, fontsize=9)
ax.set_xscale("log")
ax.set_title("Frequency estimate w over iterations", loc="left")
ax.set_xlabel("Iteration (log scale)"); ax.set_ylabel("w (rad/s)"); ax.legend()

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("internals.png", dpi=150)
print("Saved comparison.png and internals.png")

plt.close("all")   # PNGs are saved; only the interactive window below is shown


# ---------------------------------------------------------------------------
# 6. Interactive view: scrub through training with a slider and buttons
#    The time axis is LOSS EVALUATIONS, so at every frame all three methods
#    have used the same compute (GD: 2 evals/iter, CEM & CMA-ES: 20 evals/iter).
# ---------------------------------------------------------------------------
from matplotlib.widgets import Slider, Button

BUDGET = GD_ITERS * 2                          # 1200 evals, same for all three
frame_evals = np.arange(2, BUDGET + 1, 2)      # one frame per GD iteration
last = len(frame_evals) - 1
tt = np.linspace(0, 6, 400)

# loss with the best amplitude A for each (w, phi): shows the bumps in w
WW, PP = np.meshgrid(np.linspace(0.3, 5.2, 150), np.linspace(-np.pi / 2, np.pi / 2, 120))
S = np.sin(WW[..., None] * T + PP[..., None])
A_best = (S * Y).sum(-1) / (S * S).sum(-1)
LP = np.mean((A_best[..., None] * S - Y) ** 2, -1)


def until(h, th0, evals):
    """Start point + every iterate finished within `evals` loss evaluations."""
    i = np.searchsorted(h["evals"], evals, side="right")
    return ([0] + h["evals"][:i], [loss(th0)] + h["loss"][:i],
            np.array([th0] + h["theta"][:i]))


def w_phi_path(thetas):
    """(w, phi) with phi folded into [-pi/2, pi/2] (A -> -A, phi -> phi+pi is
    the same wave). NaNs break the line where phi wraps around."""
    w, phi = thetas[:, 1], (thetas[:, 2] + np.pi / 2) % np.pi - np.pi / 2
    cut = np.where(np.abs(np.diff(phi)) > np.pi / 2)[0] + 1
    return np.insert(w, cut, np.nan), np.insert(phi, cut, np.nan)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6.5))
fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.30, wspace=0.22)
title = fig.suptitle("", fontsize=13, x=0.02, ha="left")
box = dict(fc=SURFACE, ec="none", alpha=0.85)

# 1. loss going down over compute
ax1.set_yscale("log")
ax1.set_xlim(0, BUDGET)
ax1.axhline(SOLVED, color=INK2, linewidth=1, linestyle=":")
ax1.set_title("Loss vs. loss evaluations", loc="left")
ax1.set_xlabel("Loss evaluations used"); ax1.set_ylabel("Loss (log scale)")
loss_line = {n: ax1.plot([], [], color=COL[n], linewidth=2, label=n)[0] for n in results}
loss_dot = {n: ax1.plot([], [], "o", color=COL[n])[0] for n in results}
leg1 = ax1.legend(loc="upper right")
info1 = ax1.text(0.02, 0.04, "", transform=ax1.transAxes, bbox=box)

# 2. the fitted wave getting closer to the measurements
ax2.scatter(T, Y, s=18, color=INK2, zorder=3, label="Measured angle")
fit_line = {n: ax2.plot(tt, 0 * tt, color=COL[n], linewidth=2, label=n)[0] for n in results}
ax2.set_xlim(0, 6); ax2.set_ylim(-3, 3.5)
ax2.set_title("Fitted curve (current guess of each method)", loc="left")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Joint angle (rad)")
leg2 = ax2.legend(loc="upper center", ncol=2, fontsize=9)
info2 = ax2.text(0.98, 0.04, "", transform=ax2.transAxes, ha="right", bbox=box)

# 3. path of (w, phi) across the loss surface
ax3.contour(WW, PP, LP, levels=np.logspace(-2, 0.5, 14), cmap="viridis", linewidths=0.8)
ax3.plot(TRUE_THETA[1], TRUE_THETA[2], "k*", markersize=13, label="true (2.0, 0.5)")
path_line = {n: ax3.plot([], [], ".-", color=COL[n], markersize=3, linewidth=1.2, label=n)[0]
             for n in results}
path_dot = {n: ax3.plot([], [], "o", color=COL[n])[0] for n in results}
start_dot, = ax3.plot([], [], "kx", markersize=9, label="start")
ax3.set_xlim(WW.min(), WW.max()); ax3.set_ylim(-np.pi / 2, np.pi / 2)
ax3.grid(False)
ax3.set_title("Path on loss surface (w, phi), best A at each point", loc="left")
ax3.set_xlabel("w (frequency)"); ax3.set_ylabel("phi (phase, folded)")
leg3 = ax3.legend(loc="upper right", fontsize=9)
info3 = ax3.text(0.98, 0.04, "", transform=ax3.transAxes, ha="right", bbox=box)

seed = 0


def set_seed(sd):
    global seed
    seed = sd
    th0 = starts[sd]
    allL = np.concatenate([results[n][sd]["loss"] for n in results] + [[loss(th0)]])
    allL = allL[np.isfinite(allL)]
    ax1.set_ylim(min(allL.min(), SOLVED) / 2, allL.max() * 3)
    w0, p0 = w_phi_path(th0[None])
    start_dot.set_data(w0, p0)
    title.set_text(f"Start #{sd}: A = {th0[0]:.2f}, w = {th0[1]:.2f}, phi = {th0[2]:.2f}"
                   f"   (true: 1.50, 2.00, 0.50)  - same start for all three")
    update(int(slider.val))


def update(k):
    ev = frame_evals[k]
    th0 = starts[seed]
    for j, n in enumerate(results):
        h = results[n][seed]
        evs, Ls, ths = until(h, th0, ev)
        loss_line[n].set_data(evs, Ls)
        loss_dot[n].set_data([evs[-1]], [Ls[-1]])
        fit_line[n].set_ydata(model(ths[-1], tt))
        pw, pp = w_phi_path(ths)
        path_line[n].set_data(pw, pp)
        path_dot[n].set_data([pw[-1]], [pp[-1]])
        th = ths[-1]
        leg1.get_texts()[j].set_text(f"{n}: loss {Ls[-1]:.3g}")
        leg2.get_texts()[j + 1].set_text(f"{n}: w = {th[1]:.2f}")
        leg3.get_texts()[j + 1].set_text(f"{n}: ({th[1]:.2f}, {pp[-1]:.2f})")
    gd_it, pop_it = ev // 2, ev // 20
    txt = f"evals {ev}/{BUDGET}   GD iter {gd_it}, CEM/CMA-ES iter {pop_it}"
    for info in (info1, info2, info3):
        info.set_text(txt)
    fig.canvas.draw_idle()


# ---- controls: start picker, step slider, buttons ----
worst = int(np.argmax([r["loss"][-1] for r in results["Gradient descent"]]))
seed_slider = Slider(fig.add_axes([0.10, 0.17, 0.80, 0.03]), "start #", 0, N_SEEDS - 1,
                     valinit=worst, valstep=1, color="#9b9a96")
slider = Slider(fig.add_axes([0.10, 0.11, 0.80, 0.03]), "step", 0, last,
                valinit=0, valstep=1)
slider.valtext.set_text(f"{frame_evals[0]} evals")
seed_slider.on_changed(lambda val: set_seed(int(val)))
slider.on_changed(lambda val: (slider.valtext.set_text(f"{frame_evals[int(val)]} evals"),
                               update(int(val))))


def go_to(k):
    slider.set_val(min(max(k, 0), last))   # triggers update()


btn_start = Button(fig.add_axes([0.30, 0.025, 0.08, 0.05]), "|< Start")
btn_prev = Button(fig.add_axes([0.39, 0.025, 0.08, 0.05]), "< Prev")
btn_play = Button(fig.add_axes([0.48, 0.025, 0.08, 0.05]), "Play")
btn_next = Button(fig.add_axes([0.57, 0.025, 0.08, 0.05]), "Next >")
btn_end = Button(fig.add_axes([0.66, 0.025, 0.08, 0.05]), "End >|")

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

set_seed(worst)
plt.show()
