"""
CMA-ES on its own, one iteration at a time.

Same system as compare.py (a robot joint: angle(t) = A sin(w t + phi) + noise,
find theta = [A, w, phi]), same 20 random starts, same CMA-ES settings and
seeds, so the runs match the CMA-ES runs in compare.py exactly.

This time every iteration logs the full search distribution, so you can watch:
    sampled candidates     the 20 guesses drawn this iteration
    elites (top half)      the 10 best guesses; they move the mean and reshape C
    mean + 2 sigma ellipse where the crowd is centred and how far it reaches
    C                      covariance matrix = learned shape / tilt of the crowd
                           (samples are drawn from  mean + sigma * N(0, C))

Run:  python cmaes.py
Needs: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------------------------------------------------------------
# 1. The system: data and loss (identical to compare.py)
# ---------------------------------------------------------------------------
TRUE_THETA = np.array([1.5, 2.0, 0.5])            # A, w, phi (unknown to the optimizer)
KNOBS = ["A", "w", "phi"]

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


def profile_loss(W, P):
    """Loss on a (w, phi) grid using the best amplitude A at each point."""
    S = np.sin(W[..., None] * T + P[..., None])
    A = (S * Y).sum(-1) / np.maximum((S * S).sum(-1), 1e-12)
    return np.mean((A[..., None] * S - Y) ** 2, -1)


SOLVED = 0.02                                    # about the noise level


# ---------------------------------------------------------------------------
# 2. CMA-ES, logging the whole search distribution at every iteration
# ---------------------------------------------------------------------------
def run_cmaes(theta0, iters, pop=20, sigma0=1.0, seed=0):
    """
    Same algorithm as compare.py. Per iteration we log the distribution the
    candidates were SAMPLED from (mean, sigma, C), the candidates themselves,
    their losses, which ones were elites, and where the mean moved to.
    """
    rng = np.random.default_rng(seed)
    n = 3
    mu = pop // 2                                # elites = top half
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    mean, sigma = theta0.copy(), sigma0
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    h = dict(mean=[], sigma=[], C=[], X=[], F=[], elite=[], new_mean=[],
             loss=[], best=[], axes=[])
    best = np.inf
    for g in range(iters):
        # log the distribution we are about to sample from
        h["mean"].append(mean.copy()); h["sigma"].append(sigma); h["C"].append(C.copy())
        h["loss"].append(loss(mean))
        h["axes"].append(sigma * np.sqrt(np.sort(np.maximum(np.linalg.eigvalsh(C), 0))[::-1]))

        evals_g, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals_g, 1e-20))
        Z = rng.standard_normal((pop, n))
        Yd = (Z * D) @ B.T                         # directions shaped by C
        X = mean + sigma * Yd                      # 1. scatter the (tilted) crowd
        F = np.array([loss(x) for x in X])         # 2. score everyone
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

        h["X"].append(X); h["F"].append(F); h["elite"].append(order[:mu])
        h["new_mean"].append(mean.copy()); h["best"].append(best)
    return h


# ---------------------------------------------------------------------------
# 3. Run from the same 20 starts as compare.py
# ---------------------------------------------------------------------------
ITERS = 60
POP = 20
N_SEEDS = 20
START = 11              # the start where gradient descent got stuck in compare.py

rng_init = np.random.default_rng(42)
starts = np.column_stack([
    rng_init.uniform(0.2, 3.0, N_SEEDS),       # A
    rng_init.uniform(0.5, 5.0, N_SEEDS),       # w   (true value is 2.0)
    rng_init.uniform(-np.pi, np.pi, N_SEEDS),  # phi
])
runs = [run_cmaes(th0, ITERS, pop=POP, seed=s) for s, th0 in enumerate(starts)]

print("CMA-ES final loss per start:")
for s, h in enumerate(runs):
    m = h["new_mean"][-1]
    print(f"  start #{s:>2}: loss {loss(m):.4f}   A={m[0]:6.3f} w={m[1]:6.3f} phi={m[2]:6.3f}"
          f"   {'solved' if loss(m) < SOLVED else 'stuck'}")


# ---------------------------------------------------------------------------
# 4. Interactive view (same style and controls as compare.py)
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GREEN, GREY = "#1baf7a", "#a3a29e"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False,
})
box = dict(fc=SURFACE, ec="none", alpha=0.85)
tt = np.linspace(0, 6, 400)
circle = np.array([np.cos(np.linspace(0, 2 * np.pi, 100)),
                   np.sin(np.linspace(0, 2 * np.pi, 100))])


def ellipse_2sigma(mean, sigma, C):
    """2-sigma ellipse of the sampling distribution, projected on (w, phi)."""
    cov = sigma ** 2 * C[1:, 1:]
    vals, vecs = np.linalg.eigh(cov)
    pts = mean[1:, None] + 2 * vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circle)
    return pts[0], pts[1]


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(17, 10.5))
fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.19, wspace=0.22, hspace=0.32)
title = fig.suptitle("", fontsize=13, x=0.02, ha="left")

# 1. loss of the mean, best so far, and this iteration's candidates
ax1.set_yscale("log"); ax1.set_xlim(0, ITERS + 1)
ax1.axhline(SOLVED, color=INK2, linewidth=1, linestyle=":")
ax1.set_title("Loss per iteration", loc="left")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("Loss (log scale)")
cand_F, = ax1.plot([], [], "o", color=GREY, markersize=4, label="sampled candidates")
elite_F, = ax1.plot([], [], "o", color=GREEN, markersize=4, label="elites (top half)")
mean_L, = ax1.plot([], [], color=GREEN, linewidth=2, label="loss of mean")
best_L, = ax1.plot([], [], color=INK2, linewidth=1.2, linestyle="--", label="best candidate so far")
ax1.legend(loc="upper right", fontsize=9)
info1 = ax1.text(0.02, 0.04, "", transform=ax1.transAxes, bbox=box)

# 2. the fitted wave: mean (bold) and every candidate (thin)
ax2.scatter(T, Y, s=18, color=INK2, zorder=4, label="Measured angle")
cand_curve = [ax2.plot(tt, 0 * tt, linewidth=1)[0] for _ in range(POP)]
mean_curve, = ax2.plot(tt, 0 * tt, color=INK, linewidth=2.5, zorder=5, label="mean")
ax2.plot([], [], color=GREY, label="sampled candidates")
ax2.plot([], [], color=GREEN, label="elites (top half)")
ax2.set_xlim(0, 6); ax2.set_ylim(-3.5, 4)
ax2.set_title("Fitted curve: mean and this iteration's candidates", loc="left")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Joint angle (rad)")
ax2.legend(loc="upper center", ncol=2, fontsize=9)
info2 = ax2.text(0.98, 0.04, "", transform=ax2.transAxes, ha="right", bbox=box)

# 3. step size sigma and the 3 axes of the search ellipse
ax3.set_yscale("log"); ax3.set_xlim(0, ITERS + 1)
ax3.set_title("Step size sigma and the 3 axes of the search ellipse", loc="left")
ax3.set_xlabel("Iteration"); ax3.set_ylabel("Length (log scale)")
sigma_L, = ax3.plot([], [], color=INK2, linewidth=2, label="sigma (overall step size)")
axis_L = [ax3.plot([], [], color=GREEN, linestyle=st, linewidth=2, label=f"ellipse axis {k+1}")[0]
          for k, st in enumerate(["-", "--", ":"])]
dots3, = ax3.plot([], [], "o", color=INK2, markersize=4)
ax3.legend(loc="lower left", fontsize=9)
info3 = ax3.text(0.98, 0.96, "", transform=ax3.transAxes, ha="right", va="top", bbox=box)


# 4 + 5. search space (w, phi): candidates, elites, mean, 2-sigma ellipse
def search_artists(ax):
    a = dict(
        path=ax.plot([], [], ".-", color=INK2, linewidth=0.8, markersize=3, label="mean path")[0],
        cand=ax.plot([], [], "o", color=GREY, markersize=5, label="sampled candidates")[0],
        elite=ax.plot([], [], "o", color=GREEN, markersize=5, label="elites (top half)")[0],
        ell=ax.plot([], [], color=GREEN, linewidth=2, label="2 sigma ellipse")[0],
        step=ax.plot([], [], color=INK, linewidth=1.5, linestyle="--")[0],
        mean=ax.plot([], [], "D", color=INK, markersize=7, label="mean", zorder=5)[0],
    )
    ax.plot(TRUE_THETA[1], TRUE_THETA[2], "*", color=INK, markersize=13, label="true (2.0, 0.5)")
    ax.grid(False)
    ax.set_xlabel("w (frequency)"); ax.set_ylabel("phi (phase)")
    return a


ax4.set_title("Search space (w, phi), best A at each point", loc="left")
S4 = search_artists(ax4)
ax4.legend(loc="upper right", fontsize=8)
info4 = ax4.text(0.98, 0.04, "", transform=ax4.transAxes, ha="right", bbox=box)

ax5.set_title("Zoomed on the crowd (dashed = where the mean moves next)", loc="left")
S5 = search_artists(ax5)
full_contour = zoom_contour = None
info5 = ax5.text(0.98, 0.04, "", transform=ax5.transAxes, ha="right", bbox=box)

# 6. covariance matrix C
im = ax6.imshow(np.eye(3), cmap="RdBu_r", vmin=-1, vmax=1)
ax6.set_xticks(range(3)); ax6.set_xticklabels(KNOBS)
ax6.set_yticks(range(3)); ax6.set_yticklabels(KNOBS)
ax6.grid(False)
C_txt = [[ax6.text(j, i, "", ha="center", va="center", fontsize=12) for j in range(3)]
         for i in range(3)]
ax6.set_title("Covariance C (shape of the crowd; red +, blue -)", loc="left")
info6 = ax6.text(0.5, -0.12, "", transform=ax6.transAxes, ha="center", va="top", bbox=box)

seed = START


def remove_contour(cs):
    if cs is None:
        return
    for c in cs.collections:     # (ContourSet.remove() only exists in matplotlib >= 3.8)
        c.remove()


def set_seed(sd):
    global seed
    seed = sd
    h, th0 = runs[sd], starts[sd]
    L = np.concatenate([np.concatenate(h["F"]), h["loss"]])
    ax1.set_ylim(min(L.min(), SOLVED) / 2, np.percentile(L, 99) * 3)
    lens = np.concatenate([h["sigma"], np.concatenate(h["axes"])])
    lens = lens[lens > 0]
    ax3.set_ylim(lens.min() / 2, lens.max() * 2)
    # overview: w in [0, 6], phi in [-4, 4], widened to fit this run's whole mean path
    # (phi can drift far, e.g. phi - 4*pi is the same wave)
    global full_contour
    path = np.array(h["mean"] + h["new_mean"][-1:])
    x0, x1 = min(0, path[:, 1].min() - 0.5), max(6, path[:, 1].max() + 0.5)
    y0, y1 = min(-4, path[:, 2].min() - 1), max(4, path[:, 2].max() + 1)
    ax4.set_xlim(x0, x1); ax4.set_ylim(y0, y1)
    remove_contour(full_contour)
    Wf, Pf = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 200))
    full_contour = ax4.contour(Wf, Pf, profile_loss(Wf, Pf), levels=np.logspace(-2, 0.5, 14),
                               cmap="viridis", linewidths=0.7, zorder=0)
    title.set_text(f"CMA-ES, start #{sd}: A = {th0[0]:.2f}, w = {th0[1]:.2f}, "
                   f"phi = {th0[2]:.2f}   (true: 1.50, 2.00, 0.50)   pop = {POP}, elites = {POP // 2}")
    update(int(slider.val))


def update(g):
    global zoom_contour
    h = runs[seed]
    mean, sig, C = h["mean"][g], h["sigma"][g], h["C"][g]
    X, F, nxt = h["X"][g], h["F"][g], h["new_mean"][g]
    is_el = np.zeros(POP, bool); is_el[h["elite"][g]] = True
    it = np.arange(1, g + 2)

    # 1. losses
    cand_F.set_data(np.full((~is_el).sum(), g + 1), F[~is_el])
    elite_F.set_data(np.full(is_el.sum(), g + 1), F[is_el])
    mean_L.set_data(it, h["loss"][:g + 1])
    best_L.set_data(it, h["best"][:g + 1])

    # 2. curves
    for i, line in enumerate(cand_curve):
        line.set_ydata(model(X[i], tt))
        line.set_color(GREEN if is_el[i] else GREY)
        line.set_alpha(0.55 if is_el[i] else 0.3)
    mean_curve.set_ydata(model(mean, tt))

    # 3. sigma + axes
    sigma_L.set_data(it, h["sigma"][:g + 1])
    ax_arr = np.array(h["axes"][:g + 1])
    for k, line in enumerate(axis_L):
        line.set_data(it, ax_arr[:, k])
    dots3.set_data([g + 1] * 4, [sig, *ax_arr[-1]])

    # 4 + 5. search space
    ex, ey = ellipse_2sigma(mean, sig, C)
    path = np.array(h["mean"][:g + 1])
    for S in (S4, S5):
        S["path"].set_data(path[:, 1], path[:, 2])
        S["cand"].set_data(X[~is_el, 1], X[~is_el, 2])
        S["elite"].set_data(X[is_el, 1], X[is_el, 2])
        S["ell"].set_data(ex, ey)
        S["step"].set_data([mean[1], nxt[1]], [mean[2], nxt[2]])
        S["mean"].set_data([mean[1]], [mean[2]])

    # zoom limits around candidates + ellipse, with a fresh local contour
    px = np.concatenate([X[:, 1], ex, [nxt[1]]]); py = np.concatenate([X[:, 2], ey, [nxt[2]]])
    cx, cy = (px.min() + px.max()) / 2, (py.min() + py.max()) / 2
    hx = max((px.max() - px.min()) * 0.6, 1e-3); hy = max((py.max() - py.min()) * 0.6, 1e-3)
    ax5.set_xlim(cx - hx, cx + hx); ax5.set_ylim(cy - hy, cy + hy)
    remove_contour(zoom_contour)
    Wz, Pz = np.meshgrid(np.linspace(cx - hx, cx + hx, 70), np.linspace(cy - hy, cy + hy, 70))
    Lz = profile_loss(Wz, Pz)
    lo, hi = max(Lz.min(), 1e-4), Lz.max()
    levels = np.geomspace(lo, hi, 12) if hi > lo * 1.01 else [lo]
    zoom_contour = ax5.contour(Wz, Pz, Lz, levels=levels, cmap="viridis", linewidths=0.7,
                               zorder=0)

    # 6. C
    vmax = np.abs(C).max()
    im.set_data(C); im.set_clim(-vmax, vmax)
    for i in range(3):
        for j in range(3):
            C_txt[i][j].set_text(f"{C[i, j]:.3g}")
            C_txt[i][j].set_color("white" if abs(C[i, j]) > 0.6 * vmax else INK)
    corr = C[1, 2] / np.sqrt(C[1, 1] * C[2, 2])

    # info on every panel
    head = f"iter {g + 1}/{ITERS}   evals {(g + 1) * POP}"
    info1.set_text(f"{head}\nloss of mean {h['loss'][g]:.3g}   best {h['best'][g]:.3g}")
    info2.set_text(f"{head}\nmean: A = {mean[0]:.2f}, w = {mean[1]:.2f}, phi = {mean[2]:.2f}")
    info3.set_text(f"{head}\nsigma = {sig:.3g}   stretch = {ax_arr[-1, 0] / ax_arr[-1, 2]:.1f}")
    info4.set_text(f"{head}\nmean (w, phi) = ({mean[1]:.2f}, {mean[2]:.2f})")
    info5.set_text(f"{head}\n2 sigma: w +/- {2 * sig * np.sqrt(C[1, 1]):.3g}, "
                   f"phi +/- {2 * sig * np.sqrt(C[2, 2]):.3g}")
    info6.set_text(f"{head}   sigma = {sig:.3g}   corr(w, phi) = {corr:+.2f}")
    fig.canvas.draw_idle()


# ---- controls: start picker, step slider, buttons ----
last = ITERS - 1
seed_slider = Slider(fig.add_axes([0.10, 0.105, 0.80, 0.025]), "start #", 0, N_SEEDS - 1,
                     valinit=START, valstep=1, color="#9b9a96")
slider = Slider(fig.add_axes([0.10, 0.065, 0.80, 0.025]), "iteration", 0, last,
                valinit=0, valstep=1)
slider.valtext.set_text("1")
seed_slider.on_changed(lambda val: set_seed(int(val)))
slider.on_changed(lambda val: (slider.valtext.set_text(f"{int(val) + 1}"), update(int(val))))


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

# play/pause: a timer advances the slider one iteration at a time
timer = fig.canvas.new_timer(interval=400)
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

set_seed(START)
plt.show()
