"""
SFNN training loop, step by step, on CartPole (one of the paper's three tasks).

WHAT IS EVOLVED (the "genome", 565 numbers, the same count as the paper)
    3 neuron types   input / hidden / output         each: 4x4 weights + 4 biases   = 3 x 20  =  60
    3 synapse types  input->hidden, hidden->hidden,  each: a small GRU learning rule = 3 x 168 = 504
                     hidden->output
    1 learning rate  how fast synapses follow their rule                                     =   1
                                                                                      total = 565
    Nothing else is evolved. In particular, NO weight of the network is evolved.

ONE GENERATION (outer loop, evolution)
    1. CMA-ES samples POP genomes around its current mean.
    2. Each genome lives ONE lifetime (inner loop below), with a NEW random wiring.
    3. Lifetime score = weighted average of the 8 episode returns (later episodes count more),
       divided by the max return, so 1 = the pole stayed up in every episode.
    4. CMA-ES moves its mean toward the genomes that scored best.

ONE LIFETIME (inner loop, learning during life)
    * Random wiring: 4 input neurons, 32 hidden, 2 output; half of all possible connections
      (input->hidden, hidden->hidden, hidden->output) are switched off at random.
    * Each synapse holds a 4-number state that acts as its weight; it starts small and random.
    * Every step: the cart's 4 numbers go into the 4 input neurons, the network ticks twice,
      the output neuron with the bigger signal picks left or right, and then EVERY synapse
      updates its own state with its type's rule, from: sender activity, receiver activity,
      last reward.
    * 8 episodes in a row; the network is NOT reset between them, so it can improve.
    In CartPole the reward is +1 every step (0 on the very first step of an episode), so the
    network learns mostly from the activity it sees, not from the reward value.

Simplifications versus the paper: one task instead of three, episodes capped at 200 steps
(CartPole-v1 allows 500; use --max-steps 500), the cart's 4 numbers are divided by their
typical size before entering the network, and the synapse state moves toward its GRU
output by the learning rate.

WINDOW
    Evolution             lifetime score per generation, plus episode 1 vs episode 8
    Learning within life  return per episode, for genomes from different generations
    CartPole              the current genome living a lifetime (animated)
    The genome            the 565 evolved numbers, and which parts evolution changed per generation
    The network's weights the weights inside ONE life, at birth and after each episode
    Weight change         how far the weights moved from birth, step by step, across the 8 episodes

Controls: Evolve on/off, +10 generations, Reset, New life (new random wiring),
          slider = which moment of the life the weight matrix shows.
Keys: space = evolve, f = +10 generations, n = new life.

Run:   python sfnn_cartpole_loop.py                      (uses CUDA if available)
       python sfnn_cartpole_loop.py --pop 128 --max-steps 500    (paper-like, needs a GPU)
       python sfnn_cartpole_loop.py --save shot.png --gens 60    (headless, saves a PNG)
Needs: torch, numpy, matplotlib (GUI backend such as TkAgg or QtAgg)
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NI, NH, NO, D = 4, 32, 2, 4          # input, hidden, output neurons; size of each activity vector
N_EP, MAX_STEPS, TICKS = 8, 200, 2
POP = 64
DEV = torch.device("cpu")
CMA_DT = torch.float64
INIT_STD = 0.5                        # spread of synapse states at birth
P_CONN = 0.5                          # fraction of connections that exist in a wiring
OBS_SCALE = torch.tensor([1.0, 1.0, 1.0, 1.0])   # set in main(): cart numbers divided by their typical size
LR_MAX = 1.0                          # learning rate = LR_MAX * sigmoid(evolved number - 2)

BLOCKS = [("input neuron", 20), ("hidden neuron", 20), ("output neuron", 20),
          ("rule: input→hidden", 168), ("rule: hidden→hidden", 168),
          ("rule: hidden→output", 168), ("learning rate", 1)]
DIM = sum(n for _, n in BLOCKS)       # 565
EDGES = np.cumsum([0] + [n for _, n in BLOCKS])

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fbe3d6", "#f7f7f5", "#d3e5f7", "#2a78d6"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#a9c9ef", "#2a78d6", "#0d3a73"])


# ---------------------------------------------------------------------------
# CMA-ES (batched code, used here with a single optimizer)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, n, lam, sigma0, gen):
        self.n, self.lam, self.mu, self.gen = n, lam, lam // 2, gen
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); w /= w.sum()
        me = self.me = 1 / np.sum(w ** 2)
        self.w = torch.as_tensor(w, dtype=CMA_DT, device=DEV)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = torch.zeros(n, dtype=CMA_DT, device=DEV); self.sigma = sigma0
        self.C = torch.eye(n, dtype=CMA_DT, device=DEV)
        self.pc = torch.zeros(n, dtype=CMA_DT, device=DEV); self.ps = torch.zeros_like(self.pc); self.g = 0
        self._eig()

    def _eig(self):
        ev, self.B = torch.linalg.eigh(self.C)
        self.D = ev.clamp(min=1e-20).sqrt()

    def ask(self):
        Z = torch.randn(self.lam, self.n, dtype=CMA_DT, device=DEV, generator=self.gen)
        self.Y = (Z * self.D) @ self.B.T
        return (self.m + self.sigma * self.Y).float()

    def tell(self, fit):
        order = torch.argsort(fit.to(CMA_DT), descending=True)[:self.mu]
        Ys = self.Y[order]; yw = self.w @ Ys
        self.m = self.m + self.sigma * yw
        invsq = self.B @ ((self.B.T @ yw) / self.D)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * invsq
        psn = self.ps.norm().item()
        hs = float(psn / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (torch.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Ys.T * self.w) @ Ys)
        self.C = (self.C + self.C.T) / 2
        self.sigma = min(10.0, self.sigma * np.exp((self.cs / self.ds) * (psn / self.chiN - 1)))
        self.g += 1
        self._eig()


# ---------------------------------------------------------------------------
# Genome -> building blocks
# ---------------------------------------------------------------------------
def unpack(X):
    P = X.shape[0]; i = 0; out = {}

    def take(n, shape):
        nonlocal i
        v = X[:, i:i + n].reshape((P,) + shape); i += n; return v

    for t in ("in", "hid", "out"):
        out["W_" + t] = take(D * D, (D, D)); out["b_" + t] = take(D, (D,))
    for s in ("ih", "hh", "ho"):
        out["Wx_" + s] = take(3 * D * (2 * D + 1), (3, D, 2 * D + 1))     # gates z, r, n
        out["U_" + s] = take(3 * D * D, (3, D, D))
        out["bg_" + s] = take(3 * D, (3, D))
    out["lr"] = LR_MAX * torch.sigmoid(take(1, ()) - 2.0)
    return out


def neuron(W, b, msg):
    return torch.tanh(torch.einsum('pek,pnk->pne', W, msg) + b[:, None, :])


def gru_update(H, M, p, s, pre, post, r, lr, live):
    """One learning step for every synapse of type s. H [P,I,J,D] (receiver i, sender j)."""
    Wx, U, bg = p["Wx_" + s], p["U_" + s], p["bg_" + s]
    xp = torch.einsum('pgek,pjk->pgje', Wx[..., :D], pre)              # sender part
    xq = torch.einsum('pgek,pik->pgie', Wx[..., D:2 * D], post)         # receiver part
    xr = Wx[..., 2 * D] * r[:, None, None] + bg                         # reward part + bias
    X = xp[:, :, None] + xq[:, :, :, None] + xr[:, :, None, None]       # [P,3,I,J,D]
    Uh = torch.einsum('pgek,pijk->pgije', U, H)
    z = torch.sigmoid(X[:, 0] + Uh[:, 0]); rr = torch.sigmoid(X[:, 1] + Uh[:, 1])
    n = torch.tanh(X[:, 2] + rr * Uh[:, 2])
    Hn = H + lr[:, None, None, None] * ((1 - z) * n + z * H - H)
    return torch.where(M[..., None] & live[:, None, None, None], Hn, H)


# ---------------------------------------------------------------------------
# CartPole (same physics as Gym/Gymnax CartPole-v1), batched
# ---------------------------------------------------------------------------
def cart_step(s, right):
    x, xd, th, thd = s.unbind(1)
    force = torch.where(right, 10.0, -10.0)
    ct, st = torch.cos(th), torch.sin(th)
    temp = (force + 0.05 * thd ** 2 * st) / 1.1
    thacc = (9.8 * st - ct * temp) / (0.5 * (4.0 / 3.0 - 0.1 * ct ** 2 / 1.1))
    xacc = temp - 0.05 * thacc * ct / 1.1
    x = x + 0.02 * xd; xd = xd + 0.02 * xacc; th = th + 0.02 * thd; thd = thd + 0.02 * thacc
    s = torch.stack([x, xd, th, thd], 1)
    fell = (x.abs() > 2.4) | (th.abs() > 0.2095)
    return s, fell


# ---------------------------------------------------------------------------
# One lifetime for P genomes at once
# ---------------------------------------------------------------------------
def lifetime(theta, gen, record=False):
    P = theta.shape[0]; p = unpack(theta)
    M1 = torch.rand(P, NH, NI, device=DEV, generator=gen) < P_CONN
    M2 = torch.rand(P, NH, NH, device=DEV, generator=gen) < P_CONN
    M3 = torch.rand(P, NO, NH, device=DEV, generator=gen) < P_CONN
    H1 = INIT_STD * torch.randn(P, NH, NI, D, device=DEV, generator=gen) * M1[..., None]
    H2 = INIT_STD * torch.randn(P, NH, NH, D, device=DEV, generator=gen) * M2[..., None]
    H3 = INIT_STD * torch.randn(P, NO, NH, D, device=DEV, generator=gen) * M3[..., None]
    a_in = torch.zeros(P, NI, D, device=DEV); a_h = torch.zeros(P, NH, D, device=DEV)
    a_o = torch.zeros(P, NO, D, device=DEV)
    ret = torch.zeros(P, N_EP, device=DEV)
    rec = None
    if record:
        W0 = [H1.mean(-1).clone(), H2.mean(-1).clone(), H3.mean(-1).clone()]
        rec = dict(masks=[M1[0].cpu().numpy(), M2[0].cpu().numpy(), M3[0].cpu().numpy()],
                   snaps=[weight_matrix(W0, M1, M2, M3)], states=[], ep_of_step=[], change=[],
                   H0=(H1.clone(), H2.clone(), H3.clone()))
    for e in range(N_EP):
        s = (torch.rand(P, 4, device=DEV, generator=gen) - 0.5) * 0.1
        live = torch.ones(P, dtype=torch.bool, device=DEV)
        r = torch.zeros(P, device=DEV)
        for t in range(MAX_STEPS):
            a_in = torch.tanh((s / OBS_SCALE)[:, :, None] * p["W_in"].sum(-1)[:, None, :] + p["b_in"][:, None, :])
            for _ in range(TICKS):
                m_h = (torch.einsum('pij,pijd,pjd->pid', M1.float(), H1, a_in) / np.sqrt(NI * P_CONN)
                       + torch.einsum('pij,pijd,pjd->pid', M2.float(), H2, a_h) / np.sqrt(NH * P_CONN))
                a_h = torch.where(live[:, None, None], neuron(p["W_hid"], p["b_hid"], m_h), a_h)
                m_o = torch.einsum('pij,pijd,pjd->pid', M3.float(), H3, a_h) / np.sqrt(NH * P_CONN)
                a_o = torch.where(live[:, None, None], neuron(p["W_out"], p["b_out"], m_o), a_o)
            right = a_o[:, 1, 0] > a_o[:, 0, 0]
            H1 = gru_update(H1, M1, p, "ih", a_in, a_h, r, p["lr"], live)
            H2 = gru_update(H2, M2, p, "hh", a_h, a_h, r, p["lr"], live)
            H3 = gru_update(H3, M3, p, "ho", a_h, a_o, r, p["lr"], live)
            s2, fell = cart_step(s, right)
            ret[:, e] += live.float()
            s = torch.where(live[:, None], s2, s)
            live = live & ~fell
            r = torch.ones(P, device=DEV)
            if record:
                rec["states"].append(s[0].cpu().numpy()); rec["ep_of_step"].append(e)
                d = sum(((h - h0) ** 2).sum() for h, h0 in zip((H1, H2, H3), rec["H0"])).sqrt()
                n0 = sum((h0 ** 2).sum() for h0 in rec["H0"]).sqrt()
                rec["change"].append((d / n0).item())
                if not live[0]:
                    break
            elif t % 20 == 19 and not live.any():
                break
        if record:
            rec["snaps"].append(weight_matrix([H1.mean(-1), H2.mean(-1), H3.mean(-1)], M1, M2, M3))
    w = torch.arange(1, N_EP + 1, device=DEV, dtype=torch.float32); w = w / w.sum()
    fit = (ret / MAX_STEPS * w).sum(1)
    return fit, ret, rec


def weight_matrix(Ws, M1, M2, M3):
    """Whole network as one picture: rows = receiving neuron (32 hidden, then 2 output),
    columns = sending neuron (4 input, then 32 hidden). NaN = no connection."""
    W1, W2, W3 = [w[0].cpu().numpy() for w in Ws]
    A = np.full((NH + NO, NI + NH), np.nan)
    A[:NH, :NI] = np.where(M1[0].cpu().numpy(), W1, np.nan)
    A[:NH, NI:] = np.where(M2[0].cpu().numpy(), W2, np.nan)
    A[NH:, NI:] = np.where(M3[0].cpu().numpy(), W3, np.nan)
    return A


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class App:
    def __init__(self, budget):
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
            "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 9.5, "axes.titlesize": 10.5,
        })
        self.budget = budget
        self.fig = plt.figure(figsize=(16, 10))
        try:
            self.fig.canvas.manager.set_window_title("SFNN training loop on CartPole")
        except AttributeError:
            pass
        gs = self.fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.885, bottom=0.17,
                                   hspace=0.42, wspace=0.24, width_ratios=[1.15, 1, 1])
        self.axF = self.fig.add_subplot(gs[0, 0]); self.axL = self.fig.add_subplot(gs[0, 1])
        self.axC = self.fig.add_subplot(gs[0, 2])
        sub = gs[1, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.55)
        self.axGs = self.fig.add_subplot(sub[0]); self.axGd = self.fig.add_subplot(sub[1])
        self.axW = self.fig.add_subplot(gs[1, 1]); self.axX = self.fig.add_subplot(gs[1, 2])
        self.title = self.fig.text(0.045, 0.955, "", fontsize=13, va="center")
        self.sub = self.fig.text(0.045, 0.925, "", fontsize=9.5, color=INK2, va="center")
        self.cb = None
        self._controls()
        self.evolving = False; self.anim_i = 0
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._tick)
        self.reset()

    # ---------------- evolution ----------------
    def reset(self):
        if self.evolving:
            self.toggle()
        self.g_cma = torch.Generator(device=DEV); self.g_cma.manual_seed(1)
        self.g_env = torch.Generator(device=DEV); self.g_env.manual_seed(0)
        self.es = CMAES(DIM, POP, 0.3, self.g_cma)
        self.gen = 0; self.spg = float("nan")
        self.h_mean, self.h_best, self.h_ep1, self.h_ep8, self.h_sigma = [], [], [], [], []
        self.h_eps = {}                       # generation -> population-average return per episode
        self.h_change = []                    # per generation: how much each genome block moved
        self.genome_snap = {}
        self.new_life()

    def evolve_one(self):
        m_old = self.es.m.clone()
        X = self.es.ask()
        fit, ret, _ = lifetime(X, self.g_env)
        self.es.tell(fit)
        ret_np = ret.cpu().numpy()
        self.h_mean.append(fit.mean().item()); self.h_best.append(fit.max().item())
        self.h_ep1.append(ret_np[:, 0].mean()); self.h_ep8.append(ret_np[:, -1].mean())
        self.h_sigma.append(self.es.sigma)
        self.h_eps[self.gen] = ret_np.mean(0)
        dm = (self.es.m - m_old).abs().cpu().numpy()
        self.h_change.append([dm[EDGES[i]:EDGES[i + 1]].mean() for i in range(len(BLOCKS))])
        self.gen += 1

    def evolve_for(self, seconds=None, n=None):
        t0 = time.perf_counter(); k = 0
        while True:
            self.evolve_one(); k += 1
            if (n is not None and k >= n) or (n is None and time.perf_counter() - t0 >= seconds):
                break
        dt = (time.perf_counter() - t0) / k
        self.spg = dt if np.isnan(self.spg) else 0.7 * self.spg + 0.3 * dt

    def plus10(self):
        self.evolve_for(n=10); self.new_life()

    # ---------------- one recorded life of the current genome ----------------
    def new_life(self):
        g = torch.Generator(device=DEV); g.manual_seed(int(np.random.default_rng().integers(1 << 30)))
        theta = self.es.m.float()[None]
        _, ret, rec = lifetime(theta, g, record=True)
        self.life = rec; self.life_ret = ret[0].cpu().numpy(); self.anim_i = 0
        self.draw()

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig
        self.b_evo = Button(f.add_axes([0.045, 0.06, 0.09, 0.045]), "Evolve: off")
        self.b_ff = Button(f.add_axes([0.14, 0.06, 0.10, 0.045]), "+10 generations")
        self.b_rst = Button(f.add_axes([0.245, 0.06, 0.06, 0.045]), "Reset")
        self.b_new = Button(f.add_axes([0.31, 0.06, 0.12, 0.045]), "New life (new wiring)")
        self.b_evo.on_clicked(lambda e: self.toggle())
        self.b_ff.on_clicked(lambda e: self.plus10())
        self.b_rst.on_clicked(lambda e: self.reset())
        self.b_new.on_clicked(lambda e: self.new_life())
        self.s_ep = Slider(f.add_axes([0.6, 0.075, 0.14, 0.025]), "weights shown after episode",
                           0, N_EP, valinit=N_EP, valstep=1, color=BLUE)
        self.s_ep.label.set_fontsize(9); self.s_ep.on_changed(lambda v: self.draw())
        f.text(0.785, 0.103, "One generation =\n"
               f"  1. sample {POP} genomes (565 numbers each)\n"
               f"  2. each lives {N_EP} episodes in a NEW random wiring,\n"
               "      its synapses learning with the evolved rules\n"
               "  3. score each life, 4. move toward the best genomes",
               fontsize=8.5, color=INK2, va="top", linespacing=1.3)
        f.text(0.045, 0.02, "Keys: space = evolve on/off, f = +10 generations, n = new life",
               fontsize=8.5, color=INK2)
        f.canvas.mpl_connect("key_press_event", self._key)

    def _key(self, e):
        {" ": self.toggle, "f": self.plus10, "n": self.new_life}.get(e.key, lambda: None)()

    def toggle(self):
        self.evolving = not self.evolving
        self.b_evo.label.set_text("Evolve: on" if self.evolving else "Evolve: off")
        if not self.evolving:
            self.new_life()
        self.fig.canvas.draw_idle()

    def start(self):
        self.timer.start()

    def _tick(self):
        if self.evolving:
            self.evolve_for(seconds=self.budget)
            if self.gen % 5 == 0:
                self.new_life()
            else:
                self.draw()
        elif self.anim_i < len(self.life["states"]) - 1:
            self.anim_i = min(self.anim_i + 3, len(self.life["states"]) - 1)
            self._draw_cart(); self._draw_change(); self.fig.canvas.draw_idle()

    # ---------------- drawing ----------------
    @staticmethod
    def _clean(ax):
        ax.cla(); ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def draw(self):
        spg = "" if np.isnan(self.spg) else f"    {self.spg:.2f} s/generation"
        self.title.set_text(f"SFNN training loop on CartPole    generation {self.gen}    device: {DEV.type}{spg}")
        self.sub.set_text(f"Evolved: only the 565 numbers of the building blocks (3 neuron types, 3 synapse rules, "
                          f"1 learning rate). Each life: new random wiring, {N_EP} episodes of up to {MAX_STEPS} steps, "
                          f"weights learned during life.")
        self._draw_fitness(); self._draw_within(); self._draw_cart()
        self._draw_genome(); self._draw_weights(); self._draw_change()
        self.fig.canvas.draw_idle()

    def _draw_fitness(self):
        ax = self.axF; self._clean(ax)
        x = np.arange(1, self.gen + 1)
        if self.gen:
            ax.plot(x, self.h_best, color=BLUE, lw=1.2, alpha=0.6, label="best genome's lifetime score")
            ax.plot(x, self.h_mean, color=BLUE, lw=2.4, label="average lifetime score")
            ax.plot(x, np.array(self.h_ep1) / MAX_STEPS, color=ORANGE, lw=1.6, ls="--", label="episode 1 (average)")
            ax.plot(x, np.array(self.h_ep8) / MAX_STEPS, color=GREEN, lw=1.6, ls="--", label="episode 8 (average)")
            ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        else:
            ax.text(0.5, 0.5, "press  Evolve  or  +10 generations", transform=ax.transAxes, ha="center", color=INK2)
        ax.set_xlim(0, max(30, self.gen)); ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("generation"); ax.set_ylabel(f"score (1 = pole up {MAX_STEPS} steps)")
        ax.set_title("Outer loop (evolution): score per generation", loc="left")

    def _draw_within(self):
        ax = self.axL; self._clean(ax)
        gens = sorted(self.h_eps)
        if gens:
            pick = sorted(set([gens[0]] + [gens[int(round(q * (len(gens) - 1)))] for q in (0.25, 0.5, 0.75, 1.0)]))
            for j, g in enumerate(pick):
                c = SEQ(0.35 + 0.65 * j / max(1, len(pick) - 1))
                ax.plot(range(1, N_EP + 1), self.h_eps[g], color=c, lw=2.2, marker="o", ms=4, label=f"generation {g + 1}")
            ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        ax.set_xlim(0.7, N_EP + 0.3); ax.set_ylim(0, MAX_STEPS * 1.15)
        ax.set_xlabel("episode within one life"); ax.set_ylabel("steps the pole stayed up (average)")
        ax.set_title("Inner loop result: does the network improve during its life?", loc="left")

    def _draw_cart(self):
        ax = self.axC; ax.cla(); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.5, 1.45); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.plot([-2.4, 2.4], [0, 0], color=INK2, lw=1)
        for xx in (-2.4, 2.4):
            ax.plot([xx, xx], [-0.08, 0.08], color=RED, lw=2)
        st = self.life["states"][self.anim_i]; e = self.life["ep_of_step"][self.anim_i]
        x, th = st[0], st[2]
        ax.add_patch(plt.Rectangle((x - 0.25, -0.07), 0.5, 0.14, color=INK2))
        ax.plot([x, x + np.sin(th)], [0.07, 0.07 + np.cos(th)], color=BLUE if abs(th) < 0.2095 else RED,
                lw=5, solid_capstyle="round")
        t_in_ep = self.anim_i - self.life["ep_of_step"].index(e)
        rets = "  ".join(f"{int(v)}" for v in self.life_ret[:e + 1])
        ax.set_title(f"This life (current genome): episode {e + 1}/{N_EP}, step {t_in_ep + 1}\n"
                     f"returns so far: {rets}", loc="left")

    def _draw_genome(self):
        m = self.es.m.cpu().numpy()
        ax = self.axGs; ax.cla()
        lim = max(0.3, np.abs(m).max())
        ax.imshow(m[None], aspect="auto", cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
        for e in EDGES[1:-1]:
            ax.axvline(e - 0.5, color=INK, lw=0.8)
        ax.set_yticks([]); ax.set_xticks((EDGES[:-1] + EDGES[1:]) / 2 - 0.5)
        ax.set_xticklabels(["in", "hid", "out", "rule in→hid", "rule hid→hid", "rule hid→out", "lr"],
                           fontsize=7.5)
        lr = LR_MAX / (1 + np.exp(-(m[-1] - 2.0)))
        ax.set_title(f"The genome: 565 evolved numbers (blue +, orange −)   learning rate = {lr:.2f}",
                     loc="left", fontsize=10)
        ax = self.axGd; ax.cla()
        if self.h_change:
            Hc = np.array(self.h_change[-120:]).T
            g0 = max(0, self.gen - 120)
            ax.imshow(Hc, aspect="auto", cmap=SEQ, interpolation="nearest", origin="upper",
                      extent=[g0 + 0.5, self.gen + 0.5, len(BLOCKS) - 0.5, -0.5], vmin=0)
        else:
            ax.text(0.5, 0.5, "no generations yet", transform=ax.transAxes, ha="center", color=INK2)
        ax.set_yticks(range(len(BLOCKS))); ax.set_yticklabels([n for n, _ in BLOCKS], fontsize=8)
        ax.set_xlabel("generation")
        ax.set_title("What evolution changed each generation (darker = moved more)", loc="left", fontsize=10)

    def _draw_weights(self):
        k = int(self.s_ep.val)
        k = min(k, len(self.life["snaps"]) - 1)
        A = self.life["snaps"][k]
        ax = self.axW; ax.cla()
        if self.cb is not None:
            try:
                self.cb.remove()
            except Exception:
                pass
        ax.set_facecolor("#ecebe7")
        lim = max(1e-3, np.nanmax(np.abs(A)))           # colour scale fitted to this moment
        im = ax.imshow(A, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")
        ax.axvline(NI - 0.5, color=INK, lw=0.8); ax.axhline(NH - 0.5, color=INK, lw=0.8)
        ax.set_xticks([1.5, NI + NH / 2]); ax.set_xticklabels(["from input", "from hidden"])
        ax.set_yticks([NH / 2, NH + 0.5]); ax.set_yticklabels(["to hidden", "to output"])
        ax.tick_params(length=0)
        self.cb = self.fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        self.cb.ax.tick_params(labelsize=7.5)
        when = "at birth (random)" if k == 0 else f"after episode {k}"
        ax.set_title(f"Inside this life: every weight {when}\n(grey = no connection in this wiring)",
                     loc="left")

    def _draw_change(self):
        ax = self.axX; self._clean(ax)
        ch = np.array(self.life["change"]); eps = np.array(self.life["ep_of_step"])
        n = self.anim_i + 1
        ax.plot(np.arange(len(ch)), ch, color=BLUE, lw=1, alpha=0.25)
        ax.plot(np.arange(n), ch[:n], color=BLUE, lw=2)
        top = max(0.05, ch.max() * 1.3) if len(ch) else 1
        last_x = -1e9
        for e in range(N_EP):
            idx = np.where(eps == e)[0]
            if len(idx):
                ax.axvline(idx[0], color=GRID, lw=1)
                if idx[0] - last_x < 0.06 * max(10, len(ch)):
                    continue
                last_x = idx[0]
                ax.text(idx[0] + 2, top * 0.97, f"ep{e + 1}\n{int(self.life_ret[e])}", fontsize=7.5,
                        color=INK2, va="top")
        ax.set_xlim(0, max(10, len(ch))); ax.set_ylim(0, top)
        ax.set_xlabel("step within the life (episodes back to back)")
        ax.set_ylabel("weight change from birth (relative)")
        ax.set_title("Inner loop (learning): how far the weights moved,\nlabels = steps survived per episode",
                     loc="left")


def main():
    global DEV, POP, MAX_STEPS, N_EP, CMA_DT, OBS_SCALE
    ap = argparse.ArgumentParser(description="SFNN training loop on CartPole")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pop", type=int, default=None, help="genomes per generation (paper: 128)")
    ap.add_argument("--max-steps", type=int, default=200, help="steps per episode (CartPole-v1: 500)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.3, help="seconds of evolution per screen update")
    ap.add_argument("--save", default=None); ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    DEV = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       (a.device if a.device != "auto" else "cpu"))
    POP = a.pop or (128 if DEV.type == "cuda" else 48)
    MAX_STEPS, N_EP = a.max_steps, a.episodes
    OBS_SCALE = torch.tensor([2.4, 2.0, 0.21, 2.0], device=DEV)
    if DEV.type == "cuda":
        CMA_DT = torch.float32
        torch.set_num_threads(2)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    np.random.seed(a.seed)
    if a.save:
        plt.switch_backend("Agg")
    app = App(a.budget)
    if a.save:
        t0 = time.perf_counter()
        for i in range(a.gens):
            app.evolve_one()
            if (i + 1) % 10 == 0:
                print(f"gen {i + 1}: avg {app.h_mean[-1]:.2f}  best {app.h_best[-1]:.2f}  "
                      f"ep1 {app.h_ep1[-1]:.0f}  ep8 {app.h_ep8[-1]:.0f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        app.new_life(); app.anim_i = len(app.life["states"]) - 1; app.draw()
        app.fig.savefig(a.save, dpi=110)
        print(f"Saved {a.save}")
    else:
        app.start(); plt.show()


if __name__ == "__main__":
    main()