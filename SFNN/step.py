"""
SFNN, step by step: every equation of one time step, with real numbers.

A small SFNN (4 input neurons for CartPole's 4 sensors, 4 hidden, 2 output) runs real
CartPole time steps. Each page explains one part of the loop, in the order the
computer does it, and shows the numbers that equation produced on the current step.

Pages
    0  The big picture
    1  The genome (the only thing evolution changes)
    2  A new random wiring for each life
    3  Sensors -> input neurons
    4  Message into a hidden neuron
    5  Neuron update (2 ticks per step)
    6  Choose the action
    7  Synapse learning: GRU gates
    8  Synapse learning: new synapse state
    9  CartPole physics step
    10 Repeat: a whole life of 8 episodes
    11 The lifetime score
    12 One generation: 128 separate robots (run for real, click one to inspect it)
    13 CMA-ES moves the mean (the 128 genomes as a cloud around the mean)
    14 What is kept and what is thrown away, then back to page 2

Paper vs this demo: each page says which parts are from the paper and which are
simplified here (the paper gives the building blocks in words; the exact formulas for
the message sum and the learning-rate step are this demo's choices).

Controls
    Back / Next (or arrow keys), or click a page name on the left
    Run 1 time step (space)      advances the simulation; every page shows the new numbers
    Run to end of episode (e)
    Next hidden neuron (h)       which neuron/synapse the detailed numbers follow
    New wiring (w)               a new life: new random connections, same genome
    New genome (g)               new random building blocks
    Run 1 generation (n)         128 genomes each live a real life, CMA-ES updates (pages 12-14)

The window sizes itself to your screen and rescales its fonts when you resize it.
Run:   python sfnn_step_by_step.py
       python sfnn_step_by_step.py --size 16 9          (force a figure size in inches)
       python sfnn_step_by_step.py --save page.png --page 8   (headless screenshot)
Needs: numpy, matplotlib (GUI backend such as TkAgg or QtAgg)
"""

import argparse
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.widgets import Button

# ---------------------------------------------------------------------------
# Model sizes (paper: 32 hidden neurons; 4 here so every number fits on screen)
# ---------------------------------------------------------------------------
NI, NH, NO, D = 4, 4, 2, 4
N_EP, MAX_STEPS = 8, 200
P_CONN, INIT_STD = 0.5, 0.5
OBS_SCALE = np.array([2.4, 2.0, 0.21, 2.0])
K_IN, K_HID = NI * P_CONN, NH * P_CONN          # expected number of incoming links per type
BLOCKS = [("input neuron type", 20), ("hidden neuron type", 20), ("output neuron type", 20),
          ("synapse rule input\u2192hidden", 168), ("synapse rule hidden\u2192hidden", 168),
          ("synapse rule hidden\u2192output", 168), ("learning rate", 1)]
DIM = sum(n for _, n in BLOCKS)
EDGES = np.cumsum([0] + [n for _, n in BLOCKS])

SURFACE, INK, INK2, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b7b0", "#e4e3df"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#6250d6"
HILITE = "#fff4cc"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fbe3d6", "#f7f7f5", "#d3e5f7", "#2a78d6"])
IN_LABELS = [r"$x$", r"$\dot{x}$", r"$\phi$", r"$\dot{\phi}$"]
IN_NAMES = ["cart position x", "cart speed", "pole angle \u03c6", "pole spin"]


def sig(v):
    return 1 / (1 + np.exp(-v))


def vec(v, w=5):
    return "[" + " ".join(f"{x:+.2f}".rjust(w) for x in np.ravel(v)) + " ]"


# ---------------------------------------------------------------------------
# Genome -> building blocks (same layout and size as the evolution app: 565 numbers)
# ---------------------------------------------------------------------------
def unpack(theta):
    theta = np.asarray(theta); lead = theta.shape[:-1]      # lead = () for one genome, (P,) for P genomes
    i = 0; p = {}

    def take(n, shape):
        nonlocal i
        v = theta[..., i:i + n].reshape(lead + shape); i += n; return v

    for t in ("in", "hid", "out"):
        p["W_" + t] = take(D * D, (D, D)); p["b_" + t] = take(D, (D,))
    for s in ("ih", "hh", "ho"):
        p["Wx_" + s] = take(3 * D * (2 * D + 1), (3, D, 2 * D + 1))   # gates: z, rho, n
        p["U_" + s] = take(3 * D * D, (3, D, D))
        p["bg_" + s] = take(3 * D, (3, D))
    lr = sig(take(1, ()) - 2.0)
    p["lr"] = float(lr) if lead == () else lr
    return p


def gru_all(H, M, p, s, pre, post, r):
    """GRU learning step for every synapse of type s. H[i, j] = synapse from j to i."""
    Wx, U, bg = p["Wx_" + s], p["U_" + s], p["bg_" + s]
    xp = np.einsum('gek,jk->gje', Wx[..., :D], pre)
    xq = np.einsum('gek,ik->gie', Wx[..., D:2 * D], post)
    xr = Wx[..., 2 * D] * r + bg
    X = xp[:, None] + xq[:, :, None] + xr[:, None, None]          # [3, I, J, D]
    Uh = np.einsum('gek,ijk->gije', U, H)
    z = sig(X[0] + Uh[0]); rho = sig(X[1] + Uh[1])
    n = np.tanh(X[2] + rho * Uh[2])
    ht = (1 - z) * n + z * H
    Hn = np.where(M[..., None], H + p["lr"] * (ht - H), H)
    return Hn, dict(z=z, rho=rho, n=n, ht=ht)


def cart_step(s, right):
    x, xd, ph, phd = s
    F = 10.0 if right else -10.0
    tau = (F + 0.05 * phd ** 2 * np.sin(ph)) / 1.1
    phdd = (9.8 * np.sin(ph) - np.cos(ph) * tau) / (0.5 * (4 / 3 - 0.1 * np.cos(ph) ** 2 / 1.1))
    xdd = tau - 0.05 * phdd * np.cos(ph) / 1.1
    s2 = np.array([x + 0.02 * xd, xd + 0.02 * xdd, ph + 0.02 * phd, phd + 0.02 * phdd])
    fell = abs(s2[0]) > 2.4 or abs(s2[2]) > 0.2095
    return s2, fell, dict(F=F, tau=tau, phdd=phdd, xdd=xdd)


# ---------------------------------------------------------------------------
# One SFNN living in CartPole, one time step at a time, recording everything
# ---------------------------------------------------------------------------
class Sim:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.focus = 0
        self.new_genome()

    def new_genome(self):
        self.theta = self.rng.normal(0, 0.5, DIM)
        self.p = unpack(self.theta)
        self.new_life()

    def set_genome(self, theta):
        self.theta = np.array(theta, dtype=float)
        self.p = unpack(self.theta)
        self.new_life()

    def new_life(self):
        r = self.rng
        self.M1 = r.random((NH, NI)) < P_CONN
        for i in range(NH):                       # demo only: every hidden neuron gets >= 1 sensor link
            if not self.M1[i].any():
                self.M1[i, r.integers(NI)] = True
        self.M2 = (r.random((NH, NH)) < P_CONN) & ~np.eye(NH, dtype=bool)
        self.M3 = r.random((NO, NH)) < P_CONN
        self.H1 = INIT_STD * r.normal(size=(NH, NI, D)) * self.M1[..., None]
        self.H2 = INIT_STD * r.normal(size=(NH, NH, D)) * self.M2[..., None]
        self.H3 = INIT_STD * r.normal(size=(NO, NH, D)) * self.M3[..., None]
        self.H1_birth = self.H1.copy()
        self.a_in = np.zeros((NI, D)); self.a_h = np.zeros((NH, D)); self.a_o = np.zeros((NO, D))
        self.s = (r.random(4) - 0.5) * 0.1
        self.t = 0; self.t_ep = 0; self.ep = 0; self.r = 0.0
        self.returns = np.zeros(N_EP); self.over = False
        self.rec = None
        self.step()

    def focus_j(self):
        return int(np.argmax(self.M1[self.focus]))      # first sensor linked to the focus neuron

    STATE = ("H1", "H2", "H3", "a_in", "a_h", "a_o", "s", "t", "t_ep", "ep", "r", "returns", "over")

    def step(self):
        if self.over:
            self.new_life(); return
        self._snap = ({k: np.copy(getattr(self, k)) for k in self.STATE}, self.rng.bit_generator.state)
        p, f = self.p, self.focus
        rec = dict(t=self.t, t_ep=self.t_ep, ep=self.ep, r_prev=self.r, s=self.s.copy())
        # 3. sensors -> input neurons
        o = self.s / OBS_SCALE
        a_in = np.tanh(o[:, None] * p["W_in"].sum(1)[None, :] + p["b_in"][None, :])
        rec.update(o=o, a_in=a_in)
        # 4-5. two ticks of messages and neuron updates
        a_h = self.a_h.copy(); a_h_start = a_h.copy(); ticks = []
        for _ in range(2):
            c_in = self.M1[..., None] * self.H1 * a_in[None]          # [I, J, D]
            c_hh = self.M2[..., None] * self.H2 * a_h[None]
            m_h = c_in.sum(1) / np.sqrt(K_IN) + c_hh.sum(1) / np.sqrt(K_HID)
            pre_h = m_h @ p["W_hid"].T + p["b_hid"]
            a_h_old = a_h; a_h = np.tanh(pre_h)
            m_o = (self.M3[..., None] * self.H3 * a_h[None]).sum(1) / np.sqrt(K_HID)
            pre_o = m_o @ p["W_out"].T + p["b_out"]
            a_o = np.tanh(pre_o)
            ticks.append(dict(c_in=c_in[f], c_hh=c_hh[f], a_h_used=a_h_old, m=m_h[f], pre=pre_h[f],
                              a=a_h[f], m_o=m_o, pre_o=pre_o, a_o=a_o))
        rec.update(ticks=ticks, a_h=a_h, a_o=a_o, a_h_start=a_h_start)
        # 6. action
        right = a_o[1, 0] > a_o[0, 0]
        rec["right"] = right
        # 7-8. learning in every synapse (uses the reward of the previous step)
        j = self.focus_j()
        h_old = self.H1[f, j].copy()
        H1n, g = gru_all(self.H1, self.M1, p, "ih", a_in, a_h, self.r)
        H2n, _ = gru_all(self.H2, self.M2, p, "hh", a_h, a_h, self.r)
        H3n, _ = gru_all(self.H3, self.M3, p, "ho", a_h, a_o, self.r)
        x = np.concatenate([a_in[j], a_h[f], [self.r]])
        Wx, U, bg = p["Wx_ih"], p["U_ih"], p["bg_ih"]
        rec.update(j=j, h_old=h_old, x=x, z=g["z"][f, j], rho=g["rho"][f, j], n=g["n"][f, j],
                   ht=g["ht"][f, j], h_new=H1n[f, j],
                   Wz_x=Wx[0] @ x, Uz_h=U[0] @ h_old, Wr_x=Wx[1] @ x, Ur_h=U[1] @ h_old,
                   Wn_x=Wx[2] @ x, Un_h=U[2] @ h_old, lr=p["lr"],
                   dH=np.sqrt(sum(((a - b) ** 2).sum() for a, b in
                                  ((H1n, self.H1), (H2n, self.H2), (H3n, self.H3)))))
        self.H1, self.H2, self.H3 = H1n, H2n, H3n
        self.a_in, self.a_h, self.a_o = a_in, a_h, a_o
        # 9. environment
        s2, fell, phys = cart_step(self.s, right)
        rec.update(s2=s2, fell=fell, phys=phys)
        self.returns[self.ep] += 1
        self.t += 1; self.t_ep += 1; self.r = 1.0
        if fell or self.t_ep >= MAX_STEPS:
            rec["ended"] = True
            self.ep += 1; self.t_ep = 0; self.r = 0.0
            self.s = (self.rng.random(4) - 0.5) * 0.1
            if self.ep >= N_EP:
                self.over = True
        else:
            rec["ended"] = False
            self.s = s2
        self.rec = rec

    def redo_last(self):
        """Undo the last step and compute it again (used when the followed neuron changes)."""
        snap, rs = self._snap
        for k, v in snap.items():
            setattr(self, k, v.item() if v.ndim == 0 else v)
        self.rng.bit_generator.state = rs
        self.step()

    def run_episode(self):
        ep = self.ep
        for _ in range(MAX_STEPS + 1):
            self.step()
            if self.ep != ep or self.over:
                break


# ---------------------------------------------------------------------------
# The outer loop: 128 genomes, each living its OWN life (batched only for speed)
# ---------------------------------------------------------------------------
POP = 128


def cart_step_batch(s, right):
    x, xd, ph, phd = s.T
    F = np.where(right, 10.0, -10.0)
    tau = (F + 0.05 * phd ** 2 * np.sin(ph)) / 1.1
    phdd = (9.8 * np.sin(ph) - np.cos(ph) * tau) / (0.5 * (4 / 3 - 0.1 * np.cos(ph) ** 2 / 1.1))
    xdd = tau - 0.05 * phdd * np.cos(ph) / 1.1
    s2 = np.stack([x + 0.02 * xd, xd + 0.02 * xdd, ph + 0.02 * phd, phd + 0.02 * phdd], 1)
    fell = (np.abs(s2[:, 0]) > 2.4) | (np.abs(s2[:, 2]) > 0.2095)
    return s2, fell


def gru_batch(H, M, p, s, pre, post, r, live):
    """Same rule as gru_all, for P separate networks at once. H [P, I, J, D]."""
    Wx, U, bg = p["Wx_" + s], p["U_" + s], p["bg_" + s]
    xp = np.einsum('pgek,pjk->pgje', Wx[..., :D], pre)
    xq = np.einsum('pgek,pik->pgie', Wx[..., D:2 * D], post)
    xr = Wx[..., 2 * D] * r[:, None, None] + bg
    X = xp[:, :, None] + xq[:, :, :, None] + xr[:, :, None, None]
    Uh = np.einsum('pgek,pijk->pgije', U, H)
    z = sig(X[:, 0] + Uh[:, 0]); rho = sig(X[:, 1] + Uh[:, 1])
    n = np.tanh(X[:, 2] + rho * Uh[:, 2])
    Hn = H + p["lr"][:, None, None, None] * ((1 - z) * n + z * H - H)
    return np.where(M[..., None] & live[:, None, None, None], Hn, H)


def lives_batch(X, rng):
    """Every row of X is one genome. Each gets its own random wiring, its own random starting
    weights and its own 8-episode life in CartPole. Returns score F and steps per episode."""
    P = len(X); p = unpack(X)
    M1 = rng.random((P, NH, NI)) < P_CONN
    pi, hi = np.nonzero(~M1.any(2))
    M1[pi, hi, rng.integers(NI, size=len(pi))] = True
    M2 = (rng.random((P, NH, NH)) < P_CONN) & ~np.eye(NH, dtype=bool)
    M3 = rng.random((P, NO, NH)) < P_CONN
    F1, F2, F3 = M1.astype(float), M2.astype(float), M3.astype(float)
    H1 = INIT_STD * rng.normal(size=(P, NH, NI, D)) * M1[..., None]
    H2 = INIT_STD * rng.normal(size=(P, NH, NH, D)) * M2[..., None]
    H3 = INIT_STD * rng.normal(size=(P, NO, NH, D)) * M3[..., None]
    a_h = np.zeros((P, NH, D)); a_o = np.zeros((P, NO, D))
    ret = np.zeros((P, N_EP)); w_in = p["W_in"].sum(-1)
    for e in range(N_EP):
        s = (rng.random((P, 4)) - 0.5) * 0.1
        live = np.ones(P, dtype=bool); r = np.zeros(P)
        for _ in range(MAX_STEPS):
            a_in = np.tanh((s / OBS_SCALE)[:, :, None] * w_in[:, None, :] + p["b_in"][:, None, :])
            for _ in range(2):
                m_h = (np.einsum('pij,pijd,pjd->pid', F1, H1, a_in) / np.sqrt(K_IN)
                       + np.einsum('pij,pijd,pjd->pid', F2, H2, a_h) / np.sqrt(K_HID))
                a_h = np.where(live[:, None, None],
                               np.tanh(np.einsum('ped,pnd->pne', p["W_hid"], m_h) + p["b_hid"][:, None]), a_h)
                m_o = np.einsum('pij,pijd,pjd->pid', F3, H3, a_h) / np.sqrt(K_HID)
                a_o = np.where(live[:, None, None],
                               np.tanh(np.einsum('ped,pnd->pne', p["W_out"], m_o) + p["b_out"][:, None]), a_o)
            right = a_o[:, 1, 0] > a_o[:, 0, 0]
            H1 = gru_batch(H1, M1, p, "ih", a_in, a_h, r, live)
            H2 = gru_batch(H2, M2, p, "hh", a_h, a_h, r, live)
            H3 = gru_batch(H3, M3, p, "ho", a_h, a_o, r, live)
            s2, fell = cart_step_batch(s, right)
            ret[:, e] += live
            s = np.where(live[:, None], s2, s)
            live = live & ~fell
            r = np.ones(P)
            if not live.any():
                break
    w = np.arange(1, N_EP + 1) / np.arange(1, N_EP + 1).sum()
    return (ret / MAX_STEPS * w).sum(1), ret


class CMAES:
    def __init__(self, n, lam, sigma0, rng):
        self.n, self.lam, self.mu, self.rng = n, lam, lam // 2, rng
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum()
        me = self.me = 1 / np.sum(self.w ** 2)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.zeros(n); self.sigma = sigma0; self.C = np.eye(n)
        self.pc = np.zeros(n); self.ps = np.zeros(n); self.B = np.eye(n); self.D = np.ones(n); self.g = 0

    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.n))
        self.Y = (Z * self.D) @ self.B.T
        return self.m + self.sigma * self.Y

    def tell(self, fit):
        Ys = self.Y[np.argsort(-fit)[:self.mu]]; yw = self.w @ Ys
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * (self.B @ ((self.B.T @ yw) / self.D))
        psn = np.linalg.norm(self.ps)
        hs = float(psn / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Ys.T * self.w) @ Ys)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= np.exp((self.cs / self.ds) * (psn / self.chiN - 1))
        self.g += 1
        ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-20))


class Evo:
    """CMA-ES over the 565-number genome. One call = one real generation."""
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.es = CMAES(DIM, POP, 0.5, self.rng)
        self.gen = 0; self.last = None; self.hist = []
        u = np.random.default_rng(99).normal(size=DIM); self.u_rand = u / np.linalg.norm(u)

    def run_generation(self):
        m_old, sig_old = self.es.m.copy(), self.es.sigma
        X = self.es.ask()
        fit, ret = lives_batch(X, self.rng)
        self.es.tell(fit)
        self.gen += 1
        self.last = dict(X=X, fit=fit, ret=ret, m_old=m_old, m_new=self.es.m.copy(), sigma=sig_old,
                         order=np.argsort(-fit), sigma_new=self.es.sigma)
        self.hist.append((fit.mean(), fit.max()))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
PAGES = [
    dict(title="The big picture", hl="none",
         eqs=[r"$\theta\ \rightarrow\ \mathrm{building\ blocks}\ \rightarrow\ \mathrm{random\ wiring}"
              r"\ \rightarrow\ \mathrm{life}\ \rightarrow\ F(\theta)\ \rightarrow\ \mathrm{CMA{-}ES}\ \rightarrow\ \theta'$"],
         words="Two loops. OUTER loop (evolution): change the genome \u03b8, a list of 565 numbers that describe "
               "the building blocks. INNER loop (one life): build a network with random wiring from those "
               "blocks, let it act in CartPole for 8 episodes while every synapse keeps learning, and give "
               "the life a score F. Pages 3 to 9 are ONE time step of the inner loop, in the order the "
               "computer does it. Pages 12 to 14 are the outer loop: 128 genomes, 128 separate lives, one "
               "CMA-ES update.",
         note="Paper: SFNN = evolved neuron and synapse types, random sparse wiring every lifetime, GRU "
              "synapses that learn during life, CMA-ES with population 128.",
         panel="overview"),
    dict(title="1. The genome", hl="dim",
         eqs=[r"$\theta = [\,W_{in},b_{in}\,|\,W_{hid},b_{hid}\,|\,W_{out},b_{out}\,|\,"
              r"\mathrm{GRU}_{ih}\,|\,\mathrm{GRU}_{hh}\,|\,\mathrm{GRU}_{ho}\,|\,\theta_\eta\,]$",
              r"$3\times(16+4)\ +\ 3\times 3\times(4\cdot 9 + 4\cdot 4 + 4)\ +\ 1\ =\ 565$"],
         words="The genome holds only BUILDING BLOCKS: one small layer per neuron type (input, hidden, output), "
               "one learning rule (a GRU) per synapse type, and one learning rate. Every neuron of a type uses "
               "the same numbers; every synapse of a type uses the same rule. No connection weight is stored, "
               "so the genome is 565 numbers whether the network has 4 or 32 or 1000 hidden neurons.",
         note="Paper: 3 neuron types, 3 synapse types, 565 parameters in total. Activity of each neuron is a "
              "vector of 4 numbers.",
         panel="genome"),
    dict(title="2. New wiring for each life", hl="wiring",
         eqs=[r"$M_{ij} \sim \mathrm{Bernoulli}(0.5)$       (does the link j \u2192 i exist?)",
              r"$\mathbf{h}_{ij}(0) \sim \mathcal{N}(0,\ 0.5^2)\ \in \mathbb{R}^4$       (synapse state at birth)"],
         words="At the start of every life, half of all possible links (sensor\u2192hidden, hidden\u2192hidden, "
               "hidden\u2192output) are switched off at random. Each remaining link j\u2192i gets a 4-number state "
               "h_ij that works as its weight. Because the wiring is new every life, evolution cannot rely on "
               "any particular link: the building blocks must work for ANY wiring. That is what makes the "
               "network structurally flexible.",
         note="Paper: random sparse adjacency with half of the connections set to zero, a new one per lifetime. "
              "Starting synapse values: this demo's choice.",
         panel="wiring"),
    dict(title="3. Sensors \u2192 input neurons", hl="inputs",
         eqs=[r"$o_i = s_i\ /\ c_i$",
              r"$\mathbf{a}_i = \tanh\left(W_{in}\,(o_i\,\mathbf{1}) + \mathbf{b}_{in}\right)\ \in \mathbb{R}^4$"],
         words="Each of CartPole's 4 numbers goes into its own input neuron. The value is first divided by its "
               "typical size c_i, then the input neuron type turns the single number into a 4-number activity "
               "vector a_i. A task with 6 sensors would simply use 6 input neurons of the same type.",
         note="Paper: each input neuron copies one observation element; neurons are small linear layers with "
              "tanh. Dividing by c_i: simplified here.",
         panel="inputs"),
    dict(title="4. Message into a hidden neuron", hl="msg",
         eqs=[r"$\mathbf{m}_i = \frac{1}{\sqrt{k_{in}}}\sum_{j\in in} M_{ij}\,(\mathbf{h}_{ij}\odot\mathbf{a}_j)"
              r"\ +\ \frac{1}{\sqrt{k_{hid}}}\sum_{j\in hid} M_{ij}\,(\mathbf{h}_{ij}\odot\mathbf{a}_j)$"],
         words="Every existing link multiplies the sender's activity by its own synapse state, number by number "
               "(\u2299), and the hidden neuron adds up what arrives from sensors and from other hidden neurons. "
               "k is the expected number of links (2 here), so bigger networks do not get bigger sums. The "
               "highlighted neuron and its incoming links are the ones in the numbers below.",
         note="Paper: synapse states act as the connection weights between vector-valued neurons. The exact "
              "sum and scaling: this demo's choice.",
         panel="message"),
    dict(title="5. Neuron update (2 ticks)", hl="hidden",
         eqs=[r"$\mathbf{a}_i \leftarrow \tanh(W_{hid}\,\mathbf{m}_i + \mathbf{b}_{hid})$       (hidden)",
              r"$\mathbf{a}_k \leftarrow \tanh(W_{out}\,\mathbf{m}_k + \mathbf{b}_{out})$       (output)"],
         words="The neuron type's small layer (4\u00d74 weights + 4 biases, from the genome) turns the message "
               "into the new activity. Messages and updates run TWICE per time step (two ticks), so signals "
               "can also pass hidden\u2192hidden before the action is read.",
         note="Paper: neuron = small linear layer + tanh; 2 micro-ticks per environment step.",
         panel="neuron"),
    dict(title="6. Choose the action", hl="output",
         eqs=[r"$\mathrm{action} = \arg\max_{k\,\in\,\{left,\,right\}}\ a_k^{(1)}$"],
         words="Only the FIRST of the 4 numbers of each output neuron is read. The bigger one wins: push the "
               "cart left or right with 10 N. A task with 3 actions would use 3 output neurons of the same type.",
         note="Paper: the first element of the output neurons' activations determines the action.",
         panel="action"),
    dict(title="7. Learning: GRU gates", hl="synapse",
         eqs=[r"$\mathbf{x}_{ij} = [\,\mathbf{a}_j,\ \mathbf{a}_i,\ r_{t-1}\,]\ \in \mathbb{R}^9$",
              r"$\mathbf{z} = \sigma(W_z\mathbf{x} + U_z\mathbf{h} + \mathbf{b}_z)$       (how much to KEEP the old state)",
              r"$\rho = \sigma(W_\rho\mathbf{x} + U_\rho\mathbf{h} + \mathbf{b}_\rho)$       (how much old state to USE)"],
         words="Now EVERY synapse updates itself, all at the same time, with its type's rule. The rule sees "
               "only local information: the sender's activity a_j, the receiver's activity a_i, and the last "
               "reward. A GRU first computes two 'gates' between 0 and 1. The numbers below follow ONE "
               "synapse: the highlighted link.",
         note="Paper: each synapse vector is the hidden state of a GRU fed pre- and post-synaptic activity and "
              "the previous reward. In CartPole the reward is 1 every step (0 on an episode's first step).",
         panel="gates"),
    dict(title="8. Learning: new synapse state", hl="synapse",
         eqs=[r"$\mathbf{n} = \tanh(W_n\mathbf{x} + \rho\odot(U_n\mathbf{h}) + \mathbf{b}_n)$       (suggested new value)",
              r"$\tilde{\mathbf{h}} = (1-\mathbf{z})\odot\mathbf{n} + \mathbf{z}\odot\mathbf{h}$",
              r"$\mathbf{h} \leftarrow \mathbf{h} + \eta\,(\tilde{\mathbf{h}} - \mathbf{h}),\qquad \eta = \sigma(\theta_\eta - 2)$"],
         words="The GRU mixes a suggested new value n with the old state, using the gate z. The learning rate "
               "\u03b7 (from the genome) sets how far the synapse actually moves this step. This new h is the "
               "weight used on the NEXT time step. This is the only way weights change: there is no backprop.",
         note="Paper: GRU update added to the previous state, scaled by an evolved learning rate. Moving TOWARD "
              "the GRU output (as written here) is this demo's simplification.",
         panel="state"),
    dict(title="9. CartPole physics", hl="dim",
         eqs=[r"$\tau = \frac{F + m_p l\,\dot{\phi}^2\sin\phi}{m_c+m_p},\qquad"
              r"\ddot{\phi} = \frac{g\sin\phi - \tau\cos\phi}{l\,(4/3 - m_p\cos^2\phi\,/\,(m_c+m_p))}$",
              r"$\ddot{x} = \tau - \frac{m_p l\,\ddot{\phi}\cos\phi}{m_c+m_p}$",
              r"$x \leftarrow x+\Delta t\,\dot{x},\ \ \dot{x} \leftarrow \dot{x}+\Delta t\,\ddot{x},\ \ "
              r"\phi \leftarrow \phi+\Delta t\,\dot{\phi},\ \ \dot{\phi} \leftarrow \dot{\phi}+\Delta t\,\ddot{\phi}$",
              r"$r_t = 1;\quad \mathrm{fall\ if}\ |x|>2.4\ \mathrm{or}\ |\phi|>12^\circ$"],
         words="The chosen push F = \u00b110 N moves the cart and the pole for \u0394t = 0.02 s. The reward is +1 "
               "for every step the pole stays up. The new state becomes the input of the next time step "
               "(page 3). Constants: m_c = 1 kg, m_p = 0.1 kg, l = 0.5 m, g = 9.8.",
         note="Standard CartPole-v1 (Gym/Gymnax), as used in the paper.",
         panel="physics"),
    dict(title="10. Repeat: one whole life", hl="none",
         eqs=[r"$\mathrm{for}\ e = 1..8:\quad \mathrm{for}\ t = 1..T_e:\quad \mathrm{pages\ 3\rightarrow 9}$",
              r"$R_e = \mathrm{number\ of\ steps\ the\ pole\ stayed\ up\ in\ episode}\ e$"],
         words="Pages 3 to 9 repeat every time step until the pole falls (or 200 steps), then a new episode "
               "starts. The network is NOT reset between episodes: synapse states and activities carry on, so "
               "what it learned in episode 1 can help in episode 8. Press 'Run to end of episode' to fill "
               "the chart.",
         note="Paper: 8 episodes per lifetime, network state kept between episodes (CartPole-v1: up to 500 "
              "steps; 200 here).",
         panel="life"),
    dict(title="11. Lifetime score", hl="none",
         eqs=[r"$F = \sum_{e=1}^{8} w_e\,\frac{R_e}{R_{max}},\qquad w_e = \frac{e}{\sum_{k=1}^{8} k}$",
              r"$\mathrm{paper,\ 3\ tasks}:\ \ F = \prod_{\mathrm{task}} \hat{F}_{\mathrm{task}}\ \ (\mathrm{each\ scaled\ to}\ 0..1)$"],
         words="Later episodes count more, so evolution rewards genomes that IMPROVE during life, not only ones "
               "that start well. In the paper, one genome lives one life in each of the 3 tasks and the scores "
               "are multiplied, so it must work on all three.",
         note="Paper: later episodes weighted more; scores rescaled to (0,1) and multiplied across tasks. "
              "Weights e/36: this demo's choice.",
         panel="score"),
    dict(title="12. One generation: 128 robots", hl="none",
         eqs=[r"$\theta_k = \mathbf{m} + \sigma\,\mathbf{y}_k,\quad \mathbf{y}_k \sim \mathcal{N}(0,\,C),\quad k = 1..128$",
              r"$F_k = \mathrm{score\ of\ the\ life\ of\ robot}\ k$"],
         words="CMA-ES makes 128 genomes: its current best guess (the mean m) plus 128 different random "
               "changes. Each genome builds its OWN complete network: its own 4 sensor neurons, hidden neurons, "
               "2 action neurons and random wiring. Each lives its own life of 8 episodes and gets one score. "
               "The 128 robots never talk to each other; a computer just runs them side by side. Press "
               "'Run 1 generation' to do it for real, then click a square to load that robot on the left.",
         note="Paper: population 128, and each genome also lives in Acrobot and MountainCar (3 scores multiplied). "
              "Here: CartPole only and 4 hidden neurons, so one generation takes a few seconds.",
         panel="population"),
    dict(title="13. CMA-ES moves the mean", hl="none",
         eqs=[r"$\mathbf{m} \leftarrow \mathbf{m} + \sigma \sum_{i=1}^{64} w_i\,\mathbf{y}_{i:128}$",
              r"$C \leftarrow (1-c_1-c_\mu)\,C + c_1\,\mathbf{p}_c\mathbf{p}_c^T + c_\mu \sum_i w_i\,\mathbf{y}_{i:128}\mathbf{y}_{i:128}^T$",
              r"$\sigma \leftarrow \sigma\,\exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\Vert\mathbf{p}_\sigma\Vert}"
              r"{E\Vert\mathcal{N}(0,I)\Vert} - 1\right)\right)$"],
         words="The 128 scores are ranked. The mean moves toward the best 64 (the better, the more pull; "
               "y_{i:128} = the change of the i-th best). C learns which directions worked and σ grows or "
               "shrinks the cloud. The plot squeezes the 565-number cloud into 2 directions: across = the way "
               "the mean actually moved, up = some other direction.",
         note="Paper: CMA-ES, population 128, 565 parameters. The 2-D picture is only a projection for display; "
              "the real search happens in 565 dimensions.",
         panel="cma"),
    dict(title="14. Kept vs thrown away", hl="none",
         eqs=[r"$\mathrm{kept:}\ \ \mathbf{m},\ C,\ \sigma$",
              r"$\mathrm{thrown\ away:}\ \ 128\ \mathrm{networks,\ their\ wiring,\ weights}\ \mathbf{h}_{ij},\ \mathrm{activities}\ \mathbf{a}_i$"],
         words="After the CMA-ES step all 128 networks are deleted, including every weight they learned during "
               "their lives. Only the search cloud (mean m, shape C, size σ) is kept. The next generation "
               "starts again from random wiring and random weights, but with better RULES for learning them. "
               "After training, the final genome (the mean) is used the same way: a new random network that "
               "learns during its life.",
         note="Nothing learned within a life is inherited; what improves over generations is the genome, "
              "i.e. how to learn.",
         panel="kept"),
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def screen_inches():
    try:
        import tkinter
        root = tkinter.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        dpi = plt.rcParams["figure.dpi"]
        return max(10, 0.94 * w / dpi), max(6.5, 0.86 * h / dpi)
    except Exception:
        return 16, 9


class App:
    def __init__(self, size=None):
        for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.grid", "keymap.yscale"):
            try:
                plt.rcParams[k] = []
            except KeyError:
                pass
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                             "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
                             "axes.labelcolor": INK2, "mathtext.fontset": "cm"})
        self.sim = Sim(0)
        self.evo = Evo(1); self.robot = None; self.pop_art = None
        self.page = 0
        self.fig = plt.figure(figsize=size or screen_inches())
        try:
            self.fig.canvas.manager.set_window_title("SFNN step by step")
        except AttributeError:
            pass
        f = self.fig
        self.axList = f.add_axes([0.008, 0.10, 0.155, 0.83])
        self.axNet = f.add_axes([0.175, 0.30, 0.34, 0.60])
        self.axCart = f.add_axes([0.175, 0.10, 0.34, 0.17])
        self.axEq = f.add_axes([0.535, 0.47, 0.455, 0.43])
        self.axNum = f.add_axes([0.535, 0.10, 0.455, 0.35])
        self.title = f.text(0.01, 0.965, "", va="center", fontweight="bold")
        self.status = f.text(0.99, 0.965, "", va="center", ha="right", color=INK2)
        self._buttons()
        f.canvas.mpl_connect("key_press_event", self._key)
        f.canvas.mpl_connect("pick_event", self._pick)
        f.canvas.mpl_connect("resize_event", lambda e: self.draw())
        self.draw()

    # ---------------- controls ----------------
    def _buttons(self):
        f = self.fig; y, h = 0.018, 0.05
        spec = [("\u25c0  Back", 0.008, 0.07, lambda: self.go(-1)),
                ("Next  \u25b6", 0.083, 0.07, lambda: self.go(+1)),
                ("Run 1 time step", 0.16, 0.095, self.run_step),
                ("Run to end of episode", 0.26, 0.12, self.run_episode),
                ("Next hidden neuron", 0.385, 0.1, self.next_neuron),
                ("New wiring (new life)", 0.49, 0.115, self.new_wiring),
                ("New genome", 0.61, 0.075, self.new_genome),
                ("Run 1 generation", 0.69, 0.1, self.run_generation)]
        self.btns = []
        for label, x, w, fn in spec:
            b = Button(f.add_axes([x, y, w, h]), label, color="#ecebe7", hovercolor="#dcdad3")
            b.on_clicked(lambda e, fn=fn: fn())
            self.btns.append(b)
        self.help = f.text(0.797, y + h / 2, "keys: \u2190 \u2192 pages, space = 1 step,\ne = episode, "
                           "h = next neuron,\nw = new wiring, g = new genome,\nn = 1 generation",
                           va="center", color=INK2)

    def _key(self, e):
        {"right": lambda: self.go(1), "left": lambda: self.go(-1), " ": self.run_step,
         "e": self.run_episode, "h": self.next_neuron, "w": self.new_wiring,
         "g": self.new_genome, "n": self.run_generation}.get(e.key, lambda: None)()

    def _pick(self, e):
        if e.artist in self.list_texts:
            self.page = self.list_texts.index(e.artist); self.draw()
        elif self.pop_art is not None and e.artist is self.pop_art and self.evo.last is not None:
            self.load_robot(int(e.ind[0])); self.draw()

    def load_robot(self, k):
        """Put robot k of the last generation into the step-by-step simulation (new life, own wiring)."""
        self.robot = k
        self.sim.set_genome(self.evo.last["X"][k])

    def run_generation(self):
        self.title.set_text("running 128 lives ..."); self.fig.canvas.draw(); self.fig.canvas.flush_events()
        self.evo.run_generation()
        self.load_robot(int(self.evo.last["order"][0]))         # show the best robot of this generation
        if self.page < 12:
            self.page = 12
        self.draw()

    def go(self, d):
        self.page = (self.page + d) % len(PAGES); self.draw()

    def run_step(self):
        self.sim.step(); self.draw()

    def run_episode(self):
        self.sim.run_episode(); self.draw()

    def next_neuron(self):
        self.sim.focus = (self.sim.focus + 1) % NH
        self.sim.redo_last(); self.draw()

    def new_wiring(self):
        self.sim.new_life(); self.draw()

    def new_genome(self):
        self.robot = None
        self.sim.new_genome(); self.draw()

    # ---------------- layout helpers ----------------
    def _fs(self):
        w, h = self.fig.get_size_inches()
        return float(np.clip(min(w / 1.75, h) * 1.12, 7.5, 15))

    def _wrap(self, ax, text, fs):
        wpx = ax.get_window_extent().width
        chars = max(30, int(wpx / (fs * self.fig.dpi / 72 * 0.52)))
        return "\n".join(textwrap.fill(par, chars) for par in text.split("\n"))

    @staticmethod
    def _blank(ax):
        ax.cla(); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # ---------------- drawing ----------------
    def draw(self):
        fs = self._fs(); self.fs = fs
        pg = PAGES[self.page]; sim = self.sim; rec = sim.rec
        self.title.set_text(f"SFNN step by step   \u2014   page {self.page}/{len(PAGES) - 1}:  {pg['title']}")
        self.title.set_fontsize(fs * 1.25)
        self.status.set_text(f"life: episode {rec['ep'] + 1}/{N_EP}, step {rec['t_ep'] + 1} of this episode "
                             f"(time step {rec['t'] + 1} of the life)    returns so far: "
                             + " ".join(str(int(v)) for v in sim.returns[:min(N_EP, sim.ep + 1)])
                             + f"    |    generation {self.evo.gen}"
                             + ("" if self.robot is None else f", showing robot #{self.robot + 1}"))
        self.status.set_fontsize(fs * 0.9)
        for b in self.btns:
            b.label.set_fontsize(fs * 0.9)
        self.help.set_fontsize(fs * 0.8)
        self._draw_list(fs); self._draw_net(pg["hl"], fs); self._draw_cart(self.axCart, rec, fs)
        self._draw_eq(pg, fs)
        self._blank(self.axNum)
        getattr(self, "panel_" + pg["panel"])(self.axNum, rec, fs)
        self.fig.canvas.draw_idle()

    def _draw_list(self, fs):
        ax = self.axList; self._blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        self.list_texts = []
        n = len(PAGES); dy = 0.9 / (n + 0.5)
        ax.text(0.02, 1.0, "the loop, in order", fontsize=fs * 0.85, color=INK2, va="top")
        for i, pg in enumerate(PAGES):
            y = 1 - (i + 1.1) * dy
            cur = i == self.page
            if cur:
                ax.add_patch(FancyBboxPatch((0.0, y - dy * 0.42), 0.99, dy * 0.84, boxstyle="round,pad=0.005",
                                            fc=HILITE, ec="none"))
            inner = 3 <= i <= 9 or i >= 12
            t = ax.text(0.04 + (0.06 if inner else 0), y, pg["title"], fontsize=fs * (0.92 if cur else 0.85),
                        va="center", color=INK if cur else INK2, fontweight="bold" if cur else "normal", picker=True)
            self.list_texts.append(t)
        y3, y9 = 1 - 4.1 * dy, 1 - 10.1 * dy
        ax.plot([0.035, 0.035], [y9 - dy * 0.3, y3 + dy * 0.3], color=BLUE, lw=2)
        y12, y14 = 1 - 13.1 * dy, 1 - 15.1 * dy
        ax.plot([0.035, 0.035], [y14 - dy * 0.3, y12 + dy * 0.3], color=ORANGE, lw=2)
        ax.text(0.02, 0.0, "blue bar: pages 3\u20139 = one time step, repeated all life\n"
                "orange bar: pages 12\u201314 = evolution (outer loop)",
                fontsize=fs * 0.7, color=INK2, va="bottom")

    def _node_pos(self):
        P = {}
        for i in range(NI):
            P[("in", i)] = (0.08, 0.86 - i * 0.24)
        for i in range(NH):
            P[("hid", i)] = (0.5, 0.86 - i * 0.24)
        for i in range(NO):
            P[("out", i)] = (0.9, 0.62 - i * 0.28)
        return P

    def _draw_net(self, hl, fs):
        ax = self.axNet; self._blank(ax); ax.set_xlim(-0.08, 1.06); ax.set_ylim(-0.04, 1.02)
        sim, rec = self.sim, self.sim.rec; P = self._node_pos(); f, j = sim.focus, rec["j"]
        lim = max(0.3, max(np.abs(H).max() for H in (sim.H1, sim.H2, sim.H3)))

        def on(kind, a, b=None):
            if hl in ("none", "wiring"):
                return True
            if hl == "dim":
                return False
            if hl == "inputs":
                return kind == "node" and a[0] == "in"
            if hl == "msg":
                if kind == "node":
                    return a == ("hid", f) or (a[0] == "in" and sim.M1[f, a[1]]) or (a[0] == "hid" and sim.M2[f, a[1]])
                return b == ("hid", f)
            if hl == "hidden":
                return kind == "node" and a[0] in ("hid", "out")
            if hl == "output":
                return kind == "node" and a[0] == "out"
            if hl == "synapse":
                return (kind == "node" and a in (("in", j), ("hid", f))) or (kind == "edge" and a == ("in", j) and b == ("hid", f))
            return True

        def edge(a, b, h, rad=0.0):
            w = h.mean(); mag = 0.45 + 0.55 * min(1, abs(w) / lim)
            c = DIVERGE(0.5 + 0.5 * np.sign(w) * mag)
            act = on("edge", a, b)
            focus_syn = hl in ("synapse", "msg") and act
            arr = FancyArrowPatch(P[a], P[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                  mutation_scale=fs * (1.6 if focus_syn else 1.0), shrinkA=fs * 1.4, shrinkB=fs * 1.4,
                                  color=c if not focus_syn else PURPLE,
                                  lw=(3.5 if focus_syn else 0.8 + 2.5 * min(1, abs(w) / lim)), alpha=1 if act else 0.12,
                                  zorder=2)
            ax.add_patch(arr)

        if hl == "wiring":     # show the links that were switched off
            for i in range(NH):
                for jj in range(NI):
                    if not sim.M1[i, jj]:
                        ax.plot(*zip(P[("in", jj)], P[("hid", i)]), color=FAINT, lw=0.8, ls=":", zorder=1)
                for k in range(NO):
                    if not sim.M3[k, i]:
                        ax.plot(*zip(P[("hid", i)], P[("out", k)]), color=FAINT, lw=0.8, ls=":", zorder=1)
        for i in range(NH):
            for jj in range(NI):
                if sim.M1[i, jj]:
                    edge(("in", jj), ("hid", i), sim.H1[i, jj])
            for jj in range(NH):
                if sim.M2[i, jj]:
                    edge(("hid", jj), ("hid", i), sim.H2[i, jj], rad=0.55 if jj < i else 0.55)
            for k in range(NO):
                if sim.M3[k, i]:
                    edge(("hid", i), ("out", k), sim.H3[k, i])
        acts = {("in", i): rec["a_in"][i] for i in range(NI)}
        acts.update({("hid", i): rec["a_h"][i] for i in range(NH)})
        acts.update({("out", i): rec["a_o"][i] for i in range(NO)})
        names = {("in", i): IN_LABELS[i] for i in range(NI)}
        names.update({("hid", i): f"$h_{i + 1}$" for i in range(NH)})
        names.update({("out", 0): "left", ("out", 1): "right"})
        for key, (x, y) in P.items():
            act = on("node", key)
            col = {"in": GREEN, "hid": BLUE, "out": ORANGE}[key[0]]
            ring = PURPLE if (hl in ("msg", "synapse") and key == ("hid", f)) else col
            ax.scatter([x], [y], s=(fs * 3.2) ** 2, color=SURFACE, edgecolors=ring,
                       linewidths=3.2 if ring == PURPLE else 2, zorder=4, alpha=1 if act else 0.3)
            ax.text(x, y, names[key], ha="center", va="center", fontsize=fs * (1.05 if key[0] != "out" else 0.8),
                    zorder=5, alpha=1 if act else 0.3)
            # activity vector: 4 little squares under the node
            v = acts[key]; sw = 0.032
            for q in range(D):
                ax.add_patch(Rectangle((x - 2 * sw + q * sw, y - 0.105), sw * 0.92, 0.035,
                                       color=DIVERGE(0.5 + 0.5 * np.clip(v[q], -1, 1)), zorder=4,
                                       alpha=1 if act else 0.3, ec=FAINT, lw=0.5))
        ax.text(0.08, 1.0, "sensors (input type)", ha="center", fontsize=fs * 0.8, color=GREEN)
        ax.text(0.5, 1.0, "hidden (hidden type)", ha="center", fontsize=fs * 0.8, color=BLUE)
        ax.text(0.9, 0.80, "actions\n(output type)", ha="center", fontsize=fs * 0.8, color=ORANGE)
        ax.text(-0.07, -0.03, "squares under a neuron = its 4 activity numbers (blue +, orange \u2212)\n"
                "link colour = synapse state (blue +, orange \u2212);  purple = followed in the numbers",
                va="top",
                fontsize=fs * 0.72, color=INK2)

    def _draw_cart(self, ax, rec, fs, s=None, title=True):
        self._blank(ax)
        s = rec["s"] if s is None else s
        bb = ax.get_window_extent()
        ax.set_xlim(-2.7, 2.7); top = -0.25 + 5.4 * bb.height / max(1, bb.width)
        ax.set_ylim(-0.25, max(1.25, top))
        if top < 1.25:                       # axis too flat: widen x instead
            half = 1.5 * bb.width / max(1, bb.height) / 2
            ax.set_xlim(-max(2.7, half), max(2.7, half))
        ax.plot([-2.4, 2.4], [0, 0], color=INK2, lw=1)
        for xx in (-2.4, 2.4):
            ax.plot([xx, xx], [-0.08, 0.08], color=RED, lw=2)
        x, ph = s[0], s[2]
        ax.add_patch(Rectangle((x - 0.25, -0.07), 0.5, 0.14, color=INK2))
        ax.plot([x, x + np.sin(ph)], [0.07, 0.07 + np.cos(ph)], color=BLUE, lw=fs * 0.45, solid_capstyle="round")
        if title:
            push = "right \u2192" if rec["right"] else "\u2190 left"
            ax.text(0.0, 1.0, f"CartPole at this step:  x = {s[0]:+.2f} m,  \u03c6 = {np.degrees(s[2]):+.1f}\u00b0,  "
                    f"chosen push: {push}", fontsize=fs * 0.85, va="top", transform=ax.transAxes)

    def _draw_eq(self, pg, fs):
        ax = self.axEq; self._blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0.0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
                                    fc="#f3f2ee", ec="none", transform=ax.transAxes))
        y = 0.95
        neq = len(pg["eqs"])
        efs = fs * (1.35 if neq <= 2 else 1.25 if neq == 3 else 1.15)
        h_pt = ax.get_window_extent().height * 72 / self.fig.dpi
        for e in pg["eqs"]:
            ax.text(0.025, y, e, fontsize=efs, va="top")
            tall = "\\frac" in e or "\\sum" in e or "\\prod" in e
            y -= (3.0 if tall else 2.0) * efs / h_pt
        y -= 0.02
        t = ax.text(0.025, y, self._wrap(ax, pg["words"], fs), fontsize=fs, va="top", color=INK, linespacing=1.35)
        ax.text(0.025, 0.03, self._wrap(ax, pg["note"], fs * 0.85), fontsize=fs * 0.82, va="bottom",
                color=INK2, style="italic", linespacing=1.3)

    # ---------------- numbers panels ----------------
    def _mono(self, ax, x, y, text, fs, **kw):
        return ax.text(x, y, text, fontsize=fs * 0.92, family="monospace", va="top", transform=ax.transAxes,
                       linespacing=1.45, **kw)

    def _head(self, ax, text, fs):
        ax.text(0.0, 1.0, text, fontsize=fs, fontweight="bold", va="top", transform=ax.transAxes)

    def panel_overview(self, ax, rec, fs):
        self._head(ax, "One generation of SFNN training", fs)
        steps = ["genome \u03b8\n565 numbers", "build\nblocks", "random\nwiring",
                 "live 8 episodes,\nlearning\nevery step", "score F", "CMA-ES\nmoves \u03b8"]
        n = len(steps); xs = [(i + 0.5) / n for i in range(n)]
        for i, (x, t) in enumerate(zip(xs, steps)):
            ax.text(x, 0.55, t, ha="center", va="center", fontsize=fs * 0.82, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.45", fc=HILITE if 2 <= i <= 4 else "#ecebe7", ec="none"))
            if i < n - 1:
                ax.annotate("", xy=(xs[i + 1] - 0.072, 0.55), xytext=(x + 0.072, 0.55), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
        ax.annotate("", xy=(xs[0], 0.36), xytext=(xs[-1], 0.36), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, connectionstyle="arc3,rad=-0.12"))
        ax.text(0.5, 0.12, "repeat for thousands of generations (outer loop)", ha="center", color=BLUE,
                fontsize=fs * 0.85, transform=ax.transAxes)
        ax.text(0.5, 0.82, "yellow = the inner loop (one life), shown step by step on pages 2\u201311",
                ha="center", color=INK2, fontsize=fs * 0.8, transform=ax.transAxes)

    def panel_genome(self, ax, rec, fs):
        self._head(ax, "This genome (random here; evolution would tune it)", fs)
        th = self.sim.theta
        sub = ax.inset_axes([0.0, 0.62, 1.0, 0.18])
        sub.imshow(th[None], aspect="auto", cmap=DIVERGE, vmin=-1.5, vmax=1.5, interpolation="nearest")
        for e in EDGES[1:-1]:
            sub.axvline(e - 0.5, color=INK, lw=0.8)
        sub.set_yticks([]); sub.set_xticks((EDGES[:-1] + EDGES[1:]) / 2)
        sub.set_xticklabels(["in", "hid", "out", "rule in\u2192hid", "rule hid\u2192hid", "rule hid\u2192out", "\u03b7"],
                            fontsize=fs * 0.75)
        lines = [f"{n:<32}{c:>4} numbers" for n, c in BLOCKS] + [f"{'total':<32}{DIM:>4}"]
        self._mono(ax, 0.0, 0.42, "\n".join(lines), fs * 0.9)
        p = self.sim.p
        self._mono(ax, 0.62, 0.42, "hidden type W_hid (4x4):\n" + "\n".join(vec(r) for r in p["W_hid"])
                   + f"\nlearning rate \u03b7 = {p['lr']:.3f}", fs * 0.9)

    def panel_wiring(self, ax, rec, fs):
        sim = self.sim
        self._head(ax, "This life's wiring (dotted = switched off)", fs)
        txt = (f"sensor \u2192 hidden links : {sim.M1.sum():>2} of {NI * NH}\n"
               f"hidden \u2192 hidden links : {sim.M2.sum():>2} of {NH * (NH - 1)}\n"
               f"hidden \u2192 output links : {sim.M3.sum():>2} of {NH * NO}\n\n"
               f"followed synapse: {IN_NAMES[rec['j']]} \u2192 h{sim.focus + 1}\n"
               f"  state at birth h(0) = {vec(sim.H1_birth[sim.focus, rec['j']])}\n"
               f"  state now           = {vec(sim.H1[sim.focus, rec['j']])}")
        self._mono(ax, 0.0, 0.86, txt, fs)
        ax.text(0.0, 0.05, "Press 'New wiring' to see a different random network built from the SAME genome.",
                fontsize=fs * 0.85, color=INK2, transform=ax.transAxes)

    def panel_inputs(self, ax, rec, fs):
        self._head(ax, "This step: sensors \u2192 input neurons", fs)
        p = self.sim.p
        lines = [f"{'sensor':<16}{'s_i':>7}{'c_i':>6}{'o_i':>7}    a_i = activity (4 numbers)"]
        for i in range(NI):
            lines.append(f"{IN_NAMES[i]:<16}{rec['s'][i]:>+7.3f}{OBS_SCALE[i]:>6.2f}{rec['o'][i]:>+7.3f}    {vec(rec['a_in'][i])}")
        lines += ["", f"W_in \u00b7 1 (row sums) = {vec(p['W_in'].sum(1))}", f"b_in            = {vec(p['b_in'])}"]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs)

    def panel_message(self, ax, rec, fs):
        sim = self.sim; f = sim.focus; tk = rec["ticks"][1]
        self._head(ax, f"Message into h{f + 1} (tick 2 of this step)", fs)
        lines = [f"{'from':<14}{'h_ij (synapse)':<29}{'a_j (sender)':<29}h_ij \u2299 a_j"]
        for jj in range(NI):
            if sim.M1[f, jj]:
                h = rec["ticks"][1]["c_in"][jj] / np.where(rec["a_in"][jj] == 0, 1, rec["a_in"][jj])
                lines.append(f"{IN_NAMES[jj][:13]:<14}{vec(h):<29}{vec(rec['a_in'][jj]):<29}{vec(tk['c_in'][jj])}")
        for jj in range(NH):
            if sim.M2[f, jj]:
                a = tk["a_h_used"][jj]
                h = tk["c_hh"][jj] / np.where(a == 0, 1, a)
                lines.append(f"{'hidden h' + str(jj + 1):<14}{vec(h):<29}{vec(a):<29}{vec(tk['c_hh'][jj])}")
        lines += ["", f"sum from sensors / \u221a{K_IN:.0f} = {vec(tk['c_in'].sum(0) / np.sqrt(K_IN))}",
                  f"sum from hidden  / \u221a{K_HID:.0f} = {vec(tk['c_hh'].sum(0) / np.sqrt(K_HID))}",
                  f"message m_{f + 1}            = {vec(tk['m'])}"]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs * 0.92)
        ax.text(0.0, 0.0, "(the synapse states shown are the ones used in this step, i.e. before this step's learning)",
                fontsize=fs * 0.75, color=INK2, transform=ax.transAxes)

    def panel_neuron(self, ax, rec, fs):
        f = self.sim.focus; t1, t2 = rec["ticks"]
        self._head(ax, f"Neuron h{f + 1} and the outputs, tick by tick", fs)
        lines = [f"h{f + 1} at start of step   a = {vec(rec['a_h_start'][f])}", "",
                 "tick 1:  m            = " + vec(t1["m"]),
                 "         W_hid m + b  = " + vec(t1["pre"]),
                 "         a = tanh(..) = " + vec(t1["a"]),
                 "tick 2:  m            = " + vec(t2["m"]),
                 "         W_hid m + b  = " + vec(t2["pre"]),
                 "         a = tanh(..) = " + vec(t2["a"]), "",
                 "outputs after tick 2:  left  a = " + vec(t2["a_o"][0]),
                 "                       right a = " + vec(t2["a_o"][1])]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs)

    def panel_action(self, ax, rec, fs):
        self._head(ax, "Read the first number of each output neuron", fs)
        a = rec["a_o"][:, 0]
        sub = ax.inset_axes([0.05, 0.1, 0.45, 0.65])
        sub.barh([1, 0], a, color=[ORANGE if not rec["right"] else FAINT, ORANGE if rec["right"] else FAINT], height=0.55)
        sub.set_yticks([1, 0]); sub.set_yticklabels(["left", "right"], fontsize=fs * 0.9)
        sub.set_xlim(-1, 1); sub.axvline(0, color=INK2, lw=0.8); sub.tick_params(labelsize=fs * 0.75)
        for s in ("top", "right"):
            sub.spines[s].set_visible(False)
        self._mono(ax, 0.58, 0.7, f"a_left[1]  = {a[0]:+.3f}\na_right[1] = {a[1]:+.3f}\n\n"
                   f"action = {'RIGHT' if rec['right'] else 'LEFT'}\nforce  = {rec['phys']['F']:+.0f} N", fs * 1.05)

    def panel_gates(self, ax, rec, fs):
        f, j = self.sim.focus, rec["j"]
        self._head(ax, f"Followed synapse: {IN_NAMES[j]} \u2192 h{f + 1}  (rule: input\u2192hidden)", fs)
        x = rec["x"]
        lines = ["x = [ a_j (sender) | a_i (receiver) | r ]",
                 "  = " + vec(x[:4]) + " |", "    " + vec(x[4:8]) + f" | {x[8]:.0f}", "",
                 f"h (state now)  = {vec(rec['h_old'])}", "",
                 f"W_z x  = {vec(rec['Wz_x'])}     U_z h = {vec(rec['Uz_h'])}",
                 f"z = \u03c3(W_z x + U_z h + b_z) = {vec(rec['z'])}", "",
                 f"W_\u03c1 x  = {vec(rec['Wr_x'])}     U_\u03c1 h = {vec(rec['Ur_h'])}",
                 f"\u03c1 = \u03c3(W_\u03c1 x + U_\u03c1 h + b_\u03c1) = {vec(rec['rho'])}"]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs * 0.92)

    def panel_state(self, ax, rec, fs):
        f, j = self.sim.focus, rec["j"]
        self._head(ax, f"Followed synapse: {IN_NAMES[j]} \u2192 h{f + 1}", fs)
        lines = [f"W_n x            = {vec(rec['Wn_x'])}",
                 f"\u03c1 \u2299 (U_n h)      = {vec(rec['rho'] * rec['Un_h'])}",
                 f"n  (suggestion)  = {vec(rec['n'])}",
                 f"z  (keep gate)   = {vec(rec['z'])}",
                 f"h~ = (1-z)n + zh = {vec(rec['ht'])}", "",
                 f"h before         = {vec(rec['h_old'])}",
                 f"\u03b7 (learning rate) = {rec['lr']:.3f}",
                 f"h after          = {vec(rec['h_new'])}",
                 f"change           = {vec(rec['h_new'] - rec['h_old'])}", "",
                 f"all synapses together moved {rec['dH']:.3f} (length of the change) this step"]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs * 0.95)

    def panel_physics(self, ax, rec, fs):
        self._head(ax, "This step's physics", fs)
        ph = rec["phys"]; s, s2 = rec["s"], rec["s2"]
        lines = [f"F = {ph['F']:+.0f} N     \u03c4 = {ph['tau']:+.3f}",
                 f"\u03c6'' = {ph['phdd']:+.3f} rad/s\u00b2     x'' = {ph['xdd']:+.3f} m/s\u00b2", "",
                 f"{'':<14}{'before':>9}{'after':>9}"]
        for i in range(4):
            lines.append(f"{IN_NAMES[i][:13]:<14}{s[i]:>+9.4f}{s2[i]:>+9.4f}")
        fell_txt = "YES \u2192 new episode" if rec["fell"] else "no"
        lines += ["", f"reward r = 1     fell: {fell_txt}"]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs)
        sub = ax.inset_axes([0.58, 0.05, 0.42, 0.75])
        self._draw_cart(sub, rec, fs * 0.8, s=s2, title=False)
        sub.set_title("after the step", fontsize=fs * 0.8, color=INK2)

    def panel_life(self, ax, rec, fs):
        sim = self.sim
        self._head(ax, "Steps survived per episode in this life", fs)
        sub = ax.inset_axes([0.06, 0.12, 0.9, 0.7])
        cols = [BLUE if e < sim.ep else (ORANGE if e == sim.ep and not sim.over else GRID) for e in range(N_EP)]
        sub.bar(range(1, N_EP + 1), sim.returns, color=cols)
        sub.set_xticks(range(1, N_EP + 1)); sub.set_xlabel("episode", fontsize=fs * 0.85)
        sub.set_ylim(0, max(30, sim.returns.max() * 1.25)); sub.tick_params(labelsize=fs * 0.8)
        for s in ("top", "right"):
            sub.spines[s].set_visible(False)
        for e, v in enumerate(sim.returns):
            if v:
                sub.text(e + 1, v, f"{int(v)}", ha="center", va="bottom", fontsize=fs * 0.8)
        msg = "life finished \u2014 next step starts a new life" if sim.over else \
            f"orange = episode in progress ({int(sim.returns[min(sim.ep, N_EP - 1)])} steps so far)"
        sub.set_title(msg, fontsize=fs * 0.8, color=INK2, loc="left")

    def panel_score(self, ax, rec, fs):
        sim = self.sim
        self._head(ax, "Score of this life so far", fs)
        w = np.arange(1, N_EP + 1) / np.arange(1, N_EP + 1).sum()
        lines = [f"{'episode':<9}{'R_e':>5}{'R_e/200':>9}{'w_e':>8}{'w_e*R_e/200':>13}"]
        for e in range(N_EP):
            lines.append(f"{e + 1:<9}{int(sim.returns[e]):>5}{sim.returns[e] / MAX_STEPS:>9.3f}{w[e]:>8.3f}"
                         f"{w[e] * sim.returns[e] / MAX_STEPS:>13.4f}")
        F = (w * sim.returns / MAX_STEPS).sum()
        lines += ["", f"F = {F:.4f}" + ("" if sim.over else "   (life not finished yet)")]
        self._mono(ax, 0.0, 0.86, "\n".join(lines), fs * 0.95)

    def _need_gen(self, ax, fs):
        ax.text(0.5, 0.45, "Press  'Run 1 generation'  (key n)\nto make 128 genomes and let each one live its own life.",
                ha="center", va="center", fontsize=fs * 1.05, color=INK2, transform=ax.transAxes)

    def panel_population(self, ax, rec, fs):
        L = self.evo.last
        self.pop_art = None
        if L is None:
            self._head(ax, "Generation 0: nothing run yet", fs); self._need_gen(ax, fs); return
        fit, order = L["fit"], L["order"]
        rank = np.empty(POP, int); rank[order] = np.arange(POP)
        self._head(ax, f"Generation {self.evo.gen}: 128 separate robots, coloured by score (click one)", fs)
        g = ax.inset_axes([0.0, 0.0, 0.6, 0.86]); g.set_xticks([]); g.set_yticks([])
        for sp in g.spines.values():
            sp.set_visible(False)
        cols, rows = 16, 8
        xs = np.arange(POP) % cols; ys = rows - 1 - np.arange(POP) // cols
        norm = (fit - fit.min()) / max(1e-9, np.ptp(fit))
        cmap = LinearSegmentedColormap.from_list("sc", ["#f3f2ee", "#a9c9ef", "#2a78d6", "#0d3a73"])
        ms = (g.get_window_extent().width / cols * 0.8 * 72 / self.fig.dpi) ** 2
        self.pop_art = g.scatter(xs, ys, s=ms, marker="s", c=cmap(norm), edgecolors=[INK if rank[k] < 64 else FAINT
                                 for k in range(POP)], linewidths=[0.9 if rank[k] < 64 else 0.4 for k in range(POP)],
                                 picker=4)
        if self.robot is not None:
            g.scatter([xs[self.robot]], [ys[self.robot]], s=ms * 1.6, marker="s", facecolors="none",
                      edgecolors=PURPLE, linewidths=2.5)
        for k in range(POP):
            g.text(xs[k], ys[k], f"{fit[k]:.2f}"[1:], ha="center", va="center", fontsize=fs * 0.45,
                   color="white" if norm[k] > 0.55 else INK2)
        g.set_xlim(-0.6, cols - 0.4); g.set_ylim(-0.6, rows - 0.4)
        # one robot = one complete network
        r = ax.inset_axes([0.64, 0.5, 0.36, 0.36]); self._blank(r); r.set_xlim(0, 1); r.set_ylim(0, 1)
        pin, ph, po = [(0.1, 0.85 - i * 0.23) for i in range(4)], [(0.5, 0.85 - i * 0.23) for i in range(4)], \
            [(0.9, 0.62), (0.9, 0.38)]
        rng = np.random.default_rng(5)
        for a in pin:
            for b in ph:
                if rng.random() < 0.5:
                    r.plot(*zip(a, b), color="#0f8f8a", lw=0.8)
        for a in ph:
            for b in po:
                if rng.random() < 0.6:
                    r.plot(*zip(a, b), color="#d6559a", lw=0.8)
        for pts, c in ((pin, GREEN), (ph, BLUE), (po, ORANGE)):
            r.scatter(*zip(*pts), s=(fs * 1.3) ** 2, color=c, zorder=3)
        r.set_title("each square = one whole robot like this\n(own sensors, own wiring, own life)",
                    fontsize=fs * 0.75, color=INK2)
        k = self.robot
        txt = (f"average score : {fit.mean():.3f}\nbest score    : {fit.max():.3f}\n"
               f"black border  = best 64\n\n")
        if k is not None:
            txt += (f"robot #{k + 1}: score {fit[k]:.3f}\n(rank {rank[k] + 1} of 128)\nsteps per episode:\n"
                    + " ".join(str(int(v)) for v in L["ret"][k]))
        self._mono(ax, 0.64, 0.44, txt, fs * 0.88)

    def panel_cma(self, ax, rec, fs):
        L = self.evo.last
        if L is None:
            self._head(ax, "The cloud of 128 genomes around the mean", fs); self._need_gen(ax, fs); return
        self._head(ax, f"Generation {self.evo.gen}: the 128 genomes around the mean (565 numbers squeezed to 2-D)", fs)
        d = L["m_new"] - L["m_old"]; u1 = d / max(1e-12, np.linalg.norm(d))
        u2 = self.evo.u_rand - (self.evo.u_rand @ u1) * u1; u2 /= np.linalg.norm(u2)
        Z = L["X"] - L["m_old"]; px, py = Z @ u1, Z @ u2
        fit, order = L["fit"], L["order"]
        g = ax.inset_axes([0.06, 0.1, 0.52, 0.76])
        g.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            g.spines[sp].set_visible(False)
        top = np.zeros(POP, bool); top[order[:64]] = True
        sc = g.scatter(px[~top], py[~top], c=fit[~top], cmap="Blues", vmin=fit.min(), vmax=fit.max(), s=fs * 3,
                       edgecolors=FAINT, linewidths=0.5, zorder=2)
        g.scatter(px[top], py[top], c=fit[top], cmap="Blues", vmin=fit.min(), vmax=fit.max(), s=fs * 3,
                  edgecolors=INK, linewidths=0.9, zorder=3)
        g.scatter([0], [0], marker="*", s=fs * 30, color=ORANGE, edgecolors=INK, zorder=5, label="old mean")
        dn = np.linalg.norm(d)
        g.scatter([dn], [0], marker="D", s=fs * 10, color=GREEN, edgecolors=INK, zorder=5, label="new mean")
        g.annotate("", xy=(dn, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
        g.set_xlabel("direction the mean moved", fontsize=fs * 0.8)
        g.set_ylabel("another direction", fontsize=fs * 0.8)
        g.tick_params(labelsize=fs * 0.65)
        g.legend(loc="upper left", fontsize=fs * 0.7, frameon=False)
        cb = self.fig.colorbar(sc, cax=ax.inset_axes([0.595, 0.1, 0.012, 0.76])); cb.ax.tick_params(labelsize=fs * 0.6)
        cb.set_label("score", fontsize=fs * 0.7)
        lines = [f"λ = 128   genomes per generation", f"μ = 64    best ones (black border)",
                 f"step moved  |Δm| = {dn:.3f}", f"σ before = {L['sigma']:.3f}",
                 f"σ after  = {L['sigma_new']:.3f}", "",
                 "best ones sit mostly on the",
                 "right: the mean moves toward", "them. Next generation's cloud",
                 "is centred on the new mean."]
        self._mono(ax, 0.685, 0.84, "\n".join(lines), fs * 0.85)

    def panel_kept(self, ax, rec, fs):
        self._head(ax, "Between generations", fs)
        rows = [("CMA-ES mean m (565 numbers)", True), ("covariance C, step size σ", True),
                ("the 128 genomes themselves", False), ("each robot's random wiring", False),
                ("each robot's learned weights h_ij", False), ("neuron activities a_i", False)]
        for i, (name, kept) in enumerate(rows):
            y = 0.8 - i * 0.12
            ax.text(0.0, y, ("KEPT      " if kept else "THROWN AWAY"), color=GREEN if kept else RED,
                    fontsize=fs * 0.85, fontweight="bold", family="monospace", va="center", transform=ax.transAxes)
            ax.text(0.2, y, name, fontsize=fs * 0.88, va="center", transform=ax.transAxes)
        h = ax.inset_axes([0.62, 0.12, 0.37, 0.7])
        h.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            h.spines[sp].set_visible(False)
        if self.evo.hist:
            H = np.array(self.evo.hist); x = np.arange(1, len(H) + 1)
            h.plot(x, H[:, 1], color=BLUE, lw=1.2, alpha=0.6, marker="o", ms=3, label="best robot")
            h.plot(x, H[:, 0], color=BLUE, lw=2.2, marker="o", ms=3, label="average of 128")
            h.legend(fontsize=fs * 0.7, frameon=False, loc="upper left")
            h.set_xlim(0.5, max(5, len(H)) + 0.5)
        else:
            h.text(0.5, 0.5, "no generations yet", ha="center", transform=h.transAxes, color=INK2, fontsize=fs * 0.8)
        h.set_ylim(0, None); h.set_xlabel("generation", fontsize=fs * 0.8); h.tick_params(labelsize=fs * 0.65)
        h.set_title("score per generation", fontsize=fs * 0.8, color=INK2)

def main():
    ap = argparse.ArgumentParser(description="SFNN explained step by step")
    ap.add_argument("--size", type=float, nargs=2, default=None, metavar=("W", "H"), help="figure size in inches")
    ap.add_argument("--save", default=None); ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12, help="(with --save) time steps to run first")
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")
    app = App(tuple(a.size) if a.size else (None if not a.save else (16, 9)))
    if a.save:
        for _ in range(a.steps):
            app.sim.step()
        app.page = a.page; app.draw()
        app.fig.savefig(a.save, dpi=110)
        print("saved", a.save)
        return
    try:                                            # open maximised where the backend allows it
        m = app.fig.canvas.manager
        win = getattr(m, "window", None)
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