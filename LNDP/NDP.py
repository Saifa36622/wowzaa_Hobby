"""
ndp_minimal.py  -  How a Neural Developmental Program (NDP) works, in one small file.

THE BIG IDEA
    The genome is NOT a neural network. It is a short list of RULES.
    The rules GROW a network, the way a few genes grow a brain:

        genome (rules)  --development-->  network  --frozen-->  used for the task
                                                                     |
        evolution  <-------------------- fitness --------------------+

    Evolution never touches a weight directly. It only changes the rules,
    and checks how good the network that grows from them is.

WHAT GROWS
    Every node is a "cell" with a small state vector h (its "cell type", 4 numbers).
    We start with 4 sensor nodes, 2 motor nodes and ONE hidden seed cell.
    Then, for DEV_STEPS rounds:

      1. TALK    every cell updates its state from its own state + the mean state of
                 its neighbours (message passing, one shared rule for all cells)
      2. DIVIDE  every hidden cell asks a shared rule "should I divide?"
                 if yes, a daughter cell is born:
                   - its state = a (shared) function of the mother's state
                   - it copies the mother's connections, and the mother connects to it
    Finally:
      3. WIRE    for every connection i->j, a shared rule looks at the two cells'
                 states and decides (a) keep or delete it, (b) its weight.

    After that the network is FROZEN (that is what makes it an NDP and not an LNDP)
    and it controls a CartPole: keep a pole balanced on a cart for 500 steps.

    Original NDP: Najarro, Sudhakaran & Risi (2023), "Towards Self-Assembling
    Artificial Neural Networks through Neural Developmental Programs".
    Simplifications here: small MLP rules instead of a graph neural network,
    sensor/motor cells exist from the start instead of growing from one cell.

Run:    python ndp_minimal.py                 (evolve, then show growth + fitness)
        python ndp_minimal.py --gens 60 --save ndp.png
Needs:  numpy, matplotlib
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
D = 4              # size of each cell's state h
NI, NO = 4, 2      # CartPole: 4 sensors (x, x_dot, angle, angle_dot), 2 motors (push left / right)
DEV_STEPS = 4      # rounds of growth (1 -> 2 -> 4 -> 8 -> up to MAX_HIDDEN cells)
MAX_HIDDEN = 12
TICKS = 2          # signal passes per CartPole step (sensor -> hidden -> motor)

# The genome: every rule's numbers, flattened into one vector.
GENOME = dict(
    emb_in=(NI, D), emb_out=(NO, D), emb_seed=(D,),    # starting "cell types"
    msg_W=(D, 2 * D), msg_b=(D,),                       # 1. TALK rule
    div_w=(D,), div_b=(),                               # 2. DIVIDE rule: should I divide?
    child_W=(D, D), child_b=(D,),                       # 2. DIVIDE rule: daughter's state
    edge_W=(2, 2 * D), edge_b=(2,),                     # 3. WIRE rule: [weight, keep?]
)
N_PARAMS = sum(int(np.prod(s)) for s in GENOME.values())


def unpack(theta):
    g, i = {}, 0
    for name, shape in GENOME.items():
        n = int(np.prod(shape))
        g[name] = theta[i:i + n].reshape(shape)
        i += n
    return g


# ---------------------------------------------------------------------------
# DEVELOPMENT: the rules grow the network
# ---------------------------------------------------------------------------
def develop(theta):
    """Returns the final weight matrix W (W[i, j] = weight of i -> j) and
    a history of snapshots (states, connections, cell kinds) for drawing."""
    g = unpack(theta)
    H = np.vstack([g["emb_in"], g["emb_out"], g["emb_seed"][None]])   # cell states
    kind = ["in"] * NI + ["out"] * NO + ["hid"]
    seed = NI + NO
    A = np.zeros((len(kind), len(kind)))      # A[i, j] = 1 means a connection i -> j
    A[:NI, seed] = 1                          # sensors -> seed
    A[seed, NI:NI + NO] = 1                   # seed -> motors
    history = [(H.copy(), A.copy(), list(kind), "seed")]

    for step in range(DEV_STEPS):
        # 1. TALK: each cell looks at itself + the average of its neighbours
        nb = (A + A.T) > 0
        mean_nb = nb @ H / np.maximum(nb.sum(1, keepdims=True), 1)
        H = H + 0.5 * (np.tanh(np.concatenate([H, mean_nb], 1) @ g["msg_W"].T + g["msg_b"]) - H)

        # 2. DIVIDE: hidden cells may split in two
        mothers = [i for i, k in enumerate(kind) if k == "hid"]
        for m in mothers:
            if kind.count("hid") >= MAX_HIDDEN:
                break
            if H[m] @ g["div_w"] + g["div_b"] > 0:                       # "yes, divide"
                child_h = np.tanh(g["child_W"] @ H[m] + g["child_b"])   # daughter differentiates
                H = np.vstack([H, child_h]); kind.append("hid")
                c = len(kind) - 1
                A = np.pad(A, ((0, 1), (0, 1)))
                A[:, c] = A[:, m]          # daughter gets the mother's inputs
                A[c, :] = A[m, :]          # ... and the mother's outputs
                A[m, c] = 1                # mother -> daughter
        history.append((H.copy(), A.copy(), list(kind), f"after growth step {step + 1}"))

    # 3. WIRE: the edge rule reads both cells' states -> keep? and weight
    n = len(kind)
    pair = np.concatenate([np.repeat(H[:, None, :], n, 1), np.repeat(H[None, :, :], n, 0)], -1)
    z = pair @ g["edge_W"].T + g["edge_b"]                 # shape (n, n, 2)
    keep = (z[..., 1] > 0) * A
    W = 2 * np.tanh(z[..., 0]) * keep
    history.append((H.copy(), keep, list(kind), "after wiring (weights)"))
    return W, history


# ---------------------------------------------------------------------------
# LIFE: the grown network is FROZEN and used as a controller
# ---------------------------------------------------------------------------
class Brain:
    def __init__(self, W):
        self.W, self.v = W, np.zeros(len(W))

    def act(self, obs):
        for _ in range(TICKS):
            self.v[:NI] = obs
            self.v = np.tanh(self.v @ self.W)     # nothing here ever changes W: frozen
        return int(np.argmax(self.v[NI:NI + NO]))


def cartpole(brain, rng, max_steps=500):
    """Classic CartPole physics. Returns how many steps the pole stayed up."""
    x, xd, th, thd = rng.uniform(-0.05, 0.05, 4)
    g, mc, mp, l, F, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    for t in range(max_steps):
        f = F if brain.act(np.array([x, xd, th, thd])) == 1 else -F
        s, c = math.sin(th), math.cos(th)
        tmp = (f + mp * l * thd * thd * s) / (mc + mp)
        thacc = (g * s - c * tmp) / (l * (4 / 3 - mp * c * c / (mc + mp)))
        xacc = tmp - mp * l * thacc * c / (mc + mp)
        x, xd, th, thd = x + dt * xd, xd + dt * xacc, th + dt * thd, thd + dt * thacc
        if abs(x) > 2.4 or abs(th) > 0.2095:
            return t + 1
    return max_steps


def fitness(theta, rng, episodes=3):
    W, _ = develop(theta)                         # grow once ...
    return np.mean([cartpole(Brain(W), rng) for _ in range(episodes)])   # ... then just live


# ---------------------------------------------------------------------------
# EVOLUTION: CMA-ES searches over the RULES, never over weights
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, n, sigma, lam, seed=0):
        self.n, self.lam, self.mu = n, lam, lam // 2
        self.rng = np.random.default_rng(seed)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum()
        me = self.me = 1 / np.sum(self.w ** 2)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m, self.sigma, self.C = np.zeros(n), sigma, np.eye(n)
        self.pc, self.ps, self.g = np.zeros(n), np.zeros(n), 0
        self._eig()

    def _eig(self):
        ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-20))

    def ask(self):
        self.Y = (self.rng.standard_normal((self.lam, self.n)) * self.D) @ self.B.T
        return self.m + self.sigma * self.Y

    def tell(self, fit):
        o = np.argsort(-np.asarray(fit)); Ys = self.Y[o[:self.mu]]; yw = self.w @ Ys
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * (self.B @ ((self.B.T @ yw) / self.D))
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Ys * self.w[:, None]).T @ Ys)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.g += 1; self._eig()


# ---------------------------------------------------------------------------
# Drawing: watch the network grow
# ---------------------------------------------------------------------------
def layout(kind):
    pos, hid = {}, [i for i, k in enumerate(kind) if k == "hid"]
    for i in range(NI):
        pos[i] = (0, 8.5 - i * 2.3)
    for k in range(NO):
        pos[NI + k] = (10, 6.0 - k * 3.0)
    for r, i in enumerate(hid):                     # hidden cells on a circle, in birth order
        a = 2 * np.pi * r / max(len(hid), 1) + np.pi / 2
        rad = 0 if len(hid) == 1 else 2.6
        pos[i] = (5 + rad * np.cos(a), 5 + rad * np.sin(a))
    return pos


def draw_stage(ax, H, A, kind, title, weighted, W=None):
    pos = layout(kind)
    for i, j in zip(*np.nonzero(A)):
        (x0, y0), (x1, y1) = pos[i], pos[j]
        if weighted:
            w = W[i, j]
            ax.plot([x0, x1], [y0, y1], color="#1baf7a" if w > 0 else "#eb6834",
                    lw=0.4 + 1.2 * abs(w), alpha=0.8, zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color="#9a9893", lw=0.8, alpha=0.7, zorder=1)
    for i, k in enumerate(kind):
        if k == "hid":          # colour = the cell's state (first 3 numbers): similar colour = similar cell type
            col = np.clip(0.5 + 0.5 * H[i, :3], 0, 1)
        else:
            col = "#dcdad4" if k == "in" else "#bfbcb4"
        ax.scatter(*pos[i], s=150, color=col, edgecolors="#333", linewidths=0.8, zorder=3)
    ax.set_title(f"{title}\n{kind.count('hid')} hidden, {int((A > 0).sum())} connections", fontsize=9)
    ax.set_xlim(-1, 11); ax.set_ylim(-0.5, 9.5); ax.axis("off")


def show(theta, hist, save=None):
    W, stages = develop(theta)
    fig = plt.figure(figsize=(3 * len(stages), 7))
    gs = fig.add_gridspec(2, len(stages), height_ratios=[1.2, 1], hspace=0.35)
    for s, (H, A, kind, title) in enumerate(stages):
        draw_stage(fig.add_subplot(gs[0, s]), H, A, kind, title, s == len(stages) - 1, W)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(hist["best"], color="#1baf7a", lw=2, label="best in generation")
    ax.plot(hist["mean"], color="#888780", lw=2, label="population average")
    ax.axhline(500, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("generation"); ax.set_ylabel("steps pole stays up (max 500)")
    ax.set_title("Evolution only changes the growth RULES; each network is grown, frozen, then tested", loc="left")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NDP: a genome of rules grows a network from one seed cell (sensors left, motors right; "
                 "green = + weight, orange = − weight)", fontsize=11)
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight"); print("Saved", save)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if a.save:
        plt.switch_backend("Agg")

    print(f"Genome = {N_PARAMS} numbers of RULES (not weights).")
    es, rng = CMAES(N_PARAMS, 0.5, a.pop, seed=a.seed), np.random.default_rng(a.seed)
    hist, best_theta, best_f = {"best": [], "mean": []}, None, -1
    for gen in range(a.gens):
        X = es.ask()
        f = np.array([fitness(x, rng) for x in X])
        es.tell(f)
        hist["best"].append(f.max()); hist["mean"].append(f.mean())
        if f.max() > best_f:
            best_f, best_theta = f.max(), X[f.argmax()].copy()
        print(f"gen {gen:3d}   best {f.max():6.1f}   mean {f.mean():6.1f}")
        if f.mean() >= 495:
            print("Population solved the task."); break

    W, stages = develop(best_theta)
    check = np.mean([cartpole(Brain(W), np.random.default_rng(1000 + i)) for i in range(20)])
    print(f"Best grown network: {stages[-1][2].count('hid')} hidden cells, "
          f"{int((W != 0).sum())} connections, {N_PARAMS} genome numbers; "
          f"balances {check:.0f}/500 steps on 20 new episodes.")
    show(best_theta, hist, a.save)


if __name__ == "__main__":
    main()