"""
Paper 2: Growing neural networks for control (NDP) and the role of neuronal diversity.
GPU version (PyTorch). Everything that runs every generation lives on the GPU: the
growth of every network, the tests, AND the CMA-ES optimizers. Two windows:

    Main window         evolution progress, reaching paths, growth replay, diversity in growth
    Comparison window   all methods side by side: error per generation (population and best),
                        final distance to target, diversity across evolution, final scores,
                        and a summary table

THE TASK (a mini "Reacher", like the paper's Brax Reacher)
    A point must move to a random target on a circle. Inputs: dx, dy (target minus
    position) and vx, vy (velocity). Outputs: force x, force y.
    Score = how close it stays to the target over 40 steps (1 = perfect).
    Error = 1 - score (0 = perfect).  Final distance = distance to the target at the end.

THE GROWTH PROGRAM (NDP, as in the paper, simplified)
    Start: 6 seed cells (4 input + 2 output), connected to each other, each with a random
    4-number "extrinsic" state. For GROW steps every cell:
      differentiate  new state from its own state and its neighbours' average (DiffModel)
      divide         if its state says so, make a child connected to it (GenModel)
    Then every connection gets a weight from the states of its two cells (EdgeModel).
    Diversity fixes (by method):
      intrinsic state      each seed cell has a fixed one-hot ID, copied to its children
      lateral inhibition   a cell that acts blocks its neighbours for 2 steps (in parallel:
                           a cell may act only if it beats all its neighbours in a random draw)

THE FIVE METHODS (evolved side by side with CMA-ES, several independent runs each)
    Direct network               all NMAX x NMAX weights evolved directly (paper's reference)
    NDP, no fixes / + inhibition / + intrinsic state / + intrinsic + inhibition
    The paper treats the direct network as the expected upper bound on these tasks; its
    claim is that growth becomes TRAINABLE with the fixes. Try --nmax 32 to see the gap shrink
    (the growth program stays the same size, the direct network grows as NMAX^2).

PERFORMANCE NOTES
    * CMA-ES runs batched on the GPU (one optimizer per method x run).
    * Growth has no Python loops over cells and no GPU->CPU waits; one small transfer
      per generation brings the summary numbers back for plotting.
    * Each frame evolves for a fixed time budget, then redraws; the comparison window
      redraws about once a second. CPU threads are limited so the CPU does not spin.
    * Bigger --pop / --episodes / --nmax = more work per GPU launch = higher GPU usage.

Controls (main window): Evolve on/off, +50 generations, Reset, method to watch,
          Replay growth, program shown (evolved / random), Comparison window.
Keys: space = evolve, f = +50 generations, g = replay growth, d = evolved/random, c = comparison.

Run:   python ndp_growth_gpu.py                        (uses CUDA if available)
       python ndp_growth_gpu.py --pop 1024 --episodes 8   (more GPU work, smoother curves)
       python ndp_growth_gpu.py --nmax 32
       python ndp_growth_gpu.py --save shot.png --gens 300     (headless: saves both windows)
Needs: torch (CUDA build for GPU), numpy, matplotlib (with a GUI backend such as TkAgg/QtAgg)
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")              # stop numpy/BLAS from spinning every CPU core

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Button, RadioButtons

# ---------------------------------------------------------------------------
# Settings (some are overwritten from the command line in main())
# ---------------------------------------------------------------------------
NSEED, DE = 6, 4
DH = DE + NSEED
NMAX = 14
GROW, INHIB = 10, 2
EP_LEN, K = 40, 2
POP, RUNS, N_EP = 256, 3, 4
MAX_GENS = 5000
DEV = torch.device("cpu")
FDT = torch.float32                           # dtype for networks; CMA-ES uses CMA_DT
CMA_DT = torch.float64
IN_NAMES = ["dx", "dy", "vx", "vy", "Fx", "Fy"]

METHODS = ["Direct network", "NDP, no fixes", "NDP + inhibition", "NDP + intrinsic state",
           "NDP + intrinsic + inhibition"]
SHORT = ["Direct", "NDP", "NDP+inhib", "NDP+intr", "NDP+intr+inhib"]
KEYS = ["direct", "plain", "inhib", "intr", "intr_inhib"]
FLAGS = {"plain": (False, False), "inhib": (False, True), "intr": (True, False), "intr_inhib": (True, True)}
NDP_KEYS = KEYS[1:]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"direct": "#888780", "plain": "#e34948", "inhib": "#eda100", "intr": "#1baf7a", "intr_inhib": "#2a78d6"}
C_POS, C_NEG = "#1baf7a", "#eb6834"
LINEAGE = ["#2a78d6", "#e87ba4", "#eda100", "#6250d6", "#1baf7a", "#eb6834"]


# ---------------------------------------------------------------------------
# Batched CMA-ES on the device: G independent optimizers of the same dimension
# ---------------------------------------------------------------------------
class BatchCMAES:
    def __init__(self, G, n, lam, sigma0, gen):
        self.G, self.n, self.lam, self.mu, self.gen = G, n, lam, lam // 2, gen
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); w /= w.sum()
        me = self.me = 1 / np.sum(w ** 2)
        self.w = torch.as_tensor(w, dtype=CMA_DT, device=DEV)
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n); self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.ds = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        z = lambda *s: torch.zeros(*s, dtype=CMA_DT, device=DEV)
        self.m = z(G, n); self.sigma = torch.full((G,), sigma0, dtype=CMA_DT, device=DEV)
        self.C = torch.eye(n, dtype=CMA_DT, device=DEV).repeat(G, 1, 1)
        self.pc = z(G, n); self.ps = z(G, n); self.g = 0
        self._eig()

    def _eig(self):
        ev, self.B = torch.linalg.eigh(self.C)
        self.D = ev.clamp(min=1e-20).sqrt()

    def ask(self):
        Z = torch.randn(self.G, self.lam, self.n, dtype=CMA_DT, device=DEV, generator=self.gen)
        self.Y = (Z * self.D[:, None, :]) @ self.B.transpose(1, 2)
        return (self.m[:, None, :] + self.sigma[:, None, None] * self.Y).to(FDT)       # [G, lam, n]

    def tell(self, fit):                                                              # fit [G, lam]
        order = torch.argsort(fit.to(CMA_DT), dim=1, descending=True)[:, :self.mu]
        Ys = self.Y.gather(1, order[:, :, None].expand(-1, -1, self.n))
        yw = torch.einsum('k,gkn->gn', self.w, Ys)
        self.m = self.m + self.sigma[:, None] * yw
        Bt_yw = (self.B.transpose(1, 2) @ yw[:, :, None])[:, :, 0] / self.D
        invsq = (self.B @ Bt_yw[:, :, None])[:, :, 0]
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.me) * invsq
        psn = self.ps.norm(dim=1)
        hs = (psn / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN).to(CMA_DT)
        self.pc = (1 - self.cc) * self.pc + hs[:, None] * np.sqrt(self.cc * (2 - self.cc) * self.me) * yw
        rank_mu = torch.einsum('gkn,k,gkm->gnm', Ys, self.w, Ys)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc[:, :, None] * self.pc[:, None, :]
                               + ((1 - hs) * self.cc * (2 - self.cc))[:, None, None] * self.C)
                  + self.cmu * rank_mu)
        self.C = (self.C + self.C.transpose(1, 2)) / 2
        self.sigma = (self.sigma * torch.exp((self.cs / self.ds) * (psn / self.chiN - 1))).clamp(max=10)
        self.g += 1
        self._eig()


# ---------------------------------------------------------------------------
# Parameters: every NDP method uses the same layout (cell state of size DH);
# methods without intrinsic state have the intrinsic part padded with zeros.
# ---------------------------------------------------------------------------
def ndp_dim(intr):
    dh = DE + (NSEED if intr else 0)
    return DE * DE + DE * dh + DE + dh + 1 + dh * dh + 1


def direct_dim():
    return NMAX * NMAX


def ndp_unpack(X, intr):
    X = torch.as_tensor(X, dtype=FDT, device=DEV)
    X = X.reshape(-1, X.shape[-1])
    dh = DE + (NSEED if intr else 0); P = len(X); i = 0

    def take(n, shape):
        nonlocal i
        v = X[:, i:i + n].reshape((P,) + shape); i += n; return v

    A = take(DE * DE, (DE, DE)); B = take(DE * dh, (DE, dh)); b = take(DE, (DE,))
    wg = take(dh, (dh,)); bg = take(1, ()); M = take(dh * dh, (dh, dh)); c = take(1, ())
    if not intr:
        z = lambda *s: torch.zeros(s, dtype=FDT, device=DEV)
        B = torch.cat([B, z(P, DE, NSEED)], 2); wg = torch.cat([wg, z(P, NSEED)], 1)
        M = torch.cat([torch.cat([M, z(P, DE, NSEED)], 2), z(P, NSEED, DH)], 1)
    return dict(A=A, B=B, b=b, wg=wg, bg=bg, M=M, c=c)


def cat_params(parts):
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


# ---------------------------------------------------------------------------
# Growth, fully batched, no GPU->CPU waits
# ---------------------------------------------------------------------------
def _local_winners(cand, adj, gen):
    pri = torch.rand(cand.shape, device=DEV, generator=gen)
    pri = torch.where(cand, pri, torch.full_like(pri, -1.0))
    nb = torch.where(adj > 0, pri[:, None, :], torch.full_like(adj, -1.0)).amax(-1)
    return cand & (pri > nb)


def grow(p, intr, inhib, gen, record=False):
    B = p["A"].shape[0]; N = NMAX
    ext = torch.zeros(B, N, DE, device=DEV)
    ext[:, :NSEED] = torch.rand(B, NSEED, DE, device=DEV, generator=gen) * 2 - 1
    it = torch.zeros(B, N, NSEED, device=DEV)
    it[:, torch.arange(NSEED), torch.arange(NSEED)] = 1
    alive = torch.zeros(B, N, dtype=torch.bool, device=DEV); alive[:, :NSEED] = True
    parent = torch.full((B, N), -1, dtype=torch.long, device=DEV)
    adj = torch.zeros(B, N, N, device=DEV); adj[:, :NSEED, :NSEED] = 1
    adj[:, torch.arange(N), torch.arange(N)] = 0
    blockD = torch.zeros(B, N, device=DEV); blockG = torch.zeros(B, N, device=DEV)
    intr_f = intr.float()[:, None, None]; inh = inhib[:, None]
    ar = torch.arange(N, device=DEV)

    def state():
        return torch.cat([ext, it * intr_f], -1)

    def weights():
        h = state()
        Wt = 2 * torch.tanh(torch.einsum('bid,bde,bje->bij', h, p["M"], h) + p["c"][:, None, None])
        return Wt * adj * alive[:, :, None] * alive[:, None, :]

    def snap(acted):
        return dict(ext=ext.clone(), alive=alive.clone(), adj=adj.clone(), it=it.clone(), W=weights(),
                    parent=parent.clone(), acted=acted.clone())

    frames = [snap(torch.zeros(B, N, dtype=torch.bool, device=DEV))] if record else []
    for _ in range(GROW):
        # differentiate
        h = state()
        m = adj @ h / adj.sum(2, keepdim=True).clamp(min=1)
        new = 0.5 * ext + 0.5 * torch.tanh(torch.einsum('bde,bne->bnd', p["A"], ext)
                                           + torch.einsum('bde,bne->bnd', p["B"], m) + p["b"][:, None, :])
        elig = alive & (blockD == 0)
        act = torch.where(inh, _local_winners(elig, adj, gen), elig)
        hit = (adj * act[:, None, :].float()).sum(-1) > 0
        blockD = torch.where(inh & hit, blockD.clamp(min=INHIB + 1), blockD)
        ext = torch.where(act[..., None], new, ext)
        # divide: dense "who is whose child" matrix S[b, child, parent] (no nonzero(), no sync)
        gate = torch.einsum('bd,bnd->bn', p["wg"], state()) + p["bg"][:, None]
        want = alive & (gate > 0) & (blockG == 0)
        want = torch.where(inh, _local_winners(want, adj, gen), want)
        key = torch.where(want, torch.rand(B, N, device=DEV, generator=gen), torch.full((B, N), 2.0, device=DEV))
        rank = torch.argsort(torch.argsort(key, 1), 1)
        do = want & (rank < (~alive).sum(1, keepdim=True))
        free_sorted = torch.argsort(alive.long() * N + ar, 1)
        child = free_sorted.gather(1, rank.clamp(max=N - 1))
        S = ((child[:, None, :] == ar[None, :, None]) & do[:, None, :]).float()
        born = S.sum(-1) > 0
        noise = 0.05 * torch.randn(B, N, DE, device=DEV, generator=gen)
        ext = torch.where(born[..., None], S @ ext + noise, ext)
        it = torch.where(born[..., None], S @ it, it)
        parent = torch.where(born, (S * ar[None, None, :].float()).sum(-1).long(), parent)
        alive = alive | born
        adj = (adj + S + S.transpose(1, 2)).clamp(max=1)
        hit2 = (adj * do[:, None, :].float()).sum(-1) > 0
        blockG = torch.where(inh & hit2, blockG.clamp(min=INHIB + 1), blockG)
        blockD = (blockD - 1).clamp(min=0); blockG = (blockG - 1).clamp(min=0)
        if record:
            frames.append(snap(act))
    return weights(), ext, alive, frames


def diversity(ext, alive):
    d = torch.cdist(ext, ext)
    pair = (alive[:, :, None] & alive[:, None, :]).float()
    pair = pair * (1 - torch.eye(ext.shape[1], device=ext.device))[None]
    return (d * pair).sum((1, 2)) / pair.sum((1, 2)).clamp(min=1)


# ---------------------------------------------------------------------------
# Task, batched over networks AND episodes. Returns score and final distance per network.
# ---------------------------------------------------------------------------
def reacher(W, gen, n_ep):
    Bn, N, _ = W.shape
    Wr = W.repeat_interleave(n_ep, 0); R = Wr.shape[0]
    ang = torch.rand(R, device=DEV, generator=gen) * 2 * np.pi
    tgt = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    pos = torch.zeros(R, 2, device=DEV); vel = torch.zeros(R, 2, device=DEV)
    v = torch.zeros(R, N, device=DEV); score = torch.zeros(R, device=DEV)
    for _ in range(EP_LEN):
        obs = torch.cat([tgt - pos, vel], 1)
        for _ in range(K):
            v = torch.tanh(torch.bmm(torch.cat([obs, v[:, 4:]], 1)[:, None, :], Wr)[:, 0])
        vel = 0.8 * vel + 0.15 * v[:, 4:6]
        pos = pos + 0.15 * vel
        score += (1 - (tgt - pos).norm(dim=1)).clamp(0, 1)
    fit = (score / EP_LEN).view(Bn, n_ep).mean(1)
    dist = (tgt - pos).norm(dim=1).view(Bn, n_ep).mean(1)
    return fit, dist


# ---------------------------------------------------------------------------
# One generation for every method x run, in one batch
# ---------------------------------------------------------------------------
def evaluate_all(asked, gen):
    """asked[key] = tensor [RUNS, POP, dim]. Returns fitness [5, RUNS, POP] and summary [5, RUNS, 4]."""
    parts, intr, inhib = [], [], []
    for k in NDP_KEYS:
        fi, fh = FLAGS[k]
        parts.append(ndp_unpack(asked[k], fi))
        intr.append(torch.full((RUNS * POP,), fi, device=DEV)); inhib.append(torch.full((RUNS * POP,), fh, device=DEV))
    W_ndp, ext, alive, _ = grow(cat_params(parts), torch.cat(intr), torch.cat(inhib), gen)
    div = diversity(ext, alive).view(4, RUNS, POP)
    W_dir = asked["direct"].reshape(RUNS * POP, NMAX, NMAX)
    fit, dist = reacher(torch.cat([W_dir, W_ndp], 0), gen, N_EP)
    fit, dist = fit.view(5, RUNS, POP), dist.view(5, RUNS, POP)
    div_all = torch.cat([torch.full((1, RUNS, POP), float("nan"), device=DEV), div], 0)
    summary = torch.stack([fit.mean(2), fit.amax(2), dist.mean(2), div_all.mean(2)], -1)
    return fit, summary


def build_one(th, key, gen, record=False, copies=1):
    th = torch.as_tensor(np.asarray(th), dtype=FDT, device=DEV).reshape(1, -1).repeat(copies, 1)
    if key == "direct":
        return th.view(-1, NMAX, NMAX), None, None, []
    fi, fh = FLAGS[key]; B = len(th)
    return grow(ndp_unpack(th, fi), torch.full((B,), fi, device=DEV), torch.full((B,), fh, device=DEV), gen, record)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class App:
    def __init__(self, budget):
        plt.rcParams.update({
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
            "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
            "font.size": 10, "axes.titlesize": 11,
        })
        self.budget = budget
        self.fig = plt.figure(figsize=(15, 10.5))
        self._title(self.fig, "Paper 2: growing networks (NDP), GPU")
        gs = self.fig.add_gridspec(2, 2, left=0.05, right=0.985, top=0.875, bottom=0.245,
                                   hspace=0.42, wspace=0.16, height_ratios=[1, 1.15])
        self.axE = self.fig.add_subplot(gs[0, 0]); self.axR = self.fig.add_subplot(gs[0, 1])
        self.axG = self.fig.add_subplot(gs[1, 0]); self.axD = self.fig.add_subplot(gs[1, 1])
        self.title = self.fig.text(0.05, 0.955, "", fontsize=13, va="center")
        self.sub = self.fig.text(0.05, 0.925, "", fontsize=9.5, color=INK2, va="center")
        self.watch = "intr"; self.dna_random = False
        self.cfig = None
        self._controls()
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)
        self.evolving = False
        self.last_cmp = 0.0
        self.reset()
        self.open_comparison()

    @staticmethod
    def _title(fig, t):
        try:
            fig.canvas.manager.set_window_title(t)
        except AttributeError:
            pass

    # ---------------- evolution ----------------
    def reset(self):
        if self.evolving:
            self.toggle_evolve()
        self.dims = {"direct": direct_dim(), **{k: ndp_dim(FLAGS[k][0]) for k in FLAGS}}
        self.g_cma = torch.Generator(device=DEV); self.g_cma.manual_seed(1)
        self.g_env = torch.Generator(device=DEV); self.g_env.manual_seed(0)
        self.es = {k: BatchCMAES(RUNS, self.dims[k], POP, 0.5, self.g_cma) for k in KEYS}
        self.hist = []            # per generation: numpy [5, RUNS, 4] (mean score, best score, mean distance, diversity)
        self.gen = 0; self.sec_per_gen = float("nan")
        self.replay()

    def evolve_one(self):
        if self.gen >= MAX_GENS:
            return False
        asked = {k: self.es[k].ask() for k in KEYS}
        fit, summary = evaluate_all(asked, self.g_env)
        for i, k in enumerate(KEYS):
            self.es[k].tell(fit[i])
        self.hist.append(summary.cpu().numpy())        # the only GPU->CPU transfer per generation
        self.gen += 1
        return True

    def evolve_for(self, seconds=None, n=None):
        t0 = time.perf_counter(); done = 0
        while True:
            if not self.evolve_one():
                break
            done += 1
            if n is not None and done >= n:
                break
            if n is None and time.perf_counter() - t0 >= seconds:
                break
        if done:
            dt = (time.perf_counter() - t0) / done
            self.sec_per_gen = dt if np.isnan(self.sec_per_gen) else 0.7 * self.sec_per_gen + 0.3 * dt
        return done

    def fast_forward(self, n=50):
        self.evolve_for(n=n)
        self.replay()
        self.draw_comparison(force=True)

    def H(self):
        return np.stack(self.hist) if self.hist else np.zeros((0, 5, RUNS, 4))

    # ---------------- replay of the watched method (run 1) ----------------
    def replay(self):
        g = torch.Generator(device=DEV); g.manual_seed(int(np.random.default_rng().integers(1 << 30)))
        th = self.es[self.watch].m[0].cpu().numpy()
        if self.dna_random and self.watch != "direct":
            th = np.random.default_rng().normal(0, 0.5, self.dims[self.watch])
        _, _, _, frames = build_one(th, self.watch, g, record=True)
        self.frames = [{k: v[0].cpu().numpy() for k, v in f.items()} for f in frames]
        self.frame_i = 0
        W8 = build_one(th, self.watch, g, copies=8)[0]
        self.paths_score, self.path, self.tgt = self._paths(W8, np.linspace(0, 2 * np.pi, 8, endpoint=False))
        self._layout()
        self.draw()

    def _paths(self, W8, angles):
        R = W8.shape[0]
        tgt = torch.as_tensor(np.stack([np.cos(angles), np.sin(angles)], 1), dtype=FDT, device=DEV)
        pos = torch.zeros(R, 2, device=DEV); vel = torch.zeros(R, 2, device=DEV)
        v = torch.zeros(R, NMAX, device=DEV); score = torch.zeros(R, device=DEV); path = [pos.clone()]
        for _ in range(EP_LEN):
            obs = torch.cat([tgt - pos, vel], 1)
            for _ in range(K):
                v = torch.tanh(torch.bmm(torch.cat([obs, v[:, 4:]], 1)[:, None, :], W8)[:, 0])
            vel = 0.8 * vel + 0.15 * v[:, 4:6]; pos = pos + 0.15 * vel
            score += (1 - (tgt - pos).norm(dim=1)).clamp(0, 1); path.append(pos.clone())
        return (score / EP_LEN).mean().item(), torch.stack(path).cpu().numpy(), tgt.cpu().numpy()

    def _layout(self):
        last = self.frames[-1]; alive = last["alive"]; parent = last["parent"]
        pos = np.zeros((NMAX, 2))
        for i in range(4):
            pos[i] = (0.8, 8.2 - i * 2.1)
        pos[4], pos[5] = (9.2, 6.6), (9.2, 3.0)
        r = np.random.default_rng(3)
        for n in range(NSEED, NMAX):
            if alive[n]:
                pos[n] = np.clip(pos[parent[n]] * 0.35 + np.array([5.0, 5.0]) * 0.65 + r.normal(0, 1.1, 2),
                                 [2.2, 0.6], [7.8, 9.4])
        for _ in range(150):
            for a in range(NSEED, NMAX):
                if alive[a]:
                    d = pos[a] - pos[alive]; dist = np.linalg.norm(d, axis=1) + 1e-6
                    push = (d / dist[:, None] * (np.maximum(0, 1.3 - dist))[:, None]).sum(0)
                    pos[a] = np.clip(pos[a] + 0.1 * push, [2.2, 0.6], [7.8, 9.4])
        self.pos = pos

    # ---------------- controls ----------------
    def _controls(self):
        f = self.fig
        self.b_evo = Button(f.add_axes([0.05, 0.085, 0.10, 0.05]), "Evolve: off")
        self.b_ff = Button(f.add_axes([0.16, 0.085, 0.10, 0.05]), "+50 generations")
        self.b_rst = Button(f.add_axes([0.27, 0.085, 0.07, 0.05]), "Reset")
        self.b_rep = Button(f.add_axes([0.05, 0.025, 0.10, 0.05]), "Replay growth")
        self.b_dna = Button(f.add_axes([0.16, 0.025, 0.18, 0.05]), "Program shown: evolved")
        self.b_cmp = Button(f.add_axes([0.875, 0.085, 0.11, 0.05]), "Comparison window")
        self.b_evo.on_clicked(lambda e: self.toggle_evolve())
        self.b_ff.on_clicked(lambda e: self.fast_forward())
        self.b_rst.on_clicked(lambda e: (self.reset(), self.draw_comparison(force=True)))
        self.b_rep.on_clicked(lambda e: self.replay())
        self.b_dna.on_clicked(lambda e: self.toggle_dna())
        self.b_cmp.on_clicked(lambda e: self.open_comparison())
        axr = f.add_axes([0.38, 0.02, 0.22, 0.15])
        axr.set_title("method to watch", fontsize=9.5, loc="left", color=INK2)
        self.r_m = RadioButtons(axr, METHODS, active=3, activecolor=INK2)
        self.r_m.on_clicked(self._set_watch)
        for s in axr.spines.values():
            s.set_visible(False)
        f.text(0.62, 0.175, "fill color = cell's current state (similar color = similar state)\n"
               "ring color = cell family (which seed cell it descends from)\n"
               "edges: aqua = positive weight, orange = negative; width = size\n"
               "evolution: line = mean of runs, band = best to worst run",
               fontsize=8.5, color=INK2, va="top", linespacing=1.6)
        f.text(0.62, 0.04, "Keys: space = evolve, f = +50 gens, g = replay, d = evolved/random, c = comparison",
               fontsize=8.5, color=INK2)
        f.canvas.mpl_connect("key_press_event", self._key)

    def _key(self, e):
        {" ": self.toggle_evolve, "f": self.fast_forward, "g": self.replay, "d": self.toggle_dna,
         "c": self.open_comparison}.get(e.key, lambda: None)()

    def toggle_dna(self):
        self.dna_random = not self.dna_random
        self.b_dna.label.set_text("Program shown: " + ("random" if self.dna_random else "evolved"))
        self.replay()

    def _set_watch(self, label):
        self.watch = KEYS[METHODS.index(label)]; self.replay()

    def toggle_evolve(self):
        self.evolving = not self.evolving
        self.b_evo.label.set_text("Evolve: on" if self.evolving else "Evolve: off")
        self.fig.canvas.draw_idle()

    def start(self):
        self.timer.start()

    def _tick(self):
        if self.evolving:
            before = self.gen
            self.evolve_for(seconds=self.budget)
            if self.gen // 50 != before // 50:
                self.replay()
            else:
                self.draw()
            self.draw_comparison()
            return
        if self.frames and self.frame_i < len(self.frames) - 1:
            self.frame_i += 1
            self.draw()

    # ---------------- drawing helpers ----------------
    @staticmethod
    def _clean(ax, grid=True):
        ax.cla()
        if grid:
            ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    @staticmethod
    def _smooth(y, w):
        if len(y) < w:
            return np.arange(len(y)), y
        k = np.ones(w) / w
        s = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, y)
        return np.arange(len(s)) + len(y) - len(s), s

    def _curve(self, ax, y, k, label=None, ls="-", w=9):
        """y: [gens, RUNS]. Mean line plus best-to-worst-run band."""
        x, s = self._smooth(y, w)
        if len(s) == 0:
            return
        ax.plot(x, s.mean(1), color=COL[k], lw=2.2, ls=ls, label=label)
        if ls == "-":
            ax.fill_between(x, s.min(1), s.max(1), color=COL[k], alpha=0.15, lw=0)

    # ---------------- main window ----------------
    def draw(self):
        H = self.H()
        evals = 5 * RUNS * POP * N_EP
        spg = "" if np.isnan(self.sec_per_gen) else f"   {self.sec_per_gen * 1000:.0f} ms/generation"
        self.title.set_text(f"Paper 2: growing a controller from a few cells (NDP)    generation {self.gen}"
                            f"    device: {DEV.type}{spg}")
        self.sub.set_text(f"Task: steer a point to a random target with two forces. Up to {NMAX} cells. "
                          f"Each generation: 5 methods x {RUNS} runs x {POP} individuals x {N_EP} episodes "
                          f"= {evals:,} episodes in one batch.")
        ax = self.axE; self._clean(ax)
        for i, (name, k) in enumerate(zip(METHODS, KEYS)):
            if len(H):
                self._curve(ax, H[:, i, :, 0], k, f"{name} ({self.dims[k]} numbers)")
        ax.set_xlim(0, max(60, self.gen)); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title("Evolution: reaching score (population average, 1 = perfect)", loc="left")
        ax.set_xlabel("generation")
        if self.gen:
            ax.legend(loc="upper left", frameon=False, fontsize=8.6, ncol=2, columnspacing=1.2)
        else:
            ax.text(0.5, 0.5, "press  Evolve  or  +50 generations", transform=ax.transAxes, ha="center", color=INK2)
        ax = self.axR; ax.cla(); ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color=INK2, lw=0.8))
        for i in range(self.path.shape[1]):
            ax.plot(self.path[:, i, 0], self.path[:, i, 1], color=COL[self.watch], lw=1.6, alpha=0.9)
            ax.plot(*self.tgt[i], marker="x", color=INK, ms=7, mew=1.6)
        ax.plot(0, 0, "o", color=INK2, ms=5)
        name = METHODS[KEYS.index(self.watch)]
        which = "random program" if (self.dna_random and self.watch != "direct") else "current solution (run 1)"
        ax.set_title(f"{name}, {which}:\npaths to 8 targets (×), score {self.paths_score:.2f}", loc="left")
        ax = self.axG; ax.cla(); ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.4, 10.2); ax.axis("off")
        if self.watch == "direct":
            ax.text(5, 5, f"The direct network does not grow:\nall {NMAX}×{NMAX} = {NMAX * NMAX} weights are evolved directly.",
                    ha="center", va="center", color=INK2, fontsize=11)
            ax.set_title("Growth", loc="left")
        else:
            fr = self.frames[self.frame_i]
            alive, adj, W, ext, it = fr["alive"], fr["adj"], fr["W"], fr["ext"], fr["it"]
            proj = np.array([[0.8, -0.3, 0.5, 0.1], [0.2, 0.7, -0.4, 0.6]])
            idx = np.where(alive)[0]
            for i in idx:
                for j in idx:
                    if j > i and adj[i, j] > 0:
                        w = (W[i, j] + W[j, i]) / 2
                        ax.plot(*zip(self.pos[i], self.pos[j]), color=C_POS if w >= 0 else C_NEG,
                                lw=0.3 + 1.6 * abs(w), alpha=0.75, zorder=1, solid_capstyle="round")
            intr = FLAGS[self.watch][0]
            for i in idx:
                pp = proj @ ext[i]
                fill = hsv_to_rgb([(np.arctan2(pp[1], pp[0]) / (2 * np.pi)) % 1, 0.55, 0.92])
                ring = LINEAGE[int(it[i].argmax())] if intr else INK2
                ax.scatter(*self.pos[i], s=420 if i < NSEED else 260, color=fill, edgecolors=ring,
                           linewidths=3 if intr else 1, zorder=3)
                if i < NSEED:
                    ax.text(*(self.pos[i] + (np.array([-0.75, 0]) if i < 4 else np.array([0.75, 0]))),
                            IN_NAMES[i], ha="right" if i < 4 else "left", va="center", fontsize=9.5)
            ax.set_title(f"{name}: growth step {self.frame_i}/{GROW}, {len(idx)} cells", loc="left")
        ax = self.axD; self._clean(ax)
        if self.watch != "direct":
            intr = FLAGS[self.watch][0]
            d = [float(diversity(torch.as_tensor(f["ext"])[None], torch.as_tensor(f["alive"])[None])[0]) for f in self.frames]
            n = self.frame_i + 1
            ax.plot(range(len(d)), d, color=COL[self.watch], lw=1, alpha=0.25)
            ax.plot(range(n), d[:n], color=COL[self.watch], lw=2.4, marker="o", ms=5, label="changing state only")
            if intr:
                di = [float(diversity(torch.as_tensor(np.concatenate([f["ext"], f["it"]], -1))[None],
                                      torch.as_tensor(f["alive"])[None])[0]) for f in self.frames]
                ax.plot(range(n), di[:n], color=COL[self.watch], lw=2, ls="--", marker="s", ms=4,
                        label="changing state + fixed family ID (what sets the weights)")
                d = d + di
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            ax.set_xlim(0, GROW); ax.set_ylim(0, max(2.4, max(d) * 1.25))
        else:
            ax.text(0.5, 0.5, "no growth for the direct network", transform=ax.transAxes, ha="center", color=INK2)
        ax.axhspan(0, 0.15, color="#fcebeb", lw=0, zorder=0)
        ax.text(0.1, 0.05, "clones (all cells nearly identical)", fontsize=8.5, color="#a32d2d", va="bottom")
        ax.set_title("Neuronal diversity during this growth (average distance between cells)", loc="left")
        ax.set_xlabel("growth step")
        self.fig.canvas.draw_idle()

    # ---------------- comparison window ----------------
    def open_comparison(self):
        if self.cfig is not None and plt.fignum_exists(self.cfig.number):
            self.draw_comparison(force=True)
            try:
                self.cfig.canvas.manager.show()
            except Exception:
                pass
            return
        self.cfig = plt.figure(figsize=(15, 9.5))
        self._title(self.cfig, "Paper 2: comparison of all methods")
        g = self.cfig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.88, bottom=0.08, hspace=0.42, wspace=0.26)
        self.cax = [[self.cfig.add_subplot(g[i, j]) for j in range(3)] for i in range(2)]
        self.ctitle = self.cfig.text(0.055, 0.955, "", fontsize=13, va="center")
        self.cfig.text(0.055, 0.925, "Error = 1 − reaching score (0 = perfect).  Lines = mean of the independent runs, "
                       "bands = best to worst run.  Log scale on the error plots.", fontsize=9.5, color=INK2, va="center")
        self.draw_comparison(force=True)
        try:
            self.cfig.show()
        except Exception:
            pass

    def draw_comparison(self, force=False):
        if self.cfig is None or not plt.fignum_exists(self.cfig.number):
            return
        now = time.perf_counter()
        if not force and now - self.last_cmp < 1.0:
            return
        self.last_cmp = now
        H = self.H()
        self.ctitle.set_text(f"Comparison of all methods    generation {self.gen}    "
                             f"({RUNS} runs x {POP} individuals x {N_EP} episodes per method)")
        (a1, a2, a3), (a4, a5, a6) = self.cax
        for a in (a1, a2, a3, a4):
            self._clean(a)
        if len(H):
            for i, k in enumerate(KEYS):
                self._curve(a1, np.clip(1 - H[:, i, :, 0], 1e-3, 1), k, SHORT[i])
                self._curve(a2, np.clip(1 - H[:, i, :, 1], 1e-3, 1), k, SHORT[i])
                self._curve(a3, H[:, i, :, 2], k, SHORT[i])
                if k != "direct":
                    self._curve(a4, H[:, i, :, 3], k, SHORT[i])
        for a, t, yl in [(a1, "Error per generation (population average)", "error = 1 − score"),
                         (a2, "Error of the best individual per generation", "error = 1 − score"),
                         (a3, "Distance to target at the end of an episode", "distance (population average)"),
                         (a4, "Neuronal diversity of grown networks across evolution", "avg distance between cells")]:
            a.set_title(t, loc="left", fontsize=10.5); a.set_ylabel(yl); a.set_xlabel("generation")
            a.set_xlim(0, max(60, self.gen))
        for a in (a1, a2):
            a.set_yscale("log"); a.set_ylim(0.03, 1.05)
        a3.set_ylim(0, None)
        a4.axhspan(0, 0.15, color="#fcebeb", lw=0, zorder=0)
        a4.text(0.01, 0.03, "clones", transform=a4.transAxes, fontsize=8.5, color="#a32d2d")
        if len(H):
            a1.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
        # final comparison bars (last 10 generations)
        self._clean(a5); a5.grid(False, axis="x")
        if len(H):
            last = H[-10:].mean(0)                          # [5, RUNS, 4]
            err = 1 - last[:, :, 0]
            x = np.arange(5)
            a5.bar(x, err.mean(1), color=[COL[k] for k in KEYS], width=0.62)
            for i in range(5):
                a5.scatter(np.full(RUNS, x[i]), err[i], color=INK, s=14, zorder=3)
                a5.text(x[i], err[i].max() + 0.02, f"{err[i].mean():.2f}", ha="center", va="bottom", fontsize=9)
            a5.set_xticks(x); a5.set_xticklabels([f"{s}\n({self.dims[k]})" for s, k in zip(SHORT, KEYS)], fontsize=8.5)
            a5.set_ylim(0, 1.05)
        a5.set_title("Final error (last 10 gens; dots = runs; (n) = evolved values)", loc="left", fontsize=10.5)
        a5.set_ylabel("error = 1 − score")
        # summary table
        a6.cla(); a6.axis("off")
        a6.set_title("Summary (mean over runs, last 10 generations)", loc="left", fontsize=10.5)
        if len(H):
            last = H[-10:].mean(0)
            rows = []
            for i, k in enumerate(KEYS):
                y = H[:, i, :, 0].mean(1)
                final = last[i, :, 0].mean()
                target = 0.9 * final
                reach = int(np.argmax(y >= target)) if np.any(y >= target) and final > 0.2 else None
                div = last[i, :, 3].mean()
                rows.append([SHORT[i], f"{self.dims[k]}", f"{final:.2f}", f"{last[i, :, 1].mean():.2f}",
                             f"{last[i, :, 2].mean():.2f}", "–" if k == "direct" else f"{div:.2f}",
                             "–" if reach is None else str(reach)])
            tb = a6.table(cellText=rows, colLabels=["method", "evolved\nvalues", "score\n(avg)", "score\n(best)",
                                                     "final\ndistance", "diversity", "gens to 90%\nof final"],
                          loc="upper center", cellLoc="center", colLoc="center",
                          colWidths=[0.24, 0.11, 0.1, 0.1, 0.12, 0.13, 0.17])
            tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 2.1)
            for (r, c), cell in tb.get_celld().items():
                cell.set_edgecolor(GRID)
                if r == 0:
                    cell.set_text_props(color=INK2)
                elif c == 0:
                    cell.set_text_props(color=COL[KEYS[r - 1]], fontweight="bold")
            a6.text(0.0, 0.08, "gens to 90% of final: first generation whose population average\n"
                    "reached 90% of that method's final score (– = never really learned).",
                    transform=a6.transAxes, fontsize=8.5, color=INK2, va="bottom")
        self.cfig.canvas.draw_idle()


def main():
    global DEV, NMAX, POP, RUNS, N_EP, GROW, CMA_DT
    ap = argparse.ArgumentParser(description="NDP growth simulation, batched on the GPU")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--pop", type=int, default=None, help="population per method and run")
    ap.add_argument("--runs", type=int, default=3, help="independent evolution runs per method")
    ap.add_argument("--episodes", type=int, default=4, help="episodes per evaluation")
    ap.add_argument("--nmax", type=int, default=14, help="most cells a network can grow to")
    ap.add_argument("--grow", type=int, default=10, help="growth steps")
    ap.add_argument("--budget", type=float, default=0.4, help="seconds of evolution per screen update")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    ap.add_argument("--save", default=None); ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--watch", default="intr", choices=KEYS)
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    DEV = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       (a.device if a.device != "auto" else "cpu"))
    POP = a.pop or (256 if DEV.type == "cuda" else 64)
    RUNS, N_EP, NMAX, GROW = a.runs, a.episodes, a.nmax, a.grow
    if DEV.type == "cuda":
        CMA_DT = torch.float32                    # consumer GPUs are slow at float64
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_num_threads(a.threads or 2)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if a.threads:
            torch.set_num_threads(a.threads)
        print("Using CPU (install a CUDA build of PyTorch to use the GPU)")
    if a.save:
        plt.switch_backend("Agg")
    app = App(a.budget)
    if a.save:
        t0 = time.perf_counter()
        app.evolve_for(n=a.gens)
        el = time.perf_counter() - t0
        app.r_m.set_active(KEYS.index(a.watch))
        if a.random:
            app.toggle_dna()
        app.frame_i = GROW; app.draw(); app.draw_comparison(force=True)
        base = a.save[:-4] if a.save.endswith(".png") else a.save
        app.fig.savefig(f"{base}_main.png", dpi=110); app.cfig.savefig(f"{base}_compare.png", dpi=110)
        print(f"Saved {base}_main.png and {base}_compare.png  ({a.gens} generations in {el:.1f}s, "
              f"{1000 * el / a.gens:.0f} ms/gen)")
        H = app.H()[-10:].mean(0)
        for i, k in enumerate(KEYS):
            print(f"  {k:<11} final score per run: {np.round(H[i, :, 0], 2)}")
    else:
        app.start(); plt.show()


if __name__ == "__main__":
    main()