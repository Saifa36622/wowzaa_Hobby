"""
GRU Explorer - an interactive GUI to understand Gated Recurrent Units (GRU)
==========================================================================

Tabs
  1. Step by Step         - walk through one GRU update in order: what each gate represents,
                            how it is calculated, with your numbers plugged in
  2. GRU Cell (one step)  - drag sliders, watch every gate and wire change live
  3. GRU Over Time        - feed a signal through a GRU and plot z, r, h over time
  4. Compare RNN/LSTM/GRU - memory test (vanishing gradient) + parameter counts

Requirements:  Python 3.8+, numpy, matplotlib   (tkinter ships with Python)
    pip install numpy matplotlib
Run:
    python gru_explorer.py

GRU convention used in this app (Chung et al. 2014):
    z  = sigmoid(Wz*x + Uz*h_prev + bz)          update gate  (how much to rewrite)
    r  = sigmoid(Wr*x + Ur*h_prev + br)          reset gate   (how much past to use)
    h~ = tanh(Wh*x + Uh*(r*h_prev) + bh)         candidate memory
    h  = (1 - z)*h_prev + z*h~                   new hidden state
PyTorch / Keras flip the role of z (h = z*h_prev + (1-z)*h~). Same idea, mirrored gate.
"""

import warnings
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

# When a plot is squeezed very small, matplotlib's auto-layout just gives up quietly.
warnings.filterwarnings("ignore", message=".*constrained_layout.*")
warnings.filterwarnings("ignore", message=".*[Ll]ayout not applied.*")


# =============================================================================
# Math helpers & models
# =============================================================================
def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def gru_step_scalar(x, h, p):
    """One step of a 1-unit GRU. Returns a dict with every intermediate value."""
    a_r = p["Wr"] * x + p["Ur"] * h + p["br"]
    r = sigmoid(a_r)
    a_z = p["Wz"] * x + p["Uz"] * h + p["bz"]
    z = sigmoid(a_z)
    rh = r * h
    a_c = p["Wh"] * x + p["Uh"] * rh + p["bh"]
    cand = np.tanh(a_c)
    keep = (1.0 - z) * h
    write = z * cand
    h_new = keep + write
    return dict(x=x, h=h, a_r=a_r, r=r, a_z=a_z, z=z, rh=rh,
                a_c=a_c, cand=cand, keep=keep, write=write, h_new=h_new)


class VanillaRNN:
    name = "Vanilla RNN"
    color = "#d62728"

    def __init__(self, n_in, n_h, scale, rng, memory_bias=0.0):
        self.W = rng.normal(0, scale / np.sqrt(n_in), (n_h, n_in))
        self.U = rng.normal(0, scale / np.sqrt(n_h), (n_h, n_h))
        self.b = np.zeros(n_h)
        self.n_h = n_h

    def init_state(self):
        return np.zeros(self.n_h)

    def step(self, x, s):
        return np.tanh(self.W @ x + self.U @ s + self.b)

    @staticmethod
    def hidden(s):
        return s

    @staticmethod
    def n_params(n_in, n_h):
        return 1 * n_h * (n_in + n_h + 1)


class LSTM:
    name = "LSTM"
    color = "#1f77b4"

    def __init__(self, n_in, n_h, scale, rng, memory_bias=0.0):
        def pair():
            return (rng.normal(0, scale / np.sqrt(n_in), (n_h, n_in)),
                    rng.normal(0, scale / np.sqrt(n_h), (n_h, n_h)))
        self.Wf, self.Uf = pair()
        self.Wi, self.Ui = pair()
        self.Wo, self.Uo = pair()
        self.Wg, self.Ug = pair()
        self.bf = np.full(n_h, memory_bias)   # forget-gate bias: high = remember
        self.bi = np.zeros(n_h)
        self.bo = np.zeros(n_h)
        self.bg = np.zeros(n_h)
        self.n_h = n_h

    def init_state(self):
        return (np.zeros(self.n_h), np.zeros(self.n_h))

    def step(self, x, s):
        h, c = s
        f = sigmoid(self.Wf @ x + self.Uf @ h + self.bf)
        i = sigmoid(self.Wi @ x + self.Ui @ h + self.bi)
        o = sigmoid(self.Wo @ x + self.Uo @ h + self.bo)
        g = np.tanh(self.Wg @ x + self.Ug @ h + self.bg)
        c = f * c + i * g
        h = o * np.tanh(c)
        return (h, c)

    @staticmethod
    def hidden(s):
        return np.concatenate(s)   # memory lives in both h and the cell state c

    @staticmethod
    def n_params(n_in, n_h):
        return 4 * n_h * (n_in + n_h + 1)


class GRU:
    name = "GRU"
    color = "#2ca02c"

    def __init__(self, n_in, n_h, scale, rng, memory_bias=0.0):
        def pair():
            return (rng.normal(0, scale / np.sqrt(n_in), (n_h, n_in)),
                    rng.normal(0, scale / np.sqrt(n_h), (n_h, n_h)))
        self.Wz, self.Uz = pair()
        self.Wr, self.Ur = pair()
        self.Wh, self.Uh = pair()
        self.bz = np.full(n_h, -memory_bias)  # update-gate bias: low = keep old memory
        self.br = np.zeros(n_h)
        self.bh = np.zeros(n_h)
        self.n_h = n_h

    def init_state(self):
        return np.zeros(self.n_h)

    def step(self, x, h):
        z = sigmoid(self.Wz @ x + self.Uz @ h + self.bz)
        r = sigmoid(self.Wr @ x + self.Ur @ h + self.br)
        cand = np.tanh(self.Wh @ x + self.Uh @ (r * h) + self.bh)
        return (1 - z) * h + z * cand

    @staticmethod
    def hidden(s):
        return s

    @staticmethod
    def n_params(n_in, n_h):
        return 3 * n_h * (n_in + n_h + 1)


MODEL_CLASSES = [VanillaRNN, LSTM, GRU]


def memory_curve(model, X, v_dir, eps=1e-5):
    """How strongly does the state at time t still depend on the input at t=0?
    Finite-difference estimate of ||d h_t / d x_0 . v||  (a gradient-flow proxy)."""
    T = X.shape[0]
    XB = X.copy()
    XB[0] = XB[0] + eps * v_dir
    sa, sb = model.init_state(), model.init_state()
    out = np.empty(T)
    for t in range(T):
        sa = model.step(X[t], sa)
        sb = model.step(XB[t], sb)
        out[t] = np.linalg.norm(model.hidden(sa) - model.hidden(sb)) / eps
    return out


# =============================================================================
# Presets & signals for the 1-unit GRU
# =============================================================================
PARAM_SPECS = [
    ("Wz", "Wz  (x \u2192 z)", 1.0),
    ("Uz", "Uz  (h \u2192 z)", 0.5),
    ("bz", "bz  (z bias)", 0.0),
    ("Wr", "Wr  (x \u2192 r)", 1.0),
    ("Ur", "Ur  (h \u2192 r)", 0.5),
    ("br", "br  (r bias)", 0.0),
    ("Wh", "Wh  (x \u2192 h~)", 1.5),
    ("Uh", "Uh  (r\u00b7h \u2192 h~)", 1.0),
    ("bh", "bh  (h~ bias)", 0.0),
]

PRESETS = {
    "Balanced (default)":
        dict(Wz=1.0, Uz=0.5, bz=0.0, Wr=1.0, Ur=0.5, br=0.0, Wh=1.5, Uh=1.0, bh=0.0),
    "Long memory (latch)":
        dict(Wz=6.0, Uz=0.0, bz=-4.0, Wr=0.0, Ur=0.0, br=3.0, Wh=3.0, Uh=1.0, bh=0.0),
    "Flip-flop (bistable memory)":
        dict(Wz=0.0, Uz=0.0, bz=0.0, Wr=0.0, Ur=0.0, br=5.0, Wh=4.0, Uh=3.0, bh=0.0),
    "Leaky smoother (moving average)":
        dict(Wz=0.0, Uz=0.0, bz=-2.0, Wr=0.0, Ur=0.0, br=0.0, Wh=1.0, Uh=0.0, bh=0.0),
    "Fast follower (no memory)":
        dict(Wz=0.0, Uz=0.0, bz=4.0, Wr=0.0, Ur=0.0, br=0.0, Wh=2.0, Uh=0.0, bh=0.0),
    "Reset-heavy (ignore the past)":
        dict(Wz=0.0, Uz=0.0, bz=0.0, Wr=0.0, Ur=0.0, br=-5.0, Wh=2.0, Uh=3.0, bh=0.0),
}

PRESET_TIPS = {
    "Balanced (default)": "General-purpose weights. A good starting point.",
    "Long memory (latch)": "Update gate is almost shut (z\u22480) unless a strong positive input arrives. "
                           "Try the 'Pulse' signal: the GRU writes once, then holds the value.",
    "Flip-flop (bistable memory)": "Strong self-feedback (Uh) makes +1 and \u22121 stable states. "
                                   "Try 'Pulse train \u00b1': each pulse flips the memory.",
    "Leaky smoother (moving average)": "Constant z\u22480.12: each step keeps 88% of memory and writes 12% "
                                       "of the new input - an exponential moving average.",
    "Fast follower (no memory)": "z\u22481: the state is overwritten every step, so h just tracks the input.",
    "Reset-heavy (ignore the past)": "r\u22480: the candidate never looks at old memory, "
                                     "so a large Uh has no effect at all.",
}

# Walkthrough steps: (button text, title, the question this step answers)
WALK_STEPS = [
    ("0   Inputs  x(t), h(t-1)", "Step 0 · The inputs",
     "What does the GRU cell receive at time t?"),
    ("1   Reset gate  r", "Step 1 · Reset gate r",
     "How much of the old memory should be used to draft the new memory?"),
    ("2   Update gate  z", "Step 2 · Update gate z",
     "How much should the memory change at this time step?"),
    ("3   Candidate memory  h~", "Step 3 · Candidate memory h~",
     "What new content would the GRU like to write into memory?"),
    ("4   New hidden state  h(t)", "Step 4 · New hidden state h(t)",
     "What is the new memory - and the output - at time t?"),
    ("✓   Summary", "Summary · the whole time step",
     "The whole time step at a glance"),
]

# Parts of the cell diagram, in the order the walkthrough computes them
CELL_GROUPS = ("in", "r", "z", "cand", "out")

SIGNALS = ["Pulse","Pulse train \u00b1", "Step", "Sine", "Square wave", "Noisy sine", "Random noise"]


def make_signal(kind, T, amp):
    t = np.arange(T)
    x = np.zeros(T)
    if kind == "Pulse":
        x[int(T * 0.15)] = 1.0
    elif kind == "Pulse train \u00b1":
        x[int(T * 0.15)] = 1.0
        x[int(T * 0.45)] = -1.0
        x[int(T * 0.75)] = 1.0
    elif kind == "Step":
        x[int(T * 0.25):] = 1.0
    elif kind == "Sine":
        x = np.sin(2 * np.pi * t / 25)
    elif kind == "Square wave":
        x = np.sign(np.sin(2 * np.pi * t / 30 + 1e-9))
    elif kind == "Noisy sine":
        x = np.sin(2 * np.pi * t / 25) + 0.4 * np.random.default_rng(7).normal(size=T)
    elif kind == "Random noise":
        x = np.random.default_rng(3).normal(size=T)
    return amp * x


# =============================================================================
# UI helpers
# =============================================================================
def slider_row(parent, row, text, var, frm, to, fmt="{:+.2f}", length=170, tip=None):
    ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=1)
    sc = ttk.Scale(parent, from_=frm, to=to, variable=var, orient="horizontal", length=length)
    sc.grid(row=row, column=1, sticky="ew", padx=4)
    route_wheel(sc)          # mouse wheel scrolls the panel instead of moving the slider
    val = ttk.Label(parent, width=7, anchor="e")
    val.grid(row=row, column=2, sticky="e")

    def upd(*_):
        try:
            v = var.get()
            val.config(text=fmt.format(int(round(v)) if "d" in fmt else v))
        except (tk.TclError, ValueError):
            pass
    var.trace_add("write", upd)
    upd()
    return sc


def lerp_color(c0, c1, t):
    t = float(np.clip(t, 0, 1))
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gate_color(v):          # sigmoid output 0..1 : grey -> green
    return lerp_color("#e9ecef", "#1e8449", v)


def signed_color(v):        # -1..1 : blue <- grey -> orange
    if v >= 0:
        return lerp_color("#e9ecef", "#d35400", v)
    return lerp_color("#e9ecef", "#2457c5", -v)


def text_on(fill):
    r, g, b = (int(fill[i:i + 2], 16) for i in (1, 3, 5))
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#111111"


# =============================================================================
# Responsive-layout helpers
# =============================================================================
class ScrollFrame(ttk.Frame):
    """A vertically scrollable panel. Put widgets inside `.inner`.
    The content stretches to the panel width, and the scrollbar only shows up
    when the content is taller than the window."""

    def __init__(self, master, width=330, padding=10):
        super().__init__(master)
        bg = ttk.Style(self).lookup("TFrame", "background") or "#dcdad5"
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0, bd=0, bg=bg,
                                yscrollincrement=20)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = ttk.Frame(self.canvas, padding=padding)
        self.inner.columnconfigure(1, weight=1)          # sliders stretch with the panel
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._sync)
        self.canvas.bind("<Configure>", self._on_canvas)

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)
        self._sync()

    def _sync(self, _=None):
        h = self.inner.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, self.inner.winfo_reqwidth(), h))
        if h > self.canvas.winfo_height() + 1:
            self.vsb.grid()
        else:
            self.vsb.grid_remove()
            self.canvas.yview_moveto(0)

    def scroll(self, step):
        if self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1:
            self.canvas.yview_scroll(step * 2, "units")


def wheel_scroll(event):
    """Scroll whichever ScrollFrame is under the mouse pointer (Windows, macOS, Linux)."""
    try:
        w = event.widget.winfo_containing(event.x_root, event.y_root)
    except (KeyError, tk.TclError, AttributeError):
        return None
    while w is not None and not isinstance(w, ScrollFrame):
        w = getattr(w, "master", None)
    if w is None:
        return None
    if getattr(event, "num", None) == 4:
        step = -1
    elif getattr(event, "num", None) == 5:
        step = 1
    else:
        step = -1 if event.delta > 0 else 1
    w.scroll(step)
    return "break"


def route_wheel(widget):
    """Stop sliders/comboboxes from changing value under the wheel; scroll the panel instead."""
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        widget.bind(seq, lambda e: (wheel_scroll(e), "break")[1])


def auto_wrap(label, pad=6):
    """Make a label re-wrap its text whenever its width changes."""
    def on_conf(e):
        new = max(60, e.width - pad)
        if int(str(label.cget("wraplength")) or 0) != new:
            label.configure(wraplength=new)
    label.bind("<Configure>", on_conf, add="+")


def autohide(sb):
    """Scrollbar command that hides the scrollbar when everything is visible."""
    def setter(lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            sb.grid_remove()
        else:
            sb.grid()
        sb.set(lo, hi)
    return setter


def make_figure(w=6, h=4):
    """A matplotlib figure that re-lays itself out automatically when resized."""
    try:
        return Figure(figsize=(w, h), dpi=100, layout="constrained")
    except TypeError:                                    # matplotlib < 3.6
        return Figure(figsize=(w, h), dpi=100, constrained_layout=True)


def small_plot_widget(fig_canvas, master_w=520, master_h=340):
    """Pack a matplotlib canvas so it fills the space but doesn't force a big window."""
    w = fig_canvas.get_tk_widget()
    w.configure(width=master_w, height=master_h)
    w.pack(fill="both", expand=True)
    return w


# =============================================================================
# Main application
# =============================================================================
class GRUExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GRU Explorer - Gated Recurrent Units, interactively")
        # Open at a size that fits the screen; allow shrinking down to small laptops.
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1320, sw - 60), min(860, sh - 90)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(760, 520)
        self._pending = {}
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, wheel_scroll, add="+")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Head.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("Tip.TLabel", foreground="#555555")
        style.configure("Step.TLabel", font=("TkDefaultFont", 13, "bold"), foreground="#1e3a5f")
        style.configure("Step.Toolbutton", anchor="w", padding=(8, 4))
        style.map("Step.Toolbutton", background=[("selected", "#e67e22"), ("active", "#f6dcc4")],
                  foreground=[("selected", "white")])

        # Shared GRU parameters (walkthrough, cell and sequence tabs use the same weights)
        self.p = {k: tk.DoubleVar(value=d) for k, _, d in PARAM_SPECS}
        for v in self.p.values():
            v.trace_add("write", lambda *_: self.schedule("gru", self.refresh_gru_views))
        # Shared one-step inputs (walkthrough and cell tabs)
        self.x_var = tk.DoubleVar(value=0.8)
        self.hprev_var = tk.DoubleVar(value=0.5)
        for v in (self.x_var, self.hprev_var):
            v.trace_add("write", lambda *_: (self.schedule("cell", self.update_cell),
                                             self.schedule("walk", self.update_walk)))
        self.tip_labels = []
        for seq, d in (("<Left>", -1), ("<Right>", 1)):
            self.bind_all(seq, lambda e, d=d: self.walk_key(d), add="+")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb
        self.tab_walk = ttk.Frame(nb)
        self.tab_cell = ttk.Frame(nb)
        self.tab_seq = ttk.Frame(nb)
        self.tab_cmp = ttk.Frame(nb)
        nb.add(self.tab_walk, text="  1. Step by Step  ")
        nb.add(self.tab_cell, text="  2. GRU Cell (one step)  ")
        nb.add(self.tab_seq, text="  3. GRU Over Time  ")
        nb.add(self.tab_cmp, text="  4. Compare RNN / LSTM / GRU  ")

        self.build_walk_tab()
        self.build_cell_tab()
        self.build_seq_tab()
        self.build_compare_tab()

        self.refresh_gru_views()
        self.run_compare()

    # ------------------------------------------------------------------ utils
    def schedule(self, key, fn, delay=40):
        if key in self._pending:
            self.after_cancel(self._pending[key])
        self._pending[key] = self.after(delay, fn)

    def params(self):
        return {k: v.get() for k, v in self.p.items()}

    def make_scrollframe(self, pw):
        """Scrollable control panel for the left side of a horizontal PanedWindow.
        Whenever the window is resized the divider moves so the controls take what
        they need, but never more than ~36% of the width (you can still drag it)."""
        sf = ScrollFrame(pw)
        last = {"w": 0}

        def fit(e):
            if e.widget is not pw or e.width < 100 or e.width == last["w"]:
                return
            last["w"] = e.width
            want = sf.inner.winfo_reqwidth() + sf.vsb.winfo_reqwidth() + 4
            target = min(want, max(230, int(e.width * 0.36)))
            pw.after_idle(lambda: pw.sashpos(0, target))
        pw.bind("<Configure>", fit, add="+")
        return sf

    @staticmethod
    def fig_scale(fig, ref=850):
        """Font scale for a matplotlib figure: smaller text when the plot is small."""
        w = fig.get_size_inches()[0] * fig.dpi
        return float(np.clip(w / ref, 0.62, 1.15))

    def apply_preset(self, name):
        for k, val in PRESETS[name].items():
            self.p[k].set(val)
        tip = PRESET_TIPS.get(name, "")
        for lbl in self.tip_labels:
            lbl.config(text=tip)

    def build_param_panel(self, parent, start_row, preset_var):
        ttk.Label(parent, text="Preset", style="Head.TLabel").grid(
            row=start_row, column=0, columnspan=3, sticky="w", pady=(10, 2))
        cb = ttk.Combobox(parent, textvariable=preset_var, values=list(PRESETS),
                          state="readonly", width=26)
        cb.grid(row=start_row + 1, column=0, columnspan=3, sticky="ew")
        route_wheel(cb)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(preset_var.get()))

        wl = ttk.Label(parent, text="GRU weights (shared by tabs 1-3)", style="Head.TLabel")
        wl.grid(row=start_row + 2, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        auto_wrap(wl)
        groups = {"Wz": "Update gate z", "Wr": "Reset gate r", "Wh": "Candidate h~"}
        r = start_row + 3
        for key, label, _ in PARAM_SPECS:
            if key in groups:
                ttk.Label(parent, text=groups[key], foreground="#1e8449" if key != "Wh" else "#d35400"
                          ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
                r += 1
            slider_row(parent, r, label, self.p[key], -6, 6, length=150)
            r += 1
        ttk.Button(parent, text="Reset weights", command=lambda: self.apply_preset("Balanced (default)")
                   ).grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        return r + 1

    def add_tip_label(self, parent, row):
        tip = ttk.Label(parent, text=PRESET_TIPS["Balanced (default)"], wraplength=290,
                        style="Tip.TLabel", justify="left")
        tip.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        auto_wrap(tip)
        self.tip_labels.append(tip)

    def refresh_gru_views(self):
        self.update_walk()
        self.update_cell()
        self.update_seq()

    # ================================================================ TAB 1
    def build_walk_tab(self):
        pw = ttk.PanedWindow(self.tab_walk, orient="horizontal")
        pw.pack(fill="both", expand=True)
        sf = self.make_scrollframe(pw)
        left = sf.inner
        right = ttk.Frame(pw, padding=(4, 10, 10, 10))
        pw.add(sf, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Steps of one GRU update", style="Head.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        self.walk_step = tk.IntVar(value=0)
        for i, (short, _, _) in enumerate(WALK_STEPS):
            ttk.Radiobutton(left, text=short, variable=self.walk_step, value=i, style="Step.Toolbutton",
                            command=self.update_walk).grid(row=1 + i, column=0, columnspan=3,
                                                           sticky="ew", pady=1)
        r = 1 + len(WALK_STEPS)
        ttk.Button(left, text="Feed h(t) back → next time step", command=self.walk_feed_back).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        keys = ttk.Label(left, text="Tip: the ← / → arrow keys move between steps.",
                         style="Tip.TLabel")
        keys.grid(row=r + 1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        auto_wrap(keys)

        ttk.Label(left, text="Inputs for this step", style="Head.TLabel").grid(
            row=r + 2, column=0, columnspan=3, sticky="w", pady=(10, 2))
        slider_row(left, r + 3, "x(t)  input", self.x_var, -2, 2, length=150)
        slider_row(left, r + 4, "h(t-1)  memory", self.hprev_var, -1, 1, length=150)
        self.walk_preset = tk.StringVar(value="Balanced (default)")
        r = self.build_param_panel(left, r + 5, self.walk_preset)
        self.add_tip_label(left, r)

        # header: back / step title / next
        head = ttk.Frame(right)
        head.pack(fill="x")
        self.walk_back = ttk.Button(head, text="◀ Back", width=8, command=lambda: self.walk_go(-1))
        self.walk_back.pack(side="left")
        self.walk_next = ttk.Button(head, text="Next ▶", width=8, command=lambda: self.walk_go(1))
        self.walk_next.pack(side="right")
        self.walk_title = ttk.Label(head, text="", style="Step.TLabel", anchor="center")
        self.walk_title.pack(side="left", fill="x", expand=True, padx=8)

        # diagram on top, explanation + small plot below - drag the divider between them
        vp = ttk.PanedWindow(right, orient="vertical")
        vp.pack(fill="both", expand=True, pady=(6, 0))
        top = ttk.Frame(vp)
        bottom = ttk.Frame(vp, padding=(0, 6, 0, 0))
        vp.add(top, weight=1)
        vp.add(bottom, weight=1)
        self._walk = None
        self._walk_shown = None
        self.walk_canvas = tk.Canvas(top, width=560, height=250, bg="white",
                                     highlightthickness=1, highlightbackground="#d0d7de")
        self.walk_canvas.pack(fill="both", expand=True)
        self.walk_canvas.bind("<Configure>", lambda e: self.schedule("walkdraw", self.redraw_walk, 15))

        self.walk_text = tk.Text(bottom, height=12, width=40, font=("TkDefaultFont", 10), wrap="word",
                                 relief="flat", bg="#fbfbfc", padx=12, pady=8, spacing1=1, spacing3=1)
        ysb = ttk.Scrollbar(bottom, command=self.walk_text.yview)
        xsb = ttk.Scrollbar(bottom, orient="horizontal", command=self.walk_text.xview)
        self.walk_text.configure(yscrollcommand=autohide(ysb), xscrollcommand=autohide(xsb))
        self.walk_plot = tk.Canvas(bottom, width=270, height=190, bg="white",
                                   highlightthickness=1, highlightbackground="#d0d7de")
        self.walk_plot.bind("<Configure>", lambda e: self.schedule("walkplot", self.draw_walk_plot, 15))
        self.walk_text.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.walk_plot.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(8, 0))
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1)

        t = self.walk_text
        t.tag_configure("q", font=("TkDefaultFont", 12, "italic"), foreground="#1e3a5f", spacing3=4)
        t.tag_configure("h2", font=("TkDefaultFont", 11, "bold"), foreground="#1e8449",
                        spacing1=10, spacing3=3)
        t.tag_configure("mono", font=("Courier", 10), background="#eef2f7", lmargin1=16, lmargin2=16,
                        wrap="none")   # equations never wrap; scroll sideways if narrow
        t.tag_configure("bullet", lmargin1=8, lmargin2=22)

    def walk_go(self, delta):
        k = int(np.clip(self.walk_step.get() + delta, 0, len(WALK_STEPS) - 1))
        if k != self.walk_step.get():
            self.walk_step.set(k)
            self.update_walk()

    def walk_key(self, delta):
        """Arrow keys step through the walkthrough (unless a slider/combobox has the focus)."""
        try:
            if self.nb.select() != str(self.tab_walk):
                return None
            if isinstance(self.focus_get(), (ttk.Scale, ttk.Combobox, ttk.Entry, tk.Entry)):
                return None
        except (KeyError, tk.TclError):
            return None
        self.walk_go(delta)
        return "break"

    def walk_feed_back(self):
        """Start the next time step: the new memory h(t) becomes h(t-1)."""
        if self._walk is not None:
            self.hprev_var.set(round(float(np.clip(self._walk[0]["h_new"], -1, 1)), 2))
        self.walk_step.set(0)
        self.update_walk()

    def redraw_walk(self):
        if self._walk is not None:
            self.draw_cell(self.walk_canvas, *self._walk)

    def update_walk(self):
        if not hasattr(self, "walk_text"):
            return
        step = self.walk_step.get()
        # rounded to 2 decimals so every number in the explanation can be checked by hand
        p = {k: round(val, 2) for k, val in self.params().items()}
        v = gru_step_scalar(round(self.x_var.get(), 2), round(self.hprev_var.get(), 2), p)
        self._walk = (v, step)
        self.walk_title.config(text=WALK_STEPS[step][1])
        self.walk_back.state(["disabled" if step == 0 else "!disabled"])
        self.walk_next.state(["disabled" if step == len(WALK_STEPS) - 1 else "!disabled"])
        self.draw_cell(self.walk_canvas, v, step)
        self.draw_walk_plot()
        self.fill_walk_text(v, p, step)

    def fill_walk_text(self, v, p, step):
        t = self.walk_text
        yview = t.yview()[0]
        t.configure(state="normal")
        t.delete("1.0", "end")

        def add(s, *tags):
            t.insert("end", s, tags)

        def section(title):
            add(title + "\n", "h2")

        def bullets(*items):
            for s in items:
                add("• " + s + "\n", "bullet")

        def mono(s):
            add(s + "\n", "mono")

        def num(u):            # weights/inputs have 2 decimals, computed values get 4
            return f"{u:+.2f}" if abs(u - round(u, 2)) < 1e-12 else f"{u:+.4f}"

        def mul(a, b):
            return f"({num(a)}) × ({num(b)}) = {num(a * b)}"

        def total(*vals):
            s = num(vals[0])
            for u in vals[1:]:
                s += f" {'+' if u >= 0 else '-'} {num(abs(u))[1:]}"
            return s

        def expo(u):
            return f"{u:.4f}" if 1e-4 <= u < 1e5 else f"{u:.3e}"

        def row(lbl, rest, w=16):
            return f"{lbl:<{w}}{rest}"

        x, h, r, z, cand, hn = v["x"], v["h"], v["r"], v["z"], v["cand"], v["h_new"]
        add(WALK_STEPS[step][2] + "\n", "q")

        if step == 0:
            section("What goes into the cell")
            bullets(f"x(t) = {num(x)}  -  the input at this time step, e.g. today's value in a time "
                    "series, or the current word turned into numbers.",
                    f"h(t-1) = {num(h)}  -  the hidden state, i.e. the GRU's memory. It was produced by "
                    "the previous time step and summarises everything seen so far. It always stays "
                    "between -1 and +1.",
                    "The weights  -  every gate has its own three numbers: W multiplies the input, "
                    "U multiplies the memory and b is a bias. Training learns them; here you set them "
                    "with the sliders on the left.")
            section("The plan: four calculations, always in this order")
            mono("Step 1  reset gate    r    = σ( Wr·x(t) + Ur·h(t-1) + br )\n"
                 "Step 2  update gate   z    = σ( Wz·x(t) + Uz·h(t-1) + bz )\n"
                 "Step 3  candidate     h~   = tanh( Wh·x(t) + Uh·(r·h(t-1)) + bh )\n"
                 "Step 4  new memory    h(t) = (1 - z)·h(t-1) + z·h~")
            section("Two squashing functions")
            bullets("σ (sigmoid) = 1 / (1 + e^(-a)) turns any number into 0..1. The gates use it: "
                    "a gate value works like a percentage or a dimmer switch (0 = closed, 1 = fully open).",
                    "tanh = (e^a - e^(-a)) / (e^a + e^(-a)) turns any number into -1..+1. It is used for "
                    "the content that is stored in memory, which can be positive or negative.")
            section("Good to know")
            bullets("Steps 1 and 2 only need x(t) and h(t-1), so they could be computed in parallel. "
                    "Step 3 needs r, and step 4 needs z and h~.",
                    "This GRU has a single unit, so every value is a plain number. In a real layer x and h "
                    "are vectors and W, U are matrices, but each unit does exactly this arithmetic.",
                    "In this walkthrough the inputs and weights are rounded to 2 decimals, so you can "
                    "check every number by hand or with a calculator.")

        elif step in (1, 2):
            n = "r" if step == 1 else "z"
            W, U, b = p["W" + n], p["U" + n], p["b" + n]
            a, g = v["a_" + n], v[n]
            if step == 1:
                section("What the reset gate represents")
                bullets("r is a number between 0 and 1 that is multiplied with the old memory before "
                        "the candidate (step 3) looks at it  -  a filter on the past.",
                        "r ≈ 0 means 'reset': draft the new memory from the current input only, as if "
                        "there were no history (e.g. a new sentence starts and the old context no "
                        "longer matters).",
                        "r ≈ 1 means the whole memory is used when drafting.",
                        "r never changes h(t) directly; it only controls what the candidate sees.")
            else:
                section("What the update gate represents")
                bullets("z is a number between 0 and 1 that decides how much of the memory is replaced "
                        "in step 4.",
                        "z ≈ 0 means 'keep': the old memory is copied forward unchanged. This is how a "
                        "GRU remembers things for many steps (and why its gradients don't vanish).",
                        "z ≈ 1 means 'overwrite': the memory is replaced by the candidate h~.",
                        "One gate does two jobs: keep = 1 - z and write = z always add up to 1. "
                        "(An LSTM uses two separate gates, forget and input, for this.)")
            section("Formula")
            mono(f"{n} = σ( W{n}·x(t) + U{n}·h(t-1) + b{n} )")
            section("Calculation with the current numbers")
            mono(row("a) input part", f"{f'W{n}' + chr(183) + 'x(t)':<11}= {mul(W, x)}") + "\n" +
                 row("b) memory part", f"{f'U{n}' + chr(183) + 'h(t-1)':<11}= {mul(U, h)}") + "\n" +
                 row("c) bias", f"{f'b{n}':<11}= {num(b)}") + "\n" +
                 row("d) add them", f"a_{n} = {total(W * x, U * h, b)} = {num(a)}") + "\n" +
                 row("e) sigmoid", f"{n} = σ(a_{n}) = 1 / (1 + e^({-a:+.4f}))") + "\n" +
                 row("", f"  = 1 / (1 + {expo(np.exp(-a))}) = {g:.4f}"))
            section("What the result means")
            parts = {"the input  W·x": W * x, "the memory  U·h": U * h, "the bias  b": b}
            big = max(parts, key=lambda k: abs(parts[k]))
            why = (f"The biggest push came from {big} = {parts[big]:+.4f}. Positive values push {n} "
                   f"towards 1 (open), negative values towards 0 (closed)."
                   if abs(parts[big]) > 1e-9 else
                   f"Everything adds up to 0, so {n} sits exactly in the middle at 0.5.")
            if step == 1:
                if r < 0.2:
                    s = f"r = {r:.2f} is nearly CLOSED: the candidate will be built almost only from the current input."
                elif r > 0.8:
                    s = f"r = {r:.2f} is OPEN: the candidate will use the old memory fully."
                else:
                    s = f"r = {r:.2f}: {100 * r:.0f}% of the old memory is let through to the candidate."
                bullets(s, why,
                        f"So in step 3 the candidate sees r·h(t-1) = {r:.4f} × ({num(h)}) = "
                        f"{v['rh']:+.4f} instead of h(t-1) = {num(h)}.")
                section("Next")
                bullets("Step 2: the update gate is computed exactly the same way, but with its own "
                        "weights Wz, Uz, bz, so it can make a different decision.")
            else:
                if z < 0.2:
                    s = f"z = {z:.2f} is mostly CLOSED: the GRU will keep {100 * (1 - z):.0f}% of its old memory."
                elif z > 0.8:
                    s = f"z = {z:.2f} is mostly OPEN: {100 * z:.0f}% of the memory will be replaced by the candidate."
                else:
                    s = f"z = {z:.2f} is half-open: the new memory will be a blend of old and new."
                bullets(s, why,
                        f"So in step 4: keep 1 - z = {100 * (1 - z):.0f}% of the old memory and write "
                        f"z = {100 * z:.0f}% of the candidate.")
                section("Next")
                bullets("Step 3 uses r (from step 1) to build the candidate. z waits until step 4.")

        elif step == 3:
            Wh, Uh, bh, rh, a = p["Wh"], p["Uh"], p["bh"], v["rh"], v["a_c"]
            section("What the candidate represents")
            bullets("h~ is the proposal for the new memory, built from the current input and the part of "
                    "the old memory that the reset gate lets through.",
                    "It uses tanh, so it lies between -1 and +1 and can be positive or negative: it carries "
                    "content, not a percentage like the gates.",
                    "It is only a proposal: the update gate decides in step 4 how much of it is actually written.")
            section("Formula")
            mono("h~ = tanh( Wh·x(t) + Uh·(r·h(t-1)) + bh )")
            section("Calculation with the current numbers")
            ep, em = np.exp(a), np.exp(-a)
            mono(row("a) apply reset", f"{'r' + chr(183) + 'h(t-1)':<13}= {mul(r, h)}") + "\n" +
                 row("b) input part", f"{'Wh' + chr(183) + 'x(t)':<13}= {mul(Wh, x)}") + "\n" +
                 row("c) memory part", f"{'Uh' + chr(183) + '(r' + chr(183) + 'h)':<13}= {mul(Uh, rh)}") + "\n" +
                 row("d) bias", f"{'bh':<13}= {num(bh)}") + "\n" +
                 row("e) add them", f"a_h = {total(Wh * x, Uh * rh, bh)} = {num(a)}") + "\n" +
                 row("f) tanh", "h~ = tanh(a_h) = (e^a - e^(-a)) / (e^a + e^(-a))") + "\n" +
                 row("", f"   = ({expo(ep)} - {expo(em)}) / ({expo(ep)} + {expo(em)}) = {cand:+.4f}"))
            section("What the result means")
            if abs(cand) < 0.2:
                s = f"h~ = {cand:+.2f} is close to 0: a weak, almost neutral proposal."
            else:
                s = (f"h~ = {cand:+.2f}: the cell proposes to store a {'positive' if cand > 0 else 'negative'} "
                     f"value{' (strongly - tanh is almost saturated)' if abs(cand) > 0.9 else ''}.")
            full_a = Wh * x + Uh * h + bh
            if abs(Uh) < 1e-9:
                s2 = "Uh = 0, so the old memory has no effect on the candidate at all, whatever r is."
            elif r > 0.99:
                s2 = "r ≈ 1, so the reset gate changed almost nothing here."
            else:
                s2 = (f"Without the reset gate (r = 1) the candidate would be tanh({full_a:+.4f}) = "
                      f"{np.tanh(full_a):+.4f}. The reset gate changed it by {cand - np.tanh(full_a):+.4f}.")
            bullets(s, s2)
            section("Next")
            bullets("Step 4 uses z to decide how much of this proposal replaces the old memory.")

        elif step == 4:
            keep, write = v["keep"], v["write"]
            section("What the new hidden state represents")
            bullets("h(t) is a blend of the old memory and the candidate, mixed by the update gate: "
                    "(1 - z) of the old memory is kept and z of the candidate is written.",
                    "h(t) is the output of the GRU at time t, and it is also the memory handed to the next "
                    "time step, where it becomes h(t-1).")
            section("Formula")
            mono("h(t) = (1 - z)·h(t-1) + z·h~\n"
                 "     = h(t-1) + z·(h~ - h(t-1))   <- move a share z towards h~")
            section("Calculation with the current numbers")
            mono(row("a) keep share", f"1 - z = 1 - {z:.4f} = {1 - z:.4f}") + "\n" +
                 row("b) keep part", f"(1 - z)·h(t-1) = {mul(1 - z, h)}") + "\n" +
                 row("c) write part", f"z·h~ = {mul(z, cand)}") + "\n" +
                 row("d) add them", f"h(t) = {total(keep, write)} = {hn:+.4f}"))
            section("What the result means")
            bullets(f"The new memory is {100 * (1 - z):.0f}% old memory and {100 * z:.0f}% candidate.",
                    f"h(t) = {hn:+.4f} lies between h(t-1) = {num(h)} and h~ = {cand:+.4f}: the memory "
                    f"moved {100 * z:.0f}% of the way towards the candidate (it changed by {hn - h:+.4f}).",
                    "Press 'Feed h(t) back' on the left to use this h(t) as h(t-1) for the next time step.")

        else:
            section("All four calculations")
            mono(f"1  reset gate    r    = σ({v['a_r']:+.4f})    = {r:.4f}\n"
                 f"2  update gate   z    = σ({v['a_z']:+.4f})    = {z:.4f}\n"
                 f"3  candidate     h~   = tanh({v['a_c']:+.4f}) = {cand:+.4f}\n"
                 f"4  new memory    h(t) = {1 - z:.4f}·({num(h)}) + {z:.4f}·({cand:+.4f}) = {hn:+.4f}")
            section("What each part does and represents")
            bullets(f"Reset gate r = {r:.2f}  -  a filter on the past: how much of the old memory may be "
                    "used to draft the new content.",
                    f"Update gate z = {z:.2f}  -  a mixing knob: how much the memory changes "
                    "(keep 1 - z, write z). This is what lets a GRU remember for a long time.",
                    f"Candidate h~ = {cand:+.2f}  -  the proposal: new content computed from the input "
                    "and the filtered memory.",
                    f"Hidden state h(t) = {hn:+.2f}  -  the result: the output at time t and the memory "
                    "for time t+1.")
            section("Continue")
            bullets(f"Press 'Feed h(t) back' to start the next time step with h(t-1) = {hn:+.2f}.",
                    "Tab 2 shows the same cell with all values at once; tab 3 repeats these four steps "
                    "for a whole sequence.")

        t.configure(state="disabled")
        t.yview_moveto(yview if self._walk_shown == step else 0)   # keep scroll while dragging sliders
        self._walk_shown = step

    # ---- small plot next to the explanation
    def draw_walk_plot(self):
        c = self.walk_plot
        c.delete("all")
        if self._walk is None:
            return
        v, step = self._walk
        W, H = max(c.winfo_width(), 180), max(c.winfo_height(), 150)
        if step in (1, 2, 3):
            key, a_key, name = (("r", "a_r", "r"), ("z", "a_z", "z"), ("cand", "a_c", "h~"))[step - 1]
            self.plot_activation(c, W, H, v[a_key], v[key], name, tanh=step == 3)
        elif step == 4:
            self.plot_blend(c, W, H, v)
        elif step == 0:
            self.plot_bars(c, W, H, "The inputs", [("x(t)", v["x"], 2.0), ("h(t-1)", v["h"], 1.0)])
        else:
            self.plot_bars(c, W, H, "All values of this step",
                           [("r", v["r"], 0), ("z", v["z"], 0), ("h(t-1)", v["h"], 1.0),
                            ("h~", v["cand"], 1.0), ("h(t)", v["h_new"], 1.0)])

    @staticmethod
    def plot_activation(c, W, H, a, val, name, tanh):
        """The squashing curve with the current point a -> f(a) marked."""
        f = np.tanh if tanh else sigmoid
        lo = -1.0 if tanh else 0.0
        R = max(6.0, abs(a) * 1.1)
        L, Rr, T, B = 34, W - 12, 36, H - 38

        def px(u):
            return L + (u + R) / (2 * R) * (Rr - L)

        def py(y):
            return B - (y - lo) / (1 - lo) * (B - T)
        small, bold = ("TkDefaultFont", 8), ("TkDefaultFont", 9, "bold")
        c.create_text(W / 2, 14, text="tanh: any number → -1..+1" if tanh else
                      "sigmoid: any number → 0..1", font=bold, fill="#34495e")
        for yv in ((-1, 0, 1) if tanh else (0, 0.5, 1)):
            c.create_line(L, py(yv), Rr, py(yv), fill="#eef1f4")
            c.create_text(L - 4, py(yv), text=f"{yv:g}", anchor="e", font=small, fill="#666")
        c.create_line(L, py(0), Rr, py(0), fill="#9aa5b1")
        c.create_line(px(0), T, px(0), B, fill="#c3c9d0")
        for u in (-R, 0, R):
            c.create_text(px(u), B + 10, text=f"{u:+.0f}" if u else "0", font=small, fill="#666")
        pts = [coord for u in np.linspace(-R, R, 160) for coord in (px(u), py(f(u)))]
        c.create_line(*pts, fill="#34495e", width=2)
        X, Y = px(a), py(val)
        c.create_line(X, py(0), X, Y, fill="#e67e22", dash=(3, 2))
        c.create_line(L, Y, X, Y, fill="#e67e22", dash=(3, 2))
        c.create_oval(X - 5, Y - 5, X + 5, Y + 5, fill="#e67e22", outline="white", width=2)
        c.create_text(float(np.clip(X, L + 30, Rr - 30)), B + 25, text=f"a = {a:+.3f}",
                      font=bold, fill="#e67e22")
        left_side = X < (L + Rr) / 2
        c.create_text(X + 8 if left_side else X - 8, Y - 10, anchor="sw" if left_side else "se",
                      text=f"{name} = {val:+.3f}" if tanh else f"{name} = {val:.3f}",
                      font=bold, fill="#e67e22")

    @staticmethod
    def plot_blend(c, W, H, v):
        """Number line: h(t) sits between h(t-1) and h~, a share z of the way."""
        h, cand, hn, z = v["h"], v["cand"], v["h_new"], v["z"]
        L, R = 24, W - 24
        y = H * 0.56

        def px(u):
            return L + (u + 1) / 2 * (R - L)

        def lx(u):
            return float(np.clip(px(u), L + 18, R - 18))
        small, bold = ("TkDefaultFont", 8), ("TkDefaultFont", 9, "bold")
        c.create_text(W / 2, 14, text="h(t) moves a share z of the way", font=bold, fill="#34495e")
        c.create_text(W / 2, 29, text="from h(t-1) towards h~", font=bold, fill="#34495e")
        c.create_line(L, y, R, y, fill="#9aa5b1", width=2)
        for u in (-1, 0, 1):
            c.create_line(px(u), y - 4, px(u), y + 4, fill="#9aa5b1")
            c.create_text(px(u), y + 38, text=f"{u:+d}" if u else "0", font=small, fill="#666")
        c.create_line(px(h), y, px(cand), y, fill="#f5cba7", width=6)
        if abs(px(hn) - px(h)) > 3:
            c.create_line(px(h), y - 12, px(hn), y - 12, fill="#1e8449", width=2, arrow=tk.LAST)
        for u, col, rad in ((h, "#5b7083", 5), (cand, "#d35400", 5), (hn, "#1e8449", 7)):
            c.create_oval(px(u) - rad, y - rad, px(u) + rad, y + rad, fill=col, outline="white", width=2)
        c.create_text(lx(h), y - 26, text="h(t-1)", font=small, fill="#5b7083")
        c.create_text(lx(hn), y - 42, text="h(t)", font=bold, fill="#1e8449")
        c.create_text(lx(cand), y + 20, text="h~", font=bold, fill="#d35400")
        c.create_text(W / 2, H - 12, text=f"z = {z:.2f}  →  {100 * z:.0f}% of the way",
                      font=bold, fill="#1e8449")

    @staticmethod
    def plot_bars(c, W, H, title, rows):
        """Horizontal bars: rng == 0 means a gate (0..1), otherwise a signed value in -rng..+rng."""
        small, bold = ("TkDefaultFont", 7), ("TkDefaultFont", 9, "bold")
        c.create_text(W / 2, 14, text=title, font=bold, fill="#34495e")
        L, R = 56, W - 62
        row_h = min(40, (H - 34) / len(rows))
        y0 = 30 + ((H - 34) - row_h * len(rows)) / 2
        for i, (name, val, rng) in enumerate(rows):
            y = y0 + (i + 0.4) * row_h
            bh = min(9, row_h * 0.25)
            c.create_text(L - 8, y, text=name, anchor="e", font=bold, fill="#34495e")
            c.create_rectangle(L, y - bh, R, y + bh, fill="#eef1f4", outline="")
            if rng == 0:
                x0, x1, col, txt, ends = L, L + val * (R - L), gate_color(val), f"{val:.3f}", ("0", "1")
            else:
                x0 = (L + R) / 2
                x1 = x0 + float(np.clip(val / rng, -1, 1)) * (R - L) / 2
                col, txt, ends = signed_color(val / rng), f"{val:+.3f}", (f"-{rng:g}", f"+{rng:g}")
            c.create_rectangle(min(x0, x1), y - bh, max(x0, x1), y + bh, fill=col, outline="")
            if rng:
                c.create_line(x0, y - bh - 3, x0, y + bh + 3, fill="#9aa5b1")
            c.create_text(R + 6, y, text=txt, anchor="w", font=bold, fill="#222")
            c.create_text(L, y + bh + 6, text=ends[0], anchor="w", font=small, fill="#888")
            c.create_text(R, y + bh + 6, text=ends[1], anchor="e", font=small, fill="#888")

    # ================================================================ TAB 2
    def build_cell_tab(self):
        pw = ttk.PanedWindow(self.tab_cell, orient="horizontal")
        pw.pack(fill="both", expand=True)
        sf = self.make_scrollframe(pw)
        left = sf.inner
        right = ttk.Frame(pw, padding=(4, 10, 10, 10))
        pw.add(sf, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Inputs for this step", style="Head.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w")
        slider_row(left, 1, "x(t)  input", self.x_var, -2, 2, length=150)
        slider_row(left, 2, "h(t-1)  memory", self.hprev_var, -1, 1, length=150)

        self.cell_preset = tk.StringVar(value="Balanced (default)")
        r = self.build_param_panel(left, 3, self.cell_preset)
        self.add_tip_label(left, r)

        # diagram on top, worked numbers below - drag the divider between them
        vp = ttk.PanedWindow(right, orient="vertical")
        vp.pack(fill="both", expand=True)
        top = ttk.Frame(vp)
        bottom = ttk.Frame(vp, padding=(0, 6, 0, 0))
        vp.add(top, weight=3)
        vp.add(bottom, weight=1)

        head = ttk.Label(top, text="Live GRU cell diagram  -  wire thickness = signal size, "
                                   "colour = value", style="Head.TLabel")
        head.pack(anchor="w", fill="x")
        auto_wrap(head)
        self._cell_v = None
        self.cell_canvas = tk.Canvas(top, width=560, height=310, bg="white",
                                     highlightthickness=1, highlightbackground="#d0d7de")
        self.cell_canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.cell_canvas.bind("<Configure>", lambda e: self.schedule("celldraw", self.redraw_cell, 15))

        self.cell_text = tk.Text(bottom, height=9, width=40, font=("Courier", 10), wrap="word",
                                 relief="flat", bg="#f6f8fa", padx=10, pady=8)
        tsb = ttk.Scrollbar(bottom, command=self.cell_text.yview)
        self.cell_text.configure(yscrollcommand=autohide(tsb))
        self.cell_text.grid(row=0, column=0, sticky="nsew")
        tsb.grid(row=0, column=1, sticky="ns")
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1)
        self.cell_text.tag_configure("b", font=("Courier", 10, "bold"))
        self.cell_text.tag_configure("g", foreground="#1e8449", font=("Courier", 10, "bold"))

    def redraw_cell(self):
        if self._cell_v is not None:
            self.draw_cell(self.cell_canvas, self._cell_v)

    def update_cell(self):
        if not hasattr(self, "cell_canvas"):
            return
        v = gru_step_scalar(self.x_var.get(), self.hprev_var.get(), self.params())
        self._cell_v = v
        self.draw_cell(self.cell_canvas, v)
        p = self.params()
        t = self.cell_text
        t.configure(state="normal")
        t.delete("1.0", "end")

        def line(label, s, tag=None):
            t.insert("end", label, ("b",))
            t.insert("end", s + "\n", (tag,) if tag else ())
        x, h = v["x"], v["h"]
        line("Step 1  Reset gate:   ",
             f"r  = \u03c3(Wr\u00b7x + Ur\u00b7h + br) = \u03c3({p['Wr']:+.2f}\u00b7{x:+.2f} "
             f"{p['Ur']:+.2f}\u00b7{h:+.2f} {p['br']:+.2f}) = \u03c3({v['a_r']:+.3f}) = {v['r']:.3f}")
        line("Step 2  Update gate:  ",
             f"z  = \u03c3(Wz\u00b7x + Uz\u00b7h + bz) = \u03c3({p['Wz']:+.2f}\u00b7{x:+.2f} "
             f"{p['Uz']:+.2f}\u00b7{h:+.2f} {p['bz']:+.2f}) = \u03c3({v['a_z']:+.3f}) = {v['z']:.3f}")
        line("Step 3  Candidate:    ",
             f"h~ = tanh(Wh\u00b7x + Uh\u00b7(r\u00b7h) + bh) = tanh({p['Wh']:+.2f}\u00b7{x:+.2f} "
             f"{p['Uh']:+.2f}\u00b7{v['rh']:+.3f} {p['bh']:+.2f}) = tanh({v['a_c']:+.3f}) = {v['cand']:+.3f}")
        line("Step 4  New memory:   ",
             f"h(t) = (1-z)\u00b7h(t-1) + z\u00b7h~ = {1 - v['z']:.3f}\u00b7{h:+.3f} + {v['z']:.3f}\u00b7"
             f"{v['cand']:+.3f} = {v['keep']:+.3f} + {v['write']:+.3f} = ", None)
        t.insert("end", f"{v['h_new']:+.4f}\n\n", ("g",))

        t.insert("end", "What this means in plain words\n", ("b",))
        z, r = v["z"], v["r"]
        if z < 0.2:
            zs = f"The update gate is mostly CLOSED (z={z:.2f}): the GRU keeps {100 * (1 - z):.0f}% of its old memory."
        elif z > 0.8:
            zs = f"The update gate is mostly OPEN (z={z:.2f}): {100 * z:.0f}% of the memory is replaced by the candidate."
        else:
            zs = (f"The update gate is half-open (z={z:.2f}): the new memory is a blend - "
                  f"{100 * (1 - z):.0f}% old, {100 * z:.0f}% new.")
        if r < 0.2:
            rs = f"The reset gate is nearly CLOSED (r={r:.2f}): the candidate is built almost only from the current input."
        elif r > 0.8:
            rs = f"The reset gate is OPEN (r={r:.2f}): the candidate uses the old memory fully."
        else:
            rs = f"The reset gate lets {100 * r:.0f}% of the old memory into the candidate."
        t.insert("end", "\u2022 " + rs + "\n\u2022 " + zs + "\n")
        delta = v["h_new"] - h
        t.insert("end", f"\u2022 Memory changed by {delta:+.3f}  ({h:+.3f} \u2192 {v['h_new']:+.3f}).\n")
        t.configure(state="disabled")

    def draw_cell(self, c, v, step=None):
        """Draw in a fixed 780x430 design space, then scale everything to fit the canvas.
        With `step` set (walkthrough tab), parts computed in later steps are greyed out and
        the part computed in this step is highlighted."""
        c.delete("all")
        W0, H0 = 780, 430
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 40 or ch < 40:                      # not on screen yet
            cw, ch = W0, H0
        s = min(cw / W0, ch / H0)                  # keep the aspect ratio
        ox, oy = (cw - W0 * s) / 2, (ch - H0 * s) / 2

        def font(size, style="normal"):
            return ("TkDefaultFont", max(6, int(round(size * s))), style)
        TOP, HBUS, XBUS = 70, 330, 380
        GR, GZ, GC = 170, 300, 430        # x-centres of r-box, z-box, candidate box
        BY, BW, BH = 235, 104, 54         # box centre y, width, height
        NR = 14                           # node radius
        FADE, GLOW, NOW = "#dfe3e8", "#ffe08a", "#e67e22"

        def state(g):
            if step is None:
                return "done"
            k = CELL_GROUPS.index(g)
            return "future" if k > step else "now" if k == step else "done"

        def wire(pts, val, g, signed=True, arrow=False, bridge=False):
            st = state(g)
            if st == "future":
                col, w = FADE, 1.5 * s
            else:
                col = signed_color(np.clip(val, -1, 1)) if signed else gate_color(val)
                col = lerp_color("#9aa5b1", col, 0.25 + 0.75 * min(1, abs(val)))
                w = (1.5 + 4.5 * min(1.0, abs(val))) * s
            flat = [coord for p in pts for coord in p]
            if st == "now":
                c.create_line(*flat, width=w + 12 * s, fill=GLOW, capstyle="round", joinstyle="round",
                              tags="glow")
            if bridge:
                c.create_line(*flat, width=w + 8 * s, fill=GLOW if st == "now" else "white")
            c.create_line(*flat, width=w, fill=col, arrow=tk.LAST if arrow else None,
                          arrowshape=(10 * s, 12 * s, 5 * s), capstyle="round", joinstyle="round")

        def node(x, y, sym, g):
            st = state(g)
            ink = "#c3c9d0" if st == "future" else "#34495e"
            if st == "now":
                c.create_oval(x - NR - 7, y - NR - 7, x + NR + 7, y + NR + 7, fill=GLOW, outline="",
                              tags="glow")
            c.create_oval(x - NR, y - NR, x + NR, y + NR, fill="white", outline=ink, width=2 * s)
            c.create_text(x, y, text=sym, font=font(12, "bold"), fill=ink)

        def box(cx, title, func, val, g, gate=True):
            st = state(g)
            x0, y0, x1, y1 = cx - BW / 2, BY - BH / 2, cx + BW / 2, BY + BH / 2
            if st == "future":
                c.create_rectangle(x0, y0, x1, y1, fill="#f4f5f7", outline="#c3c9d0", width=2 * s,
                                   dash=(4, 3))
                c.create_text(cx, BY - 11, text=title, fill="#9aa5b1", font=font(10, "bold"))
                c.create_text(cx, BY + 11, text=f"{func} → ?", fill="#9aa5b1", font=font(10))
                return
            if st == "now":
                c.create_rectangle(x0 - 8, y0 - 8, x1 + 8, y1 + 8, fill=GLOW, outline="", tags="glow")
            fill = gate_color(val) if gate else signed_color(val)
            c.create_rectangle(x0, y0, x1, y1, fill=fill, outline=NOW if st == "now" else "#34495e",
                               width=(3 if st == "now" else 2) * s)
            fg = text_on(fill)
            c.create_text(cx, BY - 11, text=title, fill=fg, font=font(10, "bold"))
            c.create_text(cx, BY + 11, text=f"{func} → {val:+.3f}" if not gate else f"{func} → {val:.3f}",
                          fill=fg, font=font(10))

        def label(x, y, s, g=None, color="#222", anchor="center", bold=False):
            if g is not None and state(g) == "future":
                return
            c.create_text(x, y, text=s, fill=color, anchor=anchor,
                          font=font(10, "bold" if bold else "normal"))

        def badge(x, y, num, g):          # step number, walkthrough only
            if step is None:
                return
            fill = {"now": NOW, "done": "#34495e", "future": "#c3c9d0"}[state(g)]
            c.create_oval(x - 11, y - 11, x + 11, y + 11, fill=fill, outline="white", width=2 * s)
            c.create_text(x, y, text=num, fill="white", font=font(10, "bold"))

        h, x = v["h"], v["x"]
        # memory highway (top)
        wire([(20, TOP), (480 - NR, TOP)], h, "in", arrow=True)
        wire([(480 + NR, TOP), (620 - NR, TOP)], v["keep"], "out", arrow=True)
        wire([(620 + NR, TOP), (755, TOP)], v["h_new"], "out", arrow=True)
        label(18, TOP - 20, f"h(t-1) = {h:+.3f}", "in", anchor="w", bold=True)
        label(755, TOP - 20, f"h(t) = {v['h_new']:+.3f}", "out", anchor="e", bold=True, color="#1e8449")
        label(480, TOP - 26, f"keep: (1-z)·h = {v['keep']:+.3f}", "out")
        c.create_text(390, 18, text="memory highway: old state flows straight through when z ≈ 0",
                      fill="#7f8c8d", font=font(9, "italic"))

        # inputs: h_prev bus and x bus
        wire([(80, TOP), (80, HBUS), (GZ - 12, HBUS)], h, "in")
        for gx, g in ((GR, "r"), (GZ, "z")):     # the candidate only sees r*h, never h directly
            wire([(gx - 12, HBUS), (gx - 12, BY + BH / 2)], h, g, arrow=True)
        wire([(20, XBUS), (GC + 12, XBUS)], x, "in")
        for gx, g in ((GR, "r"), (GZ, "z"), (GC, "cand")):
            wire([(gx + 12, XBUS), (gx + 12, BY + BH / 2)], x, g, arrow=True)
        label(18, XBUS + 18, f"x(t) = {x:+.3f}  (input)", "in", anchor="w", bold=True)
        label(92, HBUS - 10, "h(t-1)", "in", anchor="w", color="#555")

        # reset path: r * h_prev
        wire([(GR, TOP), (GR, 140 - NR)], h, "cand", arrow=True)
        wire([(GR, BY - BH / 2), (GR, 140 + NR)], v["r"], "r", signed=False, arrow=True)
        node(GR, 140, "×", "cand")
        label(GR - 22, 140, "r·h", "cand", anchor="e", color="#555")

        # update gate bus (drawn before the r*h wire so the r*h wire bridges over it)
        wire([(GZ, BY - BH / 2), (GZ, 185), (560, 185), (560, BY - NR)], v["z"], "z", signed=False, arrow=True)
        wire([(480, 185), (480, 125 + NR)], v["z"], "out", signed=False, arrow=True)
        node(480, 125, "1-", "out")
        wire([(480, 125 - NR), (480, TOP + NR)], 1 - v["z"], "out", signed=False, arrow=True)
        node(480, TOP, "×", "out")
        label(GZ + 8, 172, f"z = {v['z']:.3f}", "z", anchor="w", color="#1e8449")
        label(492, 104, f"1-z = {1 - v['z']:.3f}", "out", anchor="w", color="#1e8449")

        wire([(GR + NR, 140), (GC, 140), (GC, BY - BH / 2)], v["rh"], "cand", arrow=True, bridge=True)
        label(300, 128, f"r·h(t-1) = {v['rh']:+.3f}", "cand")

        # candidate -> z*h~ -> plus
        wire([(GC + BW / 2, BY), (560 - NR, BY)], v["cand"], "cand", arrow=True)
        node(560, BY, "×", "out")
        wire([(560 + NR, BY), (620, BY), (620, TOP + NR)], v["write"], "out", arrow=True)
        node(620, TOP, "+", "out")
        label(560, BY + 28, f"write: z·h~ = {v['write']:+.3f}", "out")

        box(GR, "reset gate r", "σ", v["r"], "r")
        box(GZ, "update gate z", "σ", v["z"], "z")
        box(GC, "candidate h~", "tanh", v["cand"], "cand", gate=False)

        badge(GR - BW / 2 - 8, BY - BH / 2 - 8, "1", "r")
        badge(GZ - BW / 2 - 8, BY - BH / 2 - 8, "2", "z")
        badge(GC - BW / 2 - 8, BY - BH / 2 - 8, "3", "cand")
        badge(646, TOP + 30, "4", "out")

        # legend
        ly = 412
        label(470, ly, "gate 0", anchor="e", color="#555")
        for i in range(10):
            c.create_rectangle(475 + i * 8, ly - 6, 483 + i * 8, ly + 6, fill=gate_color(i / 9), outline="")
        label(560, ly, "1", anchor="w", color="#555")
        label(610, ly, "-1", anchor="e", color="#555")
        for i in range(11):
            c.create_rectangle(615 + i * 8, ly - 6, 623 + i * 8, ly + 6,
                               fill=signed_color(-1 + 2 * i / 10), outline="")
        label(707, ly, "+1", anchor="w", color="#555")

        c.tag_lower("glow")
        c.scale("all", 0, 0, s, s)
        c.move("all", ox, oy)

    # ================================================================ TAB 3
    def build_seq_tab(self):
        pw = ttk.PanedWindow(self.tab_seq, orient="horizontal")
        pw.pack(fill="both", expand=True)
        sf = self.make_scrollframe(pw)
        left = sf.inner
        right = ttk.Frame(pw, padding=(4, 10, 10, 10))
        pw.add(sf, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Input sequence", style="Head.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        self.sig_var = tk.StringVar(value="Pulse")
        cb = ttk.Combobox(left, textvariable=self.sig_var, values=SIGNALS, state="readonly", width=26)
        cb.grid(row=1, column=0, columnspan=3, sticky="ew")
        route_wheel(cb)
        cb.bind("<<ComboboxSelected>>", lambda e: self.update_seq())
        self.seq_len = tk.IntVar(value=120)
        self.seq_amp = tk.DoubleVar(value=1.0)
        self.seq_h0 = tk.DoubleVar(value=0.0)
        slider_row(left, 2, "Length", self.seq_len, 20, 300, fmt="{:d}", length=150)
        slider_row(left, 3, "Amplitude", self.seq_amp, 0, 2, length=150)
        slider_row(left, 4, "Start h(0)", self.seq_h0, -1, 1, length=150)
        self.show_rnn = tk.BooleanVar(value=True)
        self.show_cand = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Show plain RNN (no gates)", variable=self.show_rnn,
                        command=self.update_seq).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(left, text="Show candidate h~", variable=self.show_cand,
                        command=self.update_seq).grid(row=6, column=0, columnspan=3, sticky="w")
        for v in (self.seq_len, self.seq_amp, self.seq_h0):
            v.trace_add("write", lambda *_: self.schedule("seq", self.update_seq))

        self.seq_preset = tk.StringVar(value="Balanced (default)")
        r = self.build_param_panel(left, 7, self.seq_preset)
        self.add_tip_label(left, r)

        self.seq_fig = make_figure(6, 4.5)
        self.seq_axes = self.seq_fig.subplots(3, 1, sharex=True,
                                              gridspec_kw=dict(height_ratios=[1, 1.4, 1]))
        self.seq_summary = ttk.Label(right, text="", wraplength=600, justify="left")
        self.seq_summary.pack(side="bottom", fill="x", pady=(6, 0))
        auto_wrap(self.seq_summary)
        self.seq_canvas = FigureCanvasTkAgg(self.seq_fig, master=right)
        small_plot_widget(self.seq_canvas)
        self.seq_canvas.mpl_connect("resize_event",
                                    lambda e: self.schedule("seq", self.update_seq, 120))

    def update_seq(self):
        if not hasattr(self, "seq_axes"):
            return
        p = self.params()
        T = int(self.seq_len.get())
        X = make_signal(self.sig_var.get(), T, self.seq_amp.get())
        h = self.seq_h0.get()
        hr = h
        H, Z, R, C, HR = (np.empty(T) for _ in range(5))
        for t in range(T):
            v = gru_step_scalar(X[t], h, p)
            h = v["h_new"]
            H[t], Z[t], R[t], C[t] = h, v["z"], v["r"], v["cand"]
            hr = np.tanh(p["Wh"] * X[t] + p["Uh"] * hr + p["bh"])
            HR[t] = hr
        t = np.arange(1, T + 1)

        f = self.fig_scale(self.seq_fig, 800)
        a1, a2, a3 = self.seq_axes
        for a in self.seq_axes:
            a.clear()
            a.grid(alpha=0.3)
            a.tick_params(labelsize=9 * f)
        a1.plot(t, X, color="#34495e", lw=1.6)
        a1.set_ylabel("input x(t)", fontsize=10 * f)
        a1.set_title("1-unit GRU reading a sequence (same weights as tabs 1 & 2)", fontsize=11 * f)

        a2.plot(t, H, color="#1e8449", lw=2.4, label="GRU memory h(t)")
        if self.show_cand.get():
            a2.plot(t, C, color="#d35400", lw=1, alpha=0.7, label="candidate h~(t)")
        if self.show_rnn.get():
            a2.plot(t, HR, color="#d62728", lw=1.6, ls="--", label="plain RNN h(t)")
        a2.set_ylim(-1.1, 1.1)
        a2.set_ylabel("hidden state", fontsize=10 * f)
        a2.legend(loc="upper right", fontsize=9 * f)

        a3.plot(t, Z, color="#1e8449", lw=1.8, label="update gate z")
        a3.plot(t, R, color="#8e44ad", lw=1.4, label="reset gate r")
        a3.fill_between(t, 0, Z, color="#1e8449", alpha=0.12)
        a3.set_ylim(-0.05, 1.05)
        a3.set_ylabel("gate value", fontsize=10 * f)
        a3.set_xlabel("time step t", fontsize=10 * f)
        a3.legend(loc="upper right", fontsize=9 * f)
        self.seq_canvas.draw_idle()

        mz, mr = Z.mean(), R.mean()
        half = np.log(0.5) / np.log(max(1e-9, 1 - mz)) if mz > 1e-6 else np.inf
        half_s = f"{half:.1f} steps" if np.isfinite(half) and half < 1e6 else "forever"
        self.seq_summary.config(
            text=f"Average update gate z = {mz:.2f}  \u2192 each step keeps about {100 * (1 - mz):.0f}% "
                 f"of the old memory (memory half-life \u2248 {half_s}).   "
                 f"Average reset gate r = {mr:.2f}  \u2192 the candidate sees {100 * mr:.0f}% of the past "
                 f"on average.   Final h = {H[-1]:+.3f}.")

    # ================================================================ TAB 4
    def build_compare_tab(self):
        # plots on top, summary table below - drag the divider between them
        vp = ttk.PanedWindow(self.tab_cmp, orient="vertical")
        vp.pack(fill="both", expand=True)
        top = ttk.PanedWindow(vp, orient="horizontal")
        bottom = ttk.Frame(vp, padding=(10, 4, 10, 10))
        vp.add(top, weight=1)
        vp.add(bottom, weight=0)
        sf = self.make_scrollframe(top)
        left = sf.inner
        right = ttk.Frame(top, padding=(4, 10, 10, 0))
        top.add(sf, weight=0)
        top.add(right, weight=1)

        ttk.Label(left, text="Memory test settings", style="Head.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w")
        self.c_hidden = tk.IntVar(value=32)
        self.c_input = tk.IntVar(value=8)
        self.c_len = tk.IntVar(value=150)
        self.c_scale = tk.DoubleVar(value=1.0)
        self.c_noise = tk.DoubleVar(value=0.5)
        self.c_bias = tk.DoubleVar(value=3.0)
        self.c_trials = tk.IntVar(value=5)
        self.c_seed = tk.IntVar(value=0)
        rows = [("Hidden size", self.c_hidden, 2, 128, "{:d}"),
                ("Input size", self.c_input, 1, 64, "{:d}"),
                ("Sequence length", self.c_len, 10, 300, "{:d}"),
                ("Weight scale", self.c_scale, 0.2, 2.5, "{:.2f}"),
                ("Distractor noise", self.c_noise, 0, 2, "{:.2f}"),
                ("Memory bias", self.c_bias, 0, 5, "{:.2f}"),
                ("Trials (avg.)", self.c_trials, 1, 20, "{:d}"),
                ("Random seed", self.c_seed, 0, 100, "{:d}")]
        for i, (name, var, a, b, fmt) in enumerate(rows, start=1):
            slider_row(left, i, name, var, a, b, fmt=fmt, length=150)
            var.trace_add("write", lambda *_: self.schedule("cmp", self.run_compare, 250))
        self.use_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Log scale", variable=self.use_log,
                        command=self.run_compare).grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(left, text="Run again", command=self.run_compare).grid(
            row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        expl = ("What is measured?\n"
                "A random input arrives at t=0, then only random 'distractor' noise. "
                "We nudge that first input slightly and measure how much the hidden state "
                "still changes at each later step - i.e. how much of the first input the "
                "network still remembers. This is also the size of the gradient that "
                "training would send back to t=0.\n\n"
                "Memory bias sets LSTM's forget-gate bias to +b and GRU's update-gate bias "
                "to \u2212b (both mean 'lean towards keeping memory'). The plain RNN has no gate, "
                "so this slider can't help it.\n\n"
                "Weights are random (untrained), so this shows the architecture's built-in "
                "ability to carry information - the starting point for learning.")
        expl_lbl = ttk.Label(left, text=expl, wraplength=290, justify="left", style="Tip.TLabel")
        expl_lbl.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        auto_wrap(expl_lbl)

        self.cmp_fig = make_figure(6, 3.6)
        gs = self.cmp_fig.add_gridspec(1, 3)
        self.cmp_ax1 = self.cmp_fig.add_subplot(gs[0, :2])
        self.cmp_ax2 = self.cmp_fig.add_subplot(gs[0, 2])
        self.cmp_summary = ttk.Label(right, text="", wraplength=600, justify="left")
        self.cmp_summary.pack(side="bottom", fill="x", pady=(4, 0))
        auto_wrap(self.cmp_summary)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, master=right)
        small_plot_widget(self.cmp_canvas, 520, 280)
        self.cmp_canvas.mpl_connect("resize_event",
                                    lambda e: self.schedule("cmpdraw", self.draw_compare, 120))

        cols = ("aspect", "rnn", "lstm", "gru")
        tv = ttk.Treeview(bottom, columns=cols, show="headings", height=5)
        base_w = (190, 250, 330, 330)
        for col, title, w in zip(cols, ("Aspect", "Vanilla RNN", "LSTM (1997)", "GRU (2014)"), base_w):
            tv.heading(col, text=title)
            tv.column(col, width=w, minwidth=60, stretch=True, anchor="w")

        def fit_columns(e):                         # share the width out in the same proportions
            avail = max(240, e.width - 4)
            for col, w in zip(cols, base_w):
                tv.column(col, width=int(avail * w / sum(base_w)))
        tv.bind("<Configure>", fit_columns)
        tsb = ttk.Scrollbar(bottom, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=autohide(tsb))
        data = [
            ("Gates", "none", "forget, input, output (3)", "update, reset (2)"),
            ("State carried between steps", "h", "h + separate cell c", "h only"),
            ("Weight blocks (params)", "1 \u00d7 n_h(n_in+n_h+1)", "4 \u00d7 n_h(n_in+n_h+1)",
             "3 \u00d7 n_h(n_in+n_h+1)  \u2192 25% fewer than LSTM"),
            ("How memory is updated", "fully rewritten every step", "c = f\u00b7c + i\u00b7g  (keep & write independent)",
             "h = (1-z)\u00b7h + z\u00b7h~  (keep & write tied)"),
            ("Long-range memory", "poor (vanishing gradient)", "very good", "very good"),
            ("Speed / memory use", "fastest, smallest", "slowest, largest", "in between, simpler code"),
            ("Typical use", "short sequences, teaching", "long or complex sequences",
             "same tasks, smaller data, edge devices"),
        ]
        for row in data:
            tv.insert("", "end", values=row)
        tv.grid(row=0, column=0, sticky="nsew")
        tsb.grid(row=0, column=1, sticky="ns")
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1)

    def run_compare(self):
        if not hasattr(self, "cmp_ax1"):
            return
        try:
            n_h = max(1, int(self.c_hidden.get()))
            n_in = max(1, int(self.c_input.get()))
            T = max(2, int(self.c_len.get()))
            scale = float(self.c_scale.get())
            noise = float(self.c_noise.get())
            bias = float(self.c_bias.get())
            trials = max(1, int(self.c_trials.get()))
            seed = int(self.c_seed.get())
        except (tk.TclError, ValueError):
            return

        curves = {cls.name: [] for cls in MODEL_CLASSES}
        for k in range(trials):
            rng_data = np.random.default_rng(1000 * seed + k)
            X = noise * rng_data.normal(size=(T, n_in))
            v_dir = rng_data.normal(size=n_in)
            v_dir /= np.linalg.norm(v_dir)
            X[0] = v_dir
            for cls in MODEL_CLASSES:
                model = cls(n_in, n_h, scale, np.random.default_rng(10_000 * seed + 31 * k + 7), bias)
                m = memory_curve(model, X, v_dir)
                curves[cls.name].append(np.maximum(m / max(m[0], 1e-12), 1e-12))

        self._cmp_data = dict(curves=curves, T=T, trials=trials, n_in=n_in, n_h=n_h)
        self.draw_compare()

    def draw_compare(self):
        d = getattr(self, "_cmp_data", None)
        if d is None:
            return
        curves, T, trials, n_in, n_h = d["curves"], d["T"], d["trials"], d["n_in"], d["n_h"]
        f = self.fig_scale(self.cmp_fig, 900)

        ax = self.cmp_ax1
        ax.clear()
        t = np.arange(T)
        summary = []
        top_val = 1.0
        for cls in MODEL_CLASSES:
            arr = np.array(curves[cls.name])
            mean = np.exp(np.log(arr).mean(axis=0))   # geometric mean over trials
            ax.plot(t, mean, color=cls.color, lw=2.2, label=cls.name)
            if trials > 1:
                ax.fill_between(t, arr.min(axis=0), arr.max(axis=0), color=cls.color, alpha=0.10)
            top_val = max(top_val, float(arr.max()))
            below = np.where(mean < 0.01)[0]
            if len(below):
                summary.append(f"{cls.name}: drops below 1% after {below[0]} steps")
            else:
                summary.append(f"{cls.name}: still above 1% after {T - 1} steps (end: {100 * mean[-1]:.3g}%)")
        if self.use_log.get():
            ax.set_yscale("log")
            ax.set_ylim(max(1e-10, ax.get_ylim()[0]), 2 * top_val)
        else:
            ax.set_ylim(0, 1.1 * top_val)
        ax.axhline(0.01, color="#888", ls=":", lw=1)
        ax.text(T - 1, 0.012, "1% line", ha="right", va="bottom", fontsize=8 * f, color="#666")
        ax.set_xlabel("time steps after the remembered input", fontsize=10 * f)
        ax.set_ylabel("memory of input at t=0 (relative)", fontsize=10 * f)
        ax.set_title("How long does each network remember?", fontsize=11 * f)
        ax.tick_params(labelsize=9 * f)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower left", fontsize=9 * f)

        ax2 = self.cmp_ax2
        ax2.clear()
        names = [cls.name for cls in MODEL_CLASSES]
        counts = [cls.n_params(n_in, n_h) for cls in MODEL_CLASSES]
        bars = ax2.bar(["RNN", "LSTM", "GRU"], counts, color=[cls.color for cls in MODEL_CLASSES])
        for b_, n in zip(bars, counts):
            ax2.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{n:,}",
                     ha="center", va="bottom", fontsize=9 * f)
        ax2.set_title(f"Parameters\n(in={n_in}, hidden={n_h})", fontsize=11 * f)
        ax2.set_ylim(0, max(counts) * 1.18)
        ax2.tick_params(labelsize=9 * f)
        ax2.grid(alpha=0.3, axis="y")
        self.cmp_canvas.draw_idle()

        saving = 100 * (1 - counts[2] / counts[1])
        self.cmp_summary.config(
            text="   \u2022   ".join(summary) +
                 f"\nGRU uses {saving:.0f}% fewer parameters than LSTM ({names[2]} {counts[2]:,} vs "
                 f"{names[1]} {counts[1]:,}) and {counts[2] / counts[0]:.0f}\u00d7 more than a plain RNN.")

if __name__ == "__main__":
    app = GRUExplorer()
    app.mainloop()