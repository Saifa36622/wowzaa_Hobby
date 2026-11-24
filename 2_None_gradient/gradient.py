import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3];  y = [3, 5, 7]       # known answers
w, b, lr = 0.0, 0.0, 0.1            # unknown knobs + step size

history = []                        # (step, w, b, loss) for plotting

for step in range(200):
    pred = [w*xi + b for xi in x]                       # 1. predict
    err  = [p - yi for p, yi in zip(pred, y)]           # 2. compare
    loss = sum(e*e for e in err) / 3                    #    mean squared error
    history.append((step, w, b, loss))
    grad_w = 2 * sum(e*xi for e, xi in zip(err, x)) / 3 # 3. gradient for w
    grad_b = 2 * sum(err) / 3                           #    gradient for b
    w -= lr * grad_w                                    # 4. step downhill
    b -= lr * grad_b

print(w, b)   # ≈ 2.0, 1.0

# ---- interactive graphs ----
from matplotlib.widgets import Slider, Button

steps, ws, bs, losses = zip(*history)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.28, wspace=0.25)

# 1. loss going down over time
ax1.set_yscale("log")
ax1.set_xlim(0, len(steps)); ax1.set_ylim(min(losses)/2, max(losses)*2)
ax1.set_title("Loss over steps")
ax1.set_xlabel("step"); ax1.set_ylabel("MSE (log scale)")
loss_line, = ax1.plot([], [])
loss_dot,  = ax1.plot([], [], "ro")
info1 = ax1.text(0.98, 0.96, "", transform=ax1.transAxes, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.8))

# 2. the line getting closer to the data
xs = np.linspace(0, 4, 50)
ax2.scatter(x, y, color="black", zorder=3, label="data")
ax2.set_xlim(0, 4); ax2.set_ylim(-1, 10)
ax2.set_title("Fitted line")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
fit_line, = ax2.plot([], [], "r-", label="current line")
ax2.legend(loc="upper left")
info2 = ax2.text(0.98, 0.04, "", transform=ax2.transAxes, ha="right", bbox=dict(fc="white", ec="none", alpha=0.8))

# 3. path of (w, b) across the loss surface
W, B = np.meshgrid(np.linspace(-0.5, 3.5, 200), np.linspace(-1, 3, 200))
L = sum((W*xi + B - yi)**2 for xi, yi in zip(x, y)) / 3
ax3.contour(W, B, L, levels=np.logspace(-2, 2, 15), cmap="viridis")
ax3.plot(2, 1, "k*", markersize=12, label="true (2, 1)")
path_line, = ax3.plot([], [], "r.-", markersize=3, label="gradient descent path")
path_dot,  = ax3.plot([], [], "ro")
ax3.set_title("Path on loss surface")
ax3.set_xlabel("w"); ax3.set_ylabel("b"); ax3.legend()
info3 = ax3.text(0.98, 0.04, "", transform=ax3.transAxes, ha="right", bbox=dict(fc="white", ec="none", alpha=0.8))

def update(s):
    loss_line.set_data(steps[:s+1], losses[:s+1])
    loss_dot.set_data([s], [losses[s]])
    info1.set_text(f"step {s}   loss={losses[s]:.2e}")
    fit_line.set_data(xs, ws[s]*xs + bs[s])
    info2.set_text(f"step {s}   y = {ws[s]:.3f}x + {bs[s]:.3f}")
    path_line.set_data(ws[:s+1], bs[:s+1])
    path_dot.set_data([ws[s]], [bs[s]])
    info3.set_text(f"step {s}   (w, b) = ({ws[s]:.3f}, {bs[s]:.3f})")
    fig.canvas.draw_idle()

# ---- controls: slider + buttons ----
last = len(steps) - 1
slider = Slider(fig.add_axes([0.10, 0.12, 0.80, 0.04]), "step", 0, last,
                valinit=0, valstep=1)
slider.on_changed(lambda val: update(int(val)))

def go_to(s):
    slider.set_val(min(max(s, 0), last))   # triggers update()

btn_start = Button(fig.add_axes([0.30, 0.03, 0.08, 0.05]), "|< Start")
btn_prev  = Button(fig.add_axes([0.39, 0.03, 0.08, 0.05]), "< Prev")
btn_play  = Button(fig.add_axes([0.48, 0.03, 0.08, 0.05]), "Play")
btn_next  = Button(fig.add_axes([0.57, 0.03, 0.08, 0.05]), "Next >")
btn_end   = Button(fig.add_axes([0.66, 0.03, 0.08, 0.05]), "End >|")

btn_start.on_clicked(lambda e: go_to(0))
btn_prev.on_clicked(lambda e: go_to(int(slider.val) - 1))
btn_next.on_clicked(lambda e: go_to(int(slider.val) + 1))
btn_end.on_clicked(lambda e: go_to(last))

# play/pause: a timer advances the slider one step at a time
timer = fig.canvas.new_timer(interval=50)
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

update(0)
plt.show()
