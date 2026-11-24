# 2_None_gradient: gradient vs. gradient-free optimization

Background for the paper folders, which all evolve their genomes with CMA-ES instead of training weights with gradients.

| File | What it does |
|---|---|
| `gradient.py` | Gradient descent from scratch on a tiny line fit (y = 2x + 1). Interactive plots of the loss, the fitted line and the path on the loss surface. |
| `compare.py` | Gradient descent vs. CEM vs. CMA-ES on a bumpy problem (fit a vibrating robot joint, `A·sin(wt + phi)`), over 20 random starts with an equal compute budget. Shows that gradient descent can get stuck in local minima. Saves `comparison.png` and `internals.png`. |
| `cmaes.py` | CMA-ES on the same problem as `compare.py`, one iteration at a time: candidates, elites, the search ellipse, the step size sigma and the covariance matrix C. |
| `cem_cma.py` | CEM and CMA-ES side by side on a narrow tilted valley. Shows CEM's search shrinking too fast while CMA-ES learns the valley's tilt. Buttons for next generation / run / reset. |
| `gradient_scale.py` | Why neural networks use gradients: the same smooth problem with 3, 10, 100 and 1000 knobs on the same budget. Gradient-free methods slow down as knobs increase. Saves `why_gradients_scale.png`. |

Run any of them with `python <file>.py`.
