# GRWON: Growing networks for control and neuronal diversity

Rebuild of *Growing Artificial Neural Networks for Control: the Role of Neuronal Diversity* ([arXiv:2405.08510](https://arxiv.org/abs/2405.08510)).

Main idea: a Neural Developmental Program (NDP) grows a network from a few seed cells. Without extra help all cells end up alike; the paper's fixes, **intrinsic cell state** and **lateral inhibition**, keep cells diverse so growth becomes trainable.

| File | What it does |
|---|---|
| `test.py` | GPU (PyTorch) simulation on a mini "Reacher" task (move a point to a target). Evolves 5 methods side by side with batched CMA-ES: a direct network (all weights evolved) and NDP with no fixes / + inhibition / + intrinsic state / + both. Main window: evolution, reaching paths, growth replay, cell diversity. Comparison window: all methods side by side with a summary table. |

Run: `python test.py` (uses CUDA if available, otherwise CPU). Options: `--pop`, `--episodes`, `--nmax 32`, `--save shot.png --gens 300`. Needs `torch`.
