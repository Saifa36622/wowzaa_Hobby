# 3_heb: Hebbian plasticity (learning during life)

Background for the paper folders, where synapses keep changing during an agent's life with local rules instead of gradients.

| File | What it does |
|---|---|
| `hebbian.py` | Two tiny agents choose LEFT or RIGHT to find food. The **frozen** agent never changes its weights; the **plastic** agent uses a reward-modulated local rule (`Δw = eta · (reward − 0.5)` on the connection it just used). Press "Move food" and watch the plastic agent re-learn while the frozen one keeps failing. Sliders for the learning rate and speed. |

Run: `python hebbian.py` (add `--seed 3` for a repeatable run, `--save out.png` for a screenshot).
