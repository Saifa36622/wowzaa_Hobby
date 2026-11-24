# 6_Spontaneous activity: practice before birth

Background for the LNDP paper, where a network runs **spontaneous activity** (noise on its inputs) before it ever sees the task, so its local rules can pre-wire it.

The robot in all three scripts is a light seeker with 8 shuffled light sensors, 2 hidden neurons and 2 outputs (turn left / turn right).

| File | What it does |
|---|---|
| `compare.py` | Main demo. Two agents with the same birth wiring: one practises on correlated noise (Ornstein-Uhlenbeck) before birth, one does not. Shows how practice lets a "fire together, wire together" rule pre-wire the sensors to the right hidden neurons. Sliders for the noise settings and practice length, plus "Compare 100 lifetimes". The other two scripts import from this file. |
| `empty_vs_random.py` | Both agents practise; only the input→hidden wiring at birth differs: **random** vs. **empty** (all zeros or all equal). Shows that all-zero wiring never breaks the tie between the two hidden neurons, so the robot stays at guessing. |
| `zero_init_wins.py` | The opposite case: when starting from **zero** is better than random. (1) Overparameterized linear regression: zero start finds the minimum-norm solution and generalizes better. (2) The robot's output layer: zero start begins unbiased at 50/50. |

Run from inside this folder (the last two need `compare.py` next to them): `python compare.py`
