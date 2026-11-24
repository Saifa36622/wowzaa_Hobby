# LNDP: Lifelong Neural Developmental Programs

Rebuild of *Evolving Self-Assembling Neural Networks: From Spontaneous Activity to Experience-Dependent Learning* ([arXiv:2406.09787](https://arxiv.org/abs/2406.09787), [code](https://github.com/erwanplantec/LNDP)), compared with the earlier NDP (Najarro, Sudhakaran & Risi, 2023).

Main idea: an **NDP** grows a network before life and then freezes it. An **LNDP** keeps running its rules during life (changing weights with reward, growing and pruning connections), and uses spontaneous activity before life.

| File | What it does |
|---|---|
| `NDP.py` | Minimal NDP in one file: rules make cells talk, divide and wire up; the grown network is frozen and balances a CartPole. Evolves the rules with CMA-ES, then shows the growth stages and fitness. |
| `test.py` | Main LNDP paper rebuild on the non-stationary foraging task (food switches sides of a corridor). Evolves 3 versions side by side: LNDP, LNDP without structural changes, and frozen NDP. Shows the LNDP network live, with connections growing and pruning. |
| `compare.py` | Normal NDP vs. LNDP, during evolution and during one life. Includes an "after a switch" test showing that only the LNDP re-learns where the food is. Animates NDP growth and LNDP lifelong changes. |

Run: `python <file>.py` (most accept `--save shot.png --gens N` for a headless screenshot).
