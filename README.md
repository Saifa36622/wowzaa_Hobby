# Nasree

Small, interactive Python demos (mostly numpy + matplotlib) for understanding networks that **grow, wire and learn by themselves**, plus the background ideas those papers rely on (gradient-free optimization, Hebbian plasticity, GRUs, spontaneous activity).

Each folder has its own README that explains its files.

## Folders

| Folder | What it is for |
|---|---|
| `2_None_gradient/` | Background: gradient descent vs. gradient-free optimizers (CEM, CMA-ES), and why gradients win when there are many weights. CMA-ES is the optimizer used in all the paper folders. |
| `3_heb/` | Background: a frozen vs. a plastic (reward-modulated Hebbian) network, to show learning during life without gradients. |
| `6_Spontaneous activity/` | Background for LNDP: how "practice before birth" (spontaneous activity) can pre-wire a network, and when empty / zero / random starting weights help or hurt. |
| `GRU/` | Background for SFNN: a GUI explorer for Gated Recurrent Units (SFNN's synapses are small GRUs). |
| `SFNN/` | Rebuild of **Structurally Flexible Neural Networks** (paper 1). |
| `GRWON/` | Rebuild of **Growing Artificial Neural Networks for Control: the Role of Neuronal Diversity** (paper 2). |
| `LNDP/` | Rebuild of **Lifelong Neural Developmental Programs / Evolving Self-Assembling Neural Networks** (paper 3), compared with the original NDP. |
| `temp/` | Spare copies of the result images below. |

Images at the top level: `comparison.png`, `internals.png` (from `2_None_gradient/compare.py`) and `why_gradients_scale.png` (from `2_None_gradient/gradient_scale.py`).

## Reference papers

- **SFNN**: *Structurally Flexible Neural Networks: Evolving the Building Blocks for General Agents*: [arXiv:2404.15193](https://arxiv.org/abs/2404.15193), code: [Joachm/SFNN](https://github.com/Joachm/SFNN)
- **GRWON**: *Growing Artificial Neural Networks for Control: the Role of Neuronal Diversity*: [arXiv:2405.08510](https://arxiv.org/abs/2405.08510)
- **LNDP**: *Evolving Self-Assembling Neural Networks: From Spontaneous Activity to Experience-Dependent Learning*: [arXiv:2406.09787](https://arxiv.org/abs/2406.09787), code: [erwanplantec/LNDP](https://github.com/erwanplantec/LNDP)

All rebuilds are simplified so they run live on a laptop; each script's docstring lists what differs from the paper.

## Requirements

```
pip install numpy matplotlib
pip install torch        # only for SFNN/evole.py and GRWON/test.py
```

Most scripts open an interactive matplotlib window (needs a GUI backend such as TkAgg or QtAgg). Many also accept `--save shot.png` to render a screenshot without a window.
