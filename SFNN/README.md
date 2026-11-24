# SFNN: Structurally Flexible Neural Networks

Rebuild of *Structurally Flexible Neural Networks: Evolving the Building Blocks for General Agents* ([arXiv:2404.15193](https://arxiv.org/abs/2404.15193), [code](https://github.com/Joachm/SFNN)).

Main idea: evolve the **building blocks** (a few neuron types and synapse learning rules), not the weights. The same blocks are dropped into randomly wired networks of any input/output size, which then learn during life.

| File | What it does |
|---|---|
| `test.py` | Main paper rebuild. Three small tasks with different input/output sizes, evolved together with CMA-ES. Compares 4 versions (SFNN, one type, fully connected, fixed wiring) and reproduces the paper's figure views: evolution (Fig. 4), watch a lifetime, robustness (Fig. 6) and same start (Fig. 5). Uses a simplified 10-number learning rule instead of GRUs. |
| `evole.py` | The full SFNN training loop on CartPole with the paper's 565-number genome (3 neuron types + 3 GRU synapse rules + 1 learning rate). Shows evolution, learning within a life, the genome, and how the weights move across 8 episodes. Uses PyTorch. |
| `step.py` | Step-by-step teaching app: 15 pages that walk through every equation of one SFNN time step on CartPole with real numbers, from the genome to the CMA-ES update. |
| `cma.py` | What one CMA-ES generation changes in each building block, and how that changes the robot. Can swap the score for random or reversed scores to show that CMA-ES only follows the score. |
| `in_out.py` | Normal network vs. SFNN when the number of inputs/actions changes: the normal network needs new weights, SFNN reuses the same genome. Presets for CartPole, Acrobot, MountainCar and a big robot. |
| `conclude.txt` | Short personal note summarizing the paper's claim: the evolved blocks let the network wire itself for every task shown, and "combined" means the 3 tasks are trained in the same batch. |

Run: `python <file>.py` (`evole.py` also needs `pip install torch`).
