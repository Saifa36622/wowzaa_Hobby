# GRU: Gated Recurrent Unit explorer

Background for SFNN, where each synapse is a small GRU that updates its own state from pre-synaptic activity, post-synaptic activity and reward.

| File | What it does |
|---|---|
| `test.py` | Tkinter GUI with 4 tabs: **Step by Step** (one GRU update, gate by gate, with your numbers), **GRU Cell** (sliders, gates update live), **GRU Over Time** (feed a signal in, plot z, r, h), **Compare RNN/LSTM/GRU** (memory / vanishing-gradient test and parameter counts). |

Run: `python test.py` (needs Tkinter, which ships with most Python installs).
