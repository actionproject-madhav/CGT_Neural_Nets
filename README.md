# CGT-Informed Neural Networks: Pilot Study

**Alternative Neural Nets for Navigating Games, Puzzles, and the Physical World**

This project tests whether encoding combinatorial game theory (CGT) structure into neural network architectures improves generalization on compositional games, compared to standard approaches.

## Pilot Study: Multi-Heap Nim

The pilot uses multi-heap Nim as a testbed because it has:
- **Perfect ground truth** via the Sprague-Grundy theorem (XOR of heap sizes)
- **Natural compositional structure** (each heap is an independent subgame)
- **Scalable complexity** for testing generalization

### Architectures Compared

| Model | Type | Key Idea |
|-------|------|----------|
| **MLP** | Supervised | Fixed-size feedforward baseline |
| **DeepSets** | Supervised | Permutation-invariant set processing |
| **CGT-Net** | Supervised | Subgame decomposition + Grundy value learning |
| **DQN** | RL | Standard deep Q-learning via self-play |

### Core Hypothesis

The CGT-Decomposition Network should generalize to unseen game configurations (more heaps, larger heaps) better than standard architectures, because its architecture mirrors the game's compositional structure.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pilot study (~2-3 hours on CPU)
python -m experiments.run_pilot

# Quick test run (2 seeds, fewer DQN episodes)
python -m experiments.run_pilot --seeds 42 123 --dqn-episodes 5000

# Generate plots after experiment completes
python -m analysis.plot_results

# Generate the Markdown report
python -m analysis.pilot_report
```

## Project Structure

```
CGT_Neural_Nets/
├── games/nim.py              # Nim engine + CGT computations
├── data/generator.py         # Dataset generation (6 splits)
├── models/
│   ├── mlp_baseline.py       # Architecture 1: standard MLP
│   ├── deepsets_baseline.py  # Architecture 2: DeepSets
│   ├── cgt_net.py            # Architecture 3: CGT-Decomposition Net
│   └── dqn_baseline.py       # Architecture 4: DQN
├── training/
│   ├── trainer.py            # Supervised training loop
│   └── rl_trainer.py         # DQN self-play trainer
├── evaluation/evaluator.py   # Metrics + comparison
├── experiments/
│   ├── configs/pilot_config.yaml
│   └── run_pilot.py          # Master experiment script
├── analysis/
│   ├── plot_results.py       # Visualization
│   └── pilot_report.py       # Auto-generate report
└── results/                  # Output directory
```
