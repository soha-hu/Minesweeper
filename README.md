# Minesweeper
# Beating the Logic Bot: A Neural Minesweeper Agent That Learns to Reason

Minesweeper isn’t about reflexes or pattern matching.  
It’s about **reasoning under uncertainty**, where every move changes what you know next.

This project explores a question that fascinated me from the start:

> Can a neural network outperform a deterministic logic-based Minesweeper bot using only the information available to a human player?

After multiple architectures, failed experiments, and redesigns, the answer was **yes**.

---

## Why This Challenge Is Unique

Most deep learning problems come with:
- labeled datasets,
- fixed inputs and outputs,
- and static predictions.

Minesweeper offers **none of that**.

- There is no dataset.
- The agent must **generate its own training data**.
- Each action changes the future state distribution.
- Many boards are **unwinnable without guessing**, so success must be measured carefully.

The baseline I competed against was a **hand-engineered logic bot** (similar to Google’s built-in Minesweeper bot), which applies exact inference rules to deduce safe and mined cells.

My goal was not to imitate it —  
**my goal was to outperform it.**

---

## Core Idea

Instead of hard-coding logical rules, I trained neural networks to:
- infer mine probabilities from partial board states,
- learn long-range spatial dependencies,
- and prioritize survival over greedy clearing.

All models see **only what a human player would see** — no hidden mine locations, no symbolic logic, no cheating.

---

## Task 1: Predicting Mines Directly (The Breakthrough)

### Board Representation
Each game state is encoded as a **3-channel tensor (3 × 22 × 22)**:
- **Channel 0:** Revealed mask (what information is known)
- **Channel 1:** Normalized clue values (0–8 → 0–1)
- **Channel 2:** Unrevealed mask (what must be predicted)

### Output
The network outputs a **22 × 22 probability map**, where each value represents the likelihood that a cell contains a mine.  
At inference time, the agent opens the **unrevealed cell with the lowest predicted mine probability**.

---

### Model Architecture
A simple CNN was insufficient — a mine in one location can affect clues far across the board.

I used a **ResNet with dilated convolutions** to capture long-range dependencies:
- 4 residual blocks (8 caused severe overfitting)
- dilated convolutions to expand receptive field without exploding parameters
- batch normalization for training stability
- dropout (30%) and weight decay for regularization

Final model size: **~1.5M parameters**, deliberately constrained to avoid memorization.

---

### Data Generation (The Most Important Part)
Training on random boards failed badly.

The key insight:
> **Good data matters more than a bigger model.**

I generated training data by letting the **logic bot play full games**, and before every move:
- saved the visible board state,
- stored the ground-truth mine layout (for supervision),
- recorded which cells remained unrevealed.

This produced **120k–180k high-quality training samples** from ~3,000 games.

---

### Results — Beating the Logic Bot

| Difficulty | Mines | Logic Bot Win % | Neural Bot Win % |
|----------|------|----------------|-----------------|
| Easy     | 50   | 75%            | **96%** |
| Medium   | 80   | 21.5%          | **62.5%** |
| Hard     | 100  | 21.5%          | **62.5%** |

The neural agent **dramatically outperformed** the logic bot on medium and hard boards.

Interestingly, performance plateaued at higher mine counts — suggesting the network learned **robust high-difficulty strategies** that generalized beyond a single setting.

---

## Task 2: Actor–Critic Learning (Bootstrapping Intelligence)

Instead of predicting mines directly, I reframed the problem:

> “Given a board state and a candidate move, how long will the agent survive?”

### Architecture
- **Critic:** Predicts expected survival moves (regression)
- **Actor:** Selects actions based on critic evaluations

### Major Setbacks (and Fixes)
- **Policy gradient collapse:** Actor performance dropped to ~4 moves  
  → Fixed by switching to supervised imitation learning
- **Data degradation:** Actor-generated data reduced performance  
  → Fixed by mixing actor and logic bot data
- **Shape mismatches:** Action tensors misaligned  
  → Fixed via consistent tensor shaping and masking

### Final Outcome
- Logic bot baseline: **73.5 average moves**
- Supervised actor: **54.7 moves**
- Bootstrapped actor: **58.1 moves**

While the actor did not surpass the logic bot, it **demonstrated self-improvement**, learning conservative, low-risk strategies aligned with survival objectives.

---

## Task 3: Learning to “Think Longer”

I explored whether a network could improve by **iteratively refining its predictions**, similar to extended reasoning time in humans.

### Approach
- A single refinement network is run multiple times
- Each iteration sees:
  - the board state
  - its own previous prediction
- Loss is computed at every step, weighted toward later iterations

### Key Findings
- Prediction loss **decreased by 28%** across thinking steps
- Gameplay performance **peaked at 1 thinking step**, then degraded

This revealed a subtle but important phenomenon:
> **More confident predictions are not necessarily better decisions.**

The model suffered from **premature convergence** — additional refinement increased confidence but moved the agent away from optimal gameplay.

---

## Why This Project Matters

This project goes far beyond Minesweeper.

It demonstrates:
- learning under partial observability,
- replacing symbolic logic with learned inference,
- careful data generation in environments without labels,
- and the limits of naïve self-improvement without reinforcement learning.

Most importantly, it shows that **neural systems can outperform rigid rule-based reasoning**, even in logic-heavy domains.

---

## Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib
- Custom Minesweeper environment & data pipeline

---

## How to Run
```bash
python train.py
