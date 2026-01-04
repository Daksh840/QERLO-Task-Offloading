# QERLO: Quantum-Inspired Reinforcement Learning for Task Offloading

## 📖 Project Overview
**QERLO** (Quantum-Inspired Evolutionary Reinforcement Learning Optimization) is a research project designed to solve the **Scientific Workflow Scheduling Problem** in Edge Computing environments.

### The Problem
We have complex "Scientific Workflows" (e.g., Earthquake analysis, Galaxy image processing) represented as **DAGs** (Directed Acyclic Graphs).
*   **Tasks:** Individual steps in the workflow (e.g., "Process Image A").
*   **Dependencies:** Task B cannot start until Task A finishes.
*   **Environment:** A cluster of **8 Edge Nodes** (computers) with different characteristics (some fast & power-hungry, some slow & efficient).

**The Goal:** Assign every task to a node to minimize:
1.  **Makespan:** Total time to complete the workflow.
2.  **Energy Consumption:** Total power used by the nodes.

---

## ⚙️ How It Works (Project Flow)
The project follows a rigorous simulation pipeline:

1.  **Input (Data Layer):** 
    *   Reads `.gml` files from `data/` (e.g., `CyberShake_30.gml`).
    *   These files define the tasks and their computational costs.

2.  **Orchestration (The Referee):**
    *   The script `src/algorithms/integrated_evaluation.py` loads every DAG.
    *   It passes the DAG to three competing algorithms: **HEFT** (Classical), **DQN** (AI), and **QIPSO** (Quantum-Inspired).

3.  **Simulation (The World):**
    *   Located in `src/environment/scheduler_env.py`.
    *   This acts as the "Game Engine". It tracks:
        *   Which node is busy until when.
        *   How much energy each node has used.
        *   Whether dependencies are met.

4.  **Output (Results):**
    *   Results are aggregated in `results/`.
    *   `scheduler_results.csv` contains the raw metrics.
    *   Plots are automatically generated to compare the algorithms.

---

## 🧠 Algorithmic Deep Dive

### 1. QIPSO (Quantum-Inspired Particle Swarm Optimization)
**The Star of the Show.** This algorithm outperforms others by using concepts from Quantum Computing to explore the search space more effectively.

#### 🧪 The "Q-bit" Theory (How it works in this code)
In this project, a **Q-bit** is not a physical hardware component, but a **Mathematical Model** of probability.

*   **Classical State:** A task is definitely on Node 1 (State = 1).
*   **Quantum State:** A task is in a **superposition**. It has a probability of being on Node 1, Node 2, or Node 8.

**Implementation Details (`algorithms/QIPSO.py`):**
*   **The Matrix:** `self.qbits` is a 3D matrix of size `[Particles, Tasks, Machines]`.
*   **The Probability:** `self.qbits[p, t, m]` represents the probability that Particle `p` assigns Task `t` to Machine `m`.
*   **Observation (Collapse):** When we need to test a schedule, we "collapse" the Q-bit:
    ```python
    # We choose a machine based on the Q-bit probabilities
    node = np.random.choice(machines, p=qbit_probabilities)
    ```
*   **Quantum Rotation:** After finding a good schedule, we "rotate" the Q-bits (update the probabilities) to make that successful configuration more likely in the future.

### 2. DQN (Deep Q-Network)
**The AI Agent.** Located in `src/algorithms/dqn_agent.py`.
*   It treats scheduling as a video game.
*   **State:** "I have Task A, and Node 3 is free."
*   **Action:** "Assign Task A to Node 3."
*   **Reward:** "Good job! You saved energy." OR "Bad job! That took too long."
*   **Learning:** Over thousands of episodes (`train_all_dags.py`), it creates a neural network map of the best decisions.

### 3. HEFT (Heterogeneous Earliest Finish Time)
**The Classical Baseline.** Located in `src/algorithms/heft.py`.
*   A greedy algorithm that ranks tasks by "criticality".
*   It simply places the most critical task on the machine that can finish it easiest.
*   **Role:** Used as a benchmark to prove that QIPSO and DQN are better.

---

## 📂 Project Structure Map
For anyone opening this project, here is where everything lives:

```
├── algorithms/                 # Standalone Algorithm Logic
│   ├── QIPSO.py               # <--- The Core Quantum Algorithm
│   ├── dqn_agent.py           # The Deep Learning Agent
│   └── heft.py                # The Classical Baseline
│
├── data/                       # Dataset of DAGs (Workflows)
│   ├── CyberShake_dags/       # Earthquake analysis workflows
│   └── Montage_dags/          # Image processing workflows
│
├── src/                        # Main Source Code
│   ├── environment/
│   │   └── scheduler_env.py   # The Simulation Logic (Gym Environment)
│   └── algorithms/
│       └── integrated_evaluation.py  # The MAIN SCRIPT to run benchmarks
│
├── results/                    # Output logs, CSVs, and plots
└── train_all_dags.py          # Script to training the DQN agent
```

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install networkx numpy matplotlib torch gym
    ```
2.  **Run the Benchmark:**
    ```bash
    python src/algorithms/integrated_evaluation.py
    ```
    This will run all algorithms on the datasets and generate plots in `results/`.
