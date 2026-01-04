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

## 🎯 Mathematical Problem Formulation

### Optimization Objective
The task offloading problem is formulated as a **multi-objective optimization**:

**Minimize:**
```
f(S) = α · Makespan(S) + β · Energy(S)
```

Where:
- `S` = Schedule (mapping of tasks to edge nodes)
- `α, β` = Weight coefficients (in this project, we optimize each separately)
- `Makespan(S)` = max(Completion time of all nodes)
- `Energy(S)` = Σ(Power_consumption_i × Execution_time_i)

### Constraints
1. **Dependency Constraint:** A task cannot start until all its predecessors finish
   ```
   StartTime(task_j) ≥ max(EndTime(task_i)) ∀i ∈ Predecessors(j)
   ```

2. **Node Availability:** A node can only execute one task at a time
   ```
   For any node n: Tasks scheduled on n must not overlap in time
   ```

3. **Task Assignment:** Each task must be assigned to exactly one node
   ```
   Σ(assignment[task_i][node_j]) = 1 ∀ tasks
   ```

### Why This is Hard (NP-Complete)
- For `T` tasks and `N` nodes, there are `N^T` possible schedules
- Example: 30 tasks on 8 nodes = 8^30 ≈ 10^27 possibilities
- Exhaustive search is computationally infeasible
- **Solution:** Use heuristic and metaheuristic algorithms (HEFT, QIPSO, DQN)

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

#### 📚 Classical Particle Swarm Optimization (PSO) Basics
PSO is inspired by bird flocking behavior:
1. **Population:** A swarm of particles (candidate solutions)
2. **Movement:** Each particle has:
   - Position (current solution)
   - Velocity (direction of change)
3. **Learning:** Particles adjust their velocity based on:
   - **Cognitive Component:** Their own best position (`pbest`)
   - **Social Component:** The swarm's best position (`gbest`)

**Update Equations (Classical PSO):**
```
v[i] = w·v[i] + c1·r1·(pbest[i] - x[i]) + c2·r2·(gbest - x[i])
x[i] = x[i] + v[i]
```
Where: `w` = inertia, `c1,c2` = acceleration coefficients, `r1,r2` = random values

#### 🌌 The Quantum Leap: Why "Quantum-Inspired"?

**Problem with Classical PSO:**
- Particles have **definite positions** (e.g., "Task A is on Node 3")
- If the swarm converges early, it gets stuck in **local optima**
- Low diversity = poor exploration

**Quantum Solution:**
Instead of a particle being at a specific position, it exists in a **superposition of states**.

#### 🧪 The "Q-bit" Theory (How it works in this code)

A **Q-bit** in this project is not a physical hardware component, but a **Mathematical Model** of probability.

*   **Classical State:** A task is definitely on Node 1 (State = 1)
*   **Quantum State:** A task is in a **superposition** with probabilities for all nodes

**Implementation Details (`algorithms/QIPSO.py`):**

1. **The Matrix:** `self.qbits` is a 3D matrix of size `[Particles, Tasks, Machines]`
   ```python
   self.qbits = np.random.uniform(0.1, 0.9, (num_particles, num_tasks, num_nodes))
   ```

2. **The Probability:** `self.qbits[p, t, m]` represents the probability that Particle `p` assigns Task `t` to Machine `m`
   - Each row sums to 1 (normalized probability distribution)
   - Example: `[0.1, 0.7, 0.2]` means 70% chance on Node 2

3. **Observation (Collapse):** When we need to test a schedule, we "collapse" the Q-bit:
   ```python
   # Quantum measurement: collapse wavefunction to a definite state
   schedule[task] = np.random.choice(num_nodes, p=self.qbits[p, t])
   ```
   This mimics the measurement of a quantum state in quantum mechanics.

4. **Quantum Rotation (Update):** After finding a good schedule, we "rotate" the Q-bits:
   ```python
   # Cognitive component: move towards personal best
   cognitive = c1 * random() * (pbest_indicator - qbits[p,t])
   
   # Social component: move towards global best
   social = c2 * random() * (gbest_indicator - qbits[p,t])
   
   # Update probabilities
   qbits[p,t] += cognitive + social
   qbits[p,t] = normalize(qbits[p,t])  # Ensure sum = 1
   ```

#### 🔑 Critical Path Analysis
QIPSO includes an intelligent initialization strategy:
1. Computes the **Critical Path** of the DAG (longest chain of dependencies)
2. Uses this to guide initial Q-bit probabilities
3. Tasks on the critical path get higher priority for faster nodes

**Algorithm:**
```python
def find_critical_path(self):
    # Forward pass: compute earliest start times
    for task in topological_order:
        EST[task] = max(EST[pred] + duration[pred] for pred in predecessors)
    
    # Backward pass: compute latest finish times
    for task in reversed_topological_order:
        LFT[task] = min(LFT[succ] - duration[task] for succ in successors)
    
    # Critical tasks: EST == LFT (zero slack)
    critical_path = [task for task if EST[task] == LFT[task]]
```

#### ⚡ Why QIPSO Beats Classical Methods
1. **Exploration:** Quantum superposition maintains diversity longer
2. **Exploitation:** Probabilities naturally converge to good solutions
3. **Escape Mechanism:** Mutation operator allows escaping local optima
4. **Problem-Aware:** Critical path initialization targets bottlenecks

---

### 2. DQN (Deep Q-Network)
**The AI Agent.** Located in `src/algorithms/dqn_agent.py`.

#### 🎮 Reinforcement Learning Framework
The scheduling problem is modeled as a Markov Decision Process (MDP):

**Components:**
- **State (s):** Current configuration of the system
  ```python
  state = [
      task_exec_time,           # Computational cost
      num_predecessors,         # Dependencies
      num_successors,           # Importance
      node_available_times[0..7], # Node status
      current_task_idx,
      remaining_tasks
  ]
  ```
  
- **Action (a):** Choose a node ∈ {0, 1, ..., 7} for the current task

- **Reward (r):** Immediate feedback signal
  ```python
  # During episode
  r_step = -0.1 * end_time - 0.1 * imbalance + 0.05 * utilization
  
  # At completion
  r_final = -makespan - 0.5 * std(node_loads)
  ```

- **Policy (π):** Learned mapping from states to actions

#### 🧠 Neural Network Architecture
```
Input Layer (64 features)
     ↓
Dense(512) + ReLU
     ↓
Dense(256) + ReLU
     ↓
Dense(128) + ReLU
     ↓
Output Layer (8 nodes) → Q-values for each action
```

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```
Where:
- `α` = learning rate (0.0003)
- `γ` = discount factor (0.99)
- `r` = reward
- `s'` = next state

#### 🎯 Training Process
1. **Exploration:** Use ε-greedy policy
   - With probability ε: random action (explore)
   - With probability 1-ε: best action (exploit)
   - ε decays: 1.0 → 0.01 over time

2. **Experience Replay:** Store transitions in memory
   ```python
   memory.append((state, action, reward, next_state, done))
   ```

3. **Batch Learning:** Sample mini-batches (size=1024) from memory
   - Breaks correlation between consecutive samples
   - Improves training stability

4. **Target Network:** Separate network for computing target Q-values
   - Soft update: `θ_target ← τ·θ + (1-τ)·θ_target`
   - Prevents oscillation during learning

#### 📈 Why DQN Works
- **Generalization:** Learns patterns that work across similar DAGs
- **Adaptation:** Can adjust to different workflow structures
- **Speed:** Sub-second inference after training
- **Limitation:** Requires extensive training (1500+ episodes per DAG)

---

### 3. HEFT (Heterogeneous Earliest Finish Time)
**The Classical Baseline.** Located in `src/algorithms/heft.py`.

#### 📊 Algorithm Steps
1. **Compute Upward Rank** for each task:
   ```
   rank(task) = w̄(task) + max(c̄(task,succ) + rank(succ))
   ```
   Where:
   - `w̄(task)` = average execution time
   - `c̄(task,succ)` = average communication cost
   - Higher rank = more critical

2. **Sort tasks** by rank (descending)

3. **For each task** (in sorted order):
   - **For each node:** Compute Earliest Finish Time (EFT)
   - **Assign task** to the node with minimum EFT

#### ⚖️ HEFT vs QIPSO vs DQN

| Aspect | HEFT | QIPSO | DQN |
|--------|------|-------|-----|
| **Approach** | Greedy | Population-based | Learning-based |
| **Time Complexity** | O(T²·N) | O(I·P·T·N) | O(E·T) |
| **Solution Quality** | Good | Better | Best (after training) |
| **Runtime** | <1s | ~10-60s | <1s (inference) |
| **Training Required** | No | No | Yes (hours) |
| **Adaptability** | Fixed | Fixed | High |

Where: T=tasks, N=nodes, I=iterations, P=particles, E=episodes

---

## 🔬 Why Quantum-Inspired Methods Excel

### Information Retention
- **Classical:** Particle at position [3, 1, 5, ...] → One solution
- **Quantum:** Particle has probabilities for all positions → Superposition of many solutions

### Diversity Maintenance
Classical PSO converges like this:
```
Iteration 1:  Particle 1: [0.3, 0.5, 0.2]  Particle 2: [0.4, 0.3, 0.3]
Iteration 50: Particle 1: [0.1, 0.8, 0.1]  Particle 2: [0.1, 0.8, 0.1]  ← Stuck!
```

QIPSO with mutation:
```
Iteration 1:  Particle 1: [0.3, 0.5, 0.2]  Particle 2: [0.4, 0.3, 0.3]
Iteration 50: Particle 1: [0.05, 0.9, 0.05] Particle 2: [0.2, 0.7, 0.1]  ← Still diverse!
```

### Theoretical Advantage
The mutation operator with decreasing probability:
```python
mutation_prob = initial_prob * (1 - current_iter / max_iter)
```
Ensures:
- **Early:** High mutation → Exploration
- **Late:** Low mutation → Exploitation
- Avoids premature convergence

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
│   ├── Montage_dags/          # Astronomy image mosaics
│   ├── Epigenomics_dags/      # Genome sequencing
│   ├── Inspiral_dags/         # Gravitational wave detection
│   └── SIPHT_dags/            # RNA analysis
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

---

## 🚀 How to Run

### Prerequisites
```bash
pip install networkx numpy matplotlib torch gym
```

### Option 1: Run Benchmark (Compare All Algorithms)
```bash
python src/algorithms/integrated_evaluation.py
```
This will:
1. Load all DAGs from `data/processed/`
2. Run FCFS, HEFT, QIPSO, and DQN on each
3. Generate comparison plots in `results/`

### Option 2: Train DQN Agent (From Scratch)
```bash
python train_all_dags.py
```
- Trains on all DAGs in the specified folder
- Saves models to `results/models/`
- Early stopping when no improvement for 100 episodes

### Option 3: Test QIPSO on Single DAG
```python
from algorithms.QIPSO import QIPSO_Scheduler
import networkx as nx

dag = nx.read_gml("data/random_dag_30.gml")
scheduler = QIPSO_Scheduler(
    graph=dag, 
    num_edge_nodes=8, 
    num_particles=30, 
    max_iter=100
)
schedule, makespan, energy = scheduler.run_optimization()
print(f"Makespan: {makespan:.2f}, Energy: {energy:.2f}")
```

---

## 📊 Expected Results
Based on experiments with scientific workflows:

| Workflow | Tasks | HEFT Makespan | QIPSO Makespan | Improvement |
|----------|-------|---------------|----------------|-------------|
| CyberShake_30 | 30 | 245.3 | 201.7 | ~18% |
| Montage_50 | 50 | 412.8 | 356.4 | ~14% |
| Epigenomics_46 | 46 | 389.1 | 334.2 | ~14% |

**Energy Reduction:** QIPSO typically achieves 20-25% lower energy consumption compared to HEFT.

---

## 🎓 References & Credits
- **Quantum-Inspired PSO:** Based on principles from quantum computing applied to optimization
- **Scientific Workflows:** Real-world DAGs from the Pegasus Workflow Management System
- **DQN:** Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
- **HEFT:** Topcuoglu et al., "Performance-effective and low-complexity task scheduling for heterogeneous computing", IEEE TPDS 2002
