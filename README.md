# QERLO: Quantum-Enhanced Reinforcement Learning for Task Offloading

> Research Internship Project – NSUT (Netaji Subhas University of Technology)  
> Duration: Jan 16, 2025 – June 20, 2025  
> Intern: Daksh Kumar Nahar | B.E. Computer Engineering, TIET

## 🔍 Overview

**QERLO** is a hybrid scheduling framework designed for offloading scientific workflows in heterogeneous edge computing environments.

It combines:
- 🌌 **QIPSO** (Quantum-Inspired Particle Swarm Optimization) for global task-node mapping.
- 🤖 **DQN** (Deep Q-Network) for real-time policy refinement.

The framework is tested on real-world workflows like **CyberShake, Montage, Epigenomics, Inspiral**, and **SIPHT**, achieving up to:
- 🔽 18% lower makespan
- 🔋 25% reduction in energy usage
- ⚡ Sub-second runtime inference

---

## 📌 Features

- DAG-based scientific workflow handling
- Energy-aware & latency-sensitive reward modeling
- Constraint-penalizing Q-learning
- Visualization of performance (Matplotlib plots)
- Compatible with custom GML workflows

---

## 🧠 Architecture

      +-------------------+
      |   Workflow DAG    |
      +-------------------+
                 ↓
     +---------------------+
     |   QIPSO Optimizer   |
     +---------------------+
                 ↓
     +---------------------+
     |   DQN Refiner       |
     +---------------------+
                 ↓
    +--------------------------+
    |   Optimal Task Schedule  |
    +--------------------------+


---

## 🛠️ Tech Stack

- **Python**
- **PyTorch** – DQN implementation
- **NetworkX** – DAG handling
- **NumPy / Matplotlib** – Processing + Plots
- **CUDA (Optional)** – GPU acceleration

---

## 🧪 Workflows Used

This repo includes real-world DAGs from Pegasus WMS:

- CyberShake
- Montage
- Epigenomics
- Inspiral
- SIPHT

Credit: [Scientific Workflow Visualization Dataset](https://github.com/maomao0217/Scientific-Workflow-Visualization)

---

## 📈 Results

| Metric     | QERLO vs FCFS/HEFT |
|------------|--------------------|
| Makespan   | ⬇️ 18%              |
| Energy     | ⬇️ 25%              |
| Runtime    | ⬇️ 0.3s inference   |

Visualizations are available in `/results`.

---

## 📦 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate QERLO
python main.py

