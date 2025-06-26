import os
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

class QIPSO_Scheduler:
    def __init__(self, graph: nx.DiGraph, num_edge_nodes: int = 8, num_particles: int = 50, 
                 max_iter: int = 200, cognitive_weight: float = 1.7, social_weight: float = 1.7, 
                 mutation_prob: float = 0.3):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Input must be a NetworkX DiGraph")
        if num_edge_nodes < 1:
            raise ValueError("Must have at least 1 edge node")
        if mutation_prob < 0 or mutation_prob > 1:
            raise ValueError("Mutation probability must be between 0 and 1")

        # Relabel nodes to str for consistency
        self.dag = nx.relabel_nodes(graph, lambda x: str(x))
        try:
            self.task_list = [str(task) for task in nx.topological_sort(self.dag)]
            self.num_tasks = len(self.task_list)
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains cycles - must be a DAG")

        self.num_nodes = num_edge_nodes
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.cognitive = cognitive_weight
        self.social = social_weight
        self.mutation_prob = mutation_prob

        self.node_power_consumptions = [1.0 + 0.2*i for i in range(num_edge_nodes)]
        self.energy_history = []

        self.critical_path, self.critical_length = self.find_critical_path()
        print(f"Critical path length: {self.critical_length:.2f}")
        print(f"Critical path tasks: {self.critical_path}")

        self.qbits = np.random.uniform(0.1, 0.9, (num_particles, self.num_tasks, num_edge_nodes))
        self.normalize_qbits()
        self.velocities = np.zeros_like(self.qbits)

        self.personal_best_scores = [float('inf')] * num_particles
        self.personal_best_schedules = [{} for _ in range(num_particles)]
        self.global_best_score = float('inf')
        self.global_best_energy = float('inf')
        self.global_best_schedule = {}
        self.convergence_history = [] 

    def normalize_qbits(self):
        for p in range(self.num_particles):
            for t in range(self.num_tasks):
                self.qbits[p,t] = np.clip(self.qbits[p,t], 0.01, 0.99)
                total = np.sum(self.qbits[p,t]) + 1e-10
                self.qbits[p,t] /= total

    def find_critical_path(self) -> Tuple[List[str], float]:
        task_times = {}
        for task in self.task_list:
            exec_time = float(self.dag.nodes[task].get('exec_time', 
                                    self.dag.nodes[task].get('comp_cost', 10000)/1000))
            task_times[task] = max(0.1, exec_time)

        earliest_start = defaultdict(float)
        for task in self.task_list:
            for successor in self.dag.successors(task):
                earliest_start[successor] = max(
                    earliest_start[successor],
                    earliest_start[task] + task_times[task]
                )

        latest_finish = defaultdict(lambda: max(earliest_start.values()))
        for task in reversed(self.task_list):
            for predecessor in self.dag.predecessors(task):
                latest_finish[predecessor] = min(
                    latest_finish[predecessor],
                    latest_finish[task] - task_times[predecessor]
                )

        critical_path = [
            task for task in self.task_list
            if abs(earliest_start[task] - (latest_finish[task] - task_times[task])) < 1e-6
        ]
        if not critical_path:
            critical_path = [self.task_list[0]]

        return critical_path, max(earliest_start.values()) + task_times[critical_path[-1]]

    def evaluate_schedule(self, schedule: Dict[str, int]) -> Tuple[float, float]:
        device_end_times = [0.0] * self.num_nodes
        device_energy = [0.0] * self.num_nodes
        task_finish_times = {}
        
        for task in self.task_list:
            node = schedule[task]
            duration = max(0.1, float(self.dag.nodes[task].get('exec_time', 10)))
            
            ready_time = max(
                [task_finish_times.get(dep, 0) for dep in self.dag.predecessors(task)] + [0]
            )
            
            start_time = max(ready_time, device_end_times[node])
            end_time = start_time + duration
            
            device_energy[node] += self.node_power_consumptions[node] * duration
            
            task_finish_times[task] = end_time
            device_end_times[node] = end_time
            
        return max(device_end_times), sum(device_energy)

    def run_optimization(self) -> Tuple[Dict[str,int], float, float]:
        print(f"Starting QIPSO with {self.num_particles} particles for {self.max_iter} iterations")
        
        for iteration in range(self.max_iter):
            current_mutation = self.mutation_prob * (1 - iteration/self.max_iter)
            
            for p in range(self.num_particles):
                schedule = {}
                for t, task in enumerate(self.task_list):
                    probs = self.qbits[p,t].copy()
                    probs = np.clip(probs, 0.01, 0.99)
                    probs /= probs.sum()
                    schedule[task] = np.random.choice(self.num_nodes, p=probs)
                
                makespan, energy = self.evaluate_schedule(schedule)
                
                if makespan < self.personal_best_scores[p]:
                    self.personal_best_scores[p] = makespan
                    self.personal_best_schedules[p] = schedule.copy()
                    
                    if makespan < self.global_best_score:
                        self.global_best_score = makespan
                        self.global_best_energy = energy
                        self.global_best_schedule = schedule.copy()
            
            self.update_quantum_states(iteration)
            self.convergence_history.append((self.global_best_score, self.global_best_energy))
            
            if self.check_convergence(iteration):
                break
                
        return self.global_best_schedule, self.global_best_score, self.global_best_energy

    def update_quantum_states(self, iteration: int):
        for p in range(self.num_particles):
            for t, task in enumerate(self.task_list):
                personal_best_node = self.personal_best_schedules[p].get(task, 0)
                global_best_node = self.global_best_schedule.get(task, 0)
                
                cognitive = self.cognitive * random.random() * (
                    (np.arange(self.num_nodes) == personal_best_node) - self.qbits[p,t]
                )
                social = self.social * random.random() * (
                    (np.arange(self.num_nodes) == global_best_node) - self.qbits[p,t]
                )
                
                self.velocities[p,t] = np.clip(
                    self.velocities[p,t] + cognitive + social,
                    -0.2, 0.2
                )
                
                self.qbits[p,t] = np.clip(
                    self.qbits[p,t] + self.velocities[p,t],
                    0.01, 0.99
                )
                
                if random.random() < (self.mutation_prob * (1 - iteration/self.max_iter)):
                    mut_node = random.randint(0, self.num_nodes-1)
                    self.qbits[p,t][mut_node] = np.clip(
                        self.qbits[p,t][mut_node] + random.uniform(-0.1, 0.1),
                        0.01, 0.99
                    )
                
                self.qbits[p,t] /= self.qbits[p,t].sum()

    def check_convergence(self, iteration: int) -> bool:
        if iteration % 10 == 0 or iteration == self.max_iter - 1:
            print(f"Iter {iteration:4d}: Makespan = {self.global_best_score:.2f}, "
                  f"Energy = {self.global_best_energy:.2f}")

        if abs(self.global_best_score - self.critical_length) < 0.1:
            print(f"\n✅ Reached theoretical minimum at iteration {iteration}")
            return True
        
        if len(self.convergence_history) > 20:
            recent_improvement = abs(min(
                [x[0] for x in self.convergence_history[-20:]]) - self.global_best_score
            )
            if recent_improvement < 0.1:
                print(f"\n⚠️ Early stopping at iteration {iteration} (no improvement)")
                return True
        return False

    def plot_dual_metrics(self):
        makespans = [x[0] for x in self.convergence_history]
        energies = [x[1] for x in self.convergence_history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(makespans, label='Best Makespan', color='blue')
        ax1.axhline(y=self.critical_length, color='r', linestyle='--',
                    label=f'Critical Path ({self.critical_length:.2f})')
        ax1.set_ylabel("Makespan (seconds)")
        ax1.set_ylim(0, max(makespans)*1.1)
        ax1.legend()
        ax1.grid(True)

        ax2.plot(energies, label='Energy Consumption', color='green')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Energy (joules)")
        ax2.set_ylim(0, max(energies)*1.1)
        ax2.legend()
        ax2.grid(True)

        plt.suptitle("QIPSO Optimization Metrics")
        plt.tight_layout()
        plt.savefig("qipso_dual_metrics.png", dpi=300)
        plt.show()

    def print_schedule_stats(self, schedule: Dict[str, int]):
        node_times = [0.0] * self.num_nodes
        node_energy = [0.0] * self.num_nodes

        print("\nDetailed Schedule Analysis:")
        print(f"{'Task':>20} | {'Node':>5} | {'Duration':>8} | {'Energy':>8}")
        print("-" * 50)

        for task in self.task_list:
            node = schedule[task]
            duration = float(self.dag.nodes[task].get('exec_time', 10))
            energy = self.node_power_consumptions[node] * duration
            node_times[node] += duration
            node_energy[node] += energy

            print(f"{task:>20} | {node:>5} | {duration:>8.2f} | {energy:>8.2f}")

        print("\nNode Utilization:")
        print(f"{'Node':>5} | {'Busy Time':>10} | {'Energy':>10} | {'Power':>10}")
        print("-" * 40)
        for i in range(self.num_nodes):
            print(f"{i:>5} | {node_times[i]:>10.2f} | {node_energy[i]:>10.2f} | "
                  f"{self.node_power_consumptions[i]:>10.2f}")

def load_dag_safe(path: Path) -> nx.DiGraph:
    G = nx.read_gml(str(path))
    if all('label' in data for _, data in G.nodes(data=True)):
        G = nx.relabel_nodes(G, {n: data['label'] for n, data in G.nodes(data=True)})
    return G

def process_dags_batch(folder_path: str, num_edge_nodes=8, num_particles=30, max_iter=100):
    folder = Path(folder_path)
    gml_files = list(folder.glob('*.gml'))

    if not gml_files:
        print(f"No .gml files found in folder {folder_path}")
        return [], []

    makespans = []
    energies = []

    for idx, dagfile in enumerate(gml_files, 1):
        try:
            print(f"\nProcessing ({idx}/{len(gml_files)}): {dagfile.name}")
            dag = load_dag_safe(dagfile)
            scheduler = QIPSO_Scheduler(
                graph=dag,
                num_edge_nodes=num_edge_nodes,
                num_particles=num_particles,
                max_iter=max_iter
            )
            best_schedule, best_makespan, best_energy = scheduler.run_optimization()
            makespans.append(best_makespan)
            energies.append(best_energy)
            print(f"Best Makespan: {best_makespan:.2f}, Energy: {best_energy:.2f}")
        except Exception as e:
            print(f"ERROR processing {dagfile.name}: {e}")

    print(f"\nProcessed {len(makespans)} DAG files.")

    return makespans, energies

def plot_batch_metrics(makespans: List[float], energies: List[float]):
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(makespans, 'bo-')
    plt.title("QIPSO Makespan Across DAGs")
    plt.xlabel("DAG Index")
    plt.ylabel("Makespan")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(energies, 'ro-')
    plt.title("QIPSO Energy Consumption Across DAGs")
    plt.xlabel("DAG Index")
    plt.ylabel("Energy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DAG_FOLDER = r"C:\Users\palas\OneDrive\Desktop\DAD_Computing\TaskOffloadingOptimization\data\SIPHT_dags"
    # Run batch processing
    makespans, energies = process_dags_batch(DAG_FOLDER, num_edge_nodes=8, num_particles=30, max_iter=100)
    if makespans and energies:
        plot_batch_metrics(makespans, energies)
