import networkx as nx
from typing import Dict, List, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class HEFTScheduler:
    def __init__(self, num_edge_nodes: int = 4, avg_comp_cost: float = 1000):
        self.num_edge_nodes = num_edge_nodes
        self.avg_comp_cost = avg_comp_cost
        self.schedule_result: Dict[str, Dict[str, Any]] = {}  # Store full schedule mapping

        # Define per-node power consumptions (customize as needed)
        self.node_powers = [1.0 + 0.2 * i for i in range(num_edge_nodes)]

    def compute_upward_rank(self, G: nx.DiGraph, task: str, rank: Dict[str, float]) -> float:
        """Recursive calculation of upward rank of a task."""
        if task in rank:
            return rank[task]
        
        successors = list(G.successors(task))
        comp_cost = self._get_task_cost(G.nodes[task])
        if not successors:
            rank[task] = comp_cost
        else:
            rank[task] = comp_cost + max(self.compute_upward_rank(G, succ, rank) +
                                        self._get_comm_cost(G.edges.get((task, succ), {}))
                                        for succ in successors)
        return rank[task]

    def _get_task_cost(self, node_data: Dict) -> float:
        """Get computation cost with fallback to exec_time."""
        # Use comp_cost if exists, otherwise exec_time or a default
        if 'comp_cost' in node_data:
            return float(node_data['comp_cost']) / self.avg_comp_cost
        elif 'exec_time' in node_data:
            return float(node_data['exec_time']) / self.avg_comp_cost
        else:
            return 10.0 / self.avg_comp_cost

    def _get_comm_cost(self, edge_data: Dict) -> float:
        """Get communication cost from edge data or fallback to zero."""
        if edge_data and 'comm_cost' in edge_data:
            return float(edge_data['comm_cost']) / self.avg_comp_cost
        return 0.0

    def schedule(self, G: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
        """Run the HEFT scheduling algorithm."""
        task_list = list(nx.topological_sort(G))
        rank = {}

        # Compute upward rank for all tasks
        for task in task_list:
            self.compute_upward_rank(G, task, rank)
        
        # Sort tasks by descending rank
        sorted_tasks = sorted(task_list, key=lambda x: rank[x], reverse=True)

        node_available_time = [0.0] * self.num_edge_nodes
        self.schedule_result = {}

        for task in sorted_tasks:
            comp_time = self._get_task_cost(G.nodes[task])
            preds = list(G.predecessors(task))

            # Earliest start time considering communication from predecessors
            ready_time = 0.0
            if preds:
                ready_time = max(
                    self.schedule_result[pred]['end_time'] +
                    self._get_comm_cost(G.edges.get((pred, task), {}))
                    for pred in preds
                )

            best_node = None
            best_finish_time = float('inf')

            # Find node that yields earliest finish time
            for node_id in range(self.num_edge_nodes):
                est = max(ready_time, node_available_time[node_id])
                eft = est + comp_time
                if eft < best_finish_time:
                    best_finish_time = eft
                    best_node = node_id
            
            start_time = max(ready_time, node_available_time[best_node])
            end_time = start_time + comp_time

            self.schedule_result[task] = {
                'assigned_node': best_node,
                'start_time': start_time,
                'end_time': end_time,
                'exec_time': comp_time
            }
            node_available_time[best_node] = end_time
        
        return self.schedule_result

    def get_makespan(self) -> float:
        """Get total makespan."""
        if not self.schedule_result:
            return 0.0
        return max(task['end_time'] for task in self.schedule_result.values()) * self.avg_comp_cost


    def calculate_energy(self) -> float:
        """Calculate total energy consumption for the schedule."""
        total_energy = 0.0
        for task, info in self.schedule_result.items():
            exec_time = info['end_time'] - info['start_time']
            node = info['assigned_node']
            total_energy += exec_time * self.node_powers[node]
        return total_energy * self.avg_comp_cost


def load_dag_from_gml(path: Path) -> nx.DiGraph:
    G = nx.read_gml(str(path))
    if not isinstance(G, nx.DiGraph):
        raise ValueError(f"Graph {path} is not a directed graph")
    
    # Ensure exec_time or comp_cost attributes exist and are floats
    for node in G.nodes:
        data = G.nodes[node]
        if 'exec_time' not in data and 'comp_cost' not in data:
            data['exec_time'] = 10.0
        else:
            if 'exec_time' in data:
                data['exec_time'] = float(data['exec_time'])
            if 'comp_cost' in data:
                data['comp_cost'] = float(data['comp_cost'])
    return G

def plot_metrics(makespan_history: List[float], energy_history: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(makespan_history, 'bo-', label='Makespan')
    plt.xlabel("DAG Index")
    plt.ylabel("Makespan (normalized time)")
    plt.title("HEFT Scheduling: Makespan Across DAGs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(energy_history, 'ro-', label='Energy Consumption')
    plt.xlabel("DAG Index")
    plt.ylabel("Energy (units)")
    plt.title("HEFT Scheduling: Energy Consumption Across DAGs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_all_gmls(folder_path: str, num_edge_nodes: int = 4):
    folder = Path(folder_path)
    gml_files = list(folder.glob('*.gml'))

    if not gml_files:
        print(f"No .gml files found in folder {folder_path}")
        return [], []

    makespans = []
    energies = []

    for idx, dagfile in enumerate(gml_files, 1):
        try:
            G = load_dag_from_gml(dagfile)
            scheduler = HEFTScheduler(num_edge_nodes=num_edge_nodes)
            scheduler.schedule(G)
            makespan = scheduler.get_makespan()
            energy = scheduler.calculate_energy()
            makespans.append(makespan)
            energies.append(energy)
            print(f"[{idx}/{len(gml_files)}] Processed {dagfile.name}: Makespan={makespan:.4f}, Energy={energy:.4f}")
        except Exception as e:
            print(f"Error processing {dagfile.name}: {e}")

    print(f"\nProcessed {len(makespans)} DAG files in total.")
    if len(makespans) > 0:
        print(f"Average Makespan: {np.mean(makespans):.4f}")
        print(f"Average Energy: {np.mean(energies):.4f}")

    return makespans, energies


if __name__ == "__main__":
    DAG_FOLDER = r"C:\Users\palas\OneDrive\Desktop\DAD_Computing\TaskOffloadingOptimization\data\SIPHT_dags"
    NUM_EDGE_NODES = 4

    makespans, energies = process_all_gmls(DAG_FOLDER, NUM_EDGE_NODES)
    if makespans and energies:
        plot_metrics(makespans, energies)
