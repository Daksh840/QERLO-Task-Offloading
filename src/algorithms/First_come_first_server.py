import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import sys
import numpy as np
import os
import glob

def normalize_execution_times(G: nx.DiGraph, min_exec_time: float = 0.1) -> None:
    for node in G.nodes():
        exec_time = G.nodes[node].get('exec_time', 10)
        if exec_time < min_exec_time:
            G.nodes[node]['exec_time'] = min_exec_time

def calculate_energy_consumption(schedule: Dict[str, Any], node_powers: List[float]) -> float:
    total_energy = 0.0
    for task, info in schedule.items():
        exec_time = info['end_time'] - info['start_time']
        node = info['assigned_node']
        total_energy += exec_time * node_powers[node]
    return total_energy

def schedule_fcfs(G: nx.DiGraph, num_edge_nodes: int = 3,
                 verbose: bool = False, visualize: bool = False) -> Dict[str, Any]:
    normalize_execution_times(G)

    node_powers = [1.0 + 0.2 * i for i in range(num_edge_nodes)]

    node_available_time = [0.0] * num_edge_nodes
    task_list = list(nx.topological_sort(G))
    schedule = {}
    task_finish_times = {}

    for task in task_list:
        exec_time = max(0.1, float(G.nodes[task].get('exec_time', 10)))
        ready_time = max([task_finish_times.get(pred, 0) for pred in G.predecessors(task)] + [0])

        best_node = min(range(num_edge_nodes), key=lambda x: node_available_time[x])
        start_time = max(ready_time, node_available_time[best_node])
        end_time = start_time + exec_time

        schedule[task] = {
            'assigned_node': best_node,
            'start_time': start_time,
            'end_time': end_time,
            'exec_time': exec_time,
            'ready_time': ready_time
        }
        node_available_time[best_node] = end_time
        task_finish_times[task] = end_time

    makespan = max(node_available_time)
    total_energy = calculate_energy_consumption(schedule, node_powers)

    if verbose:
        print("\nFCFS Schedule Details:")
        print(f"{'Task':<25} {'Node':<5} {'Start':<10} {'End':<10} {'Duration':<10} {'Energy':<10}")
        for task, info in schedule.items():
            energy = (info['end_time'] - info['start_time']) * node_powers[info['assigned_node']]
            print(f"{task:<25} {info['assigned_node']:<5} "
                  f"{info['start_time']:<10.2f} {info['end_time']:<10.2f} "
                  f"{info['exec_time']:<10.2f} {energy:<10.2f}")

        print(f"\nTotal Makespan: {makespan:.2f} time units")
        print(f"Total Energy Consumption: {total_energy:.2f} units")

    return {
        'makespan': makespan,
        'energy': total_energy,
        'schedule': schedule,
        'node_powers': node_powers
    }

def load_dag_from_gml(path: str) -> nx.DiGraph:
    try:
        G = nx.read_gml(path)
        if not isinstance(G, nx.DiGraph):
            raise ValueError("Input is not a directed graph")

        for node in G.nodes():
            if 'exec_time' not in G.nodes[node]:
                if 'comp_cost' in G.nodes[node]:
                    G.nodes[node]['exec_time'] = float(G.nodes[node]['comp_cost']) / 1000
                else:
                    G.nodes[node]['exec_time'] = 10.0
            else:
                G.nodes[node]['exec_time'] = float(G.nodes[node]['exec_time'])
        return G
    except Exception as e:
        raise ValueError(f"Failed to load DAG {path}: {str(e)}")

def plot_metrics(makespan_history: List[float], energy_history: List[float]) -> None:
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(makespan_history, 'bo-', label='Makespan')  # Blue circles with line
    plt.xticks(range(len(makespan_history)), [str(i+1) for i in range(len(makespan_history))], rotation=90)
    plt.ylabel("Makespan (time units)")
    plt.title("Schedule Optimization Metrics Across DAGs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(energy_history, 'ro-', label='Energy Consumption')  # Red circles with line
    plt.xticks(range(len(energy_history)), [str(i+1) for i in range(len(energy_history))], rotation=90)
    plt.xlabel("DAG Index")
    plt.ylabel("Energy (units)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = r"C:\Users\palas\OneDrive\Desktop\DAD_Computing\TaskOffloadingOptimization\data\processed"
    gml_files = glob.glob(os.path.join(folder_path, "*.gml"))

    if not gml_files:
        print(f"No .gml files found in {folder_path}")
        sys.exit(1)

    makespans = []
    energies = []
    failed_files = []

    print(f"Found {len(gml_files)} DAG files. Processing...")

    for i, gml_file in enumerate(gml_files, 1):
        try:
            print(f"Processing ({i}/{len(gml_files)}): {os.path.basename(gml_file)}")
            G = load_dag_from_gml(gml_file)
            result = schedule_fcfs(G, num_edge_nodes=3, verbose=False, visualize=False)
            makespans.append(result['makespan'])
            energies.append(result['energy'])
        except Exception as e:
            print(f"ERROR processing {gml_file}: {str(e)}\nSkipping this file.")
            failed_files.append(gml_file)

    print("\nProcessing complete.")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f" - {f}")

    print(f"\nSummary of results for {len(makespans)} DAGs:")
    print(f"Mean makespan: {np.mean(makespans):.2f}")
    print(f"Mean energy: {np.mean(energies):.2f}")

    plot_metrics(makespans, energies)
