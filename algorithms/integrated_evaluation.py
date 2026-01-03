import os
import time
import json
import csv
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to reach the project root
sys.path.append(str(project_root))


from src.algorithms.First_come_first_server import schedule_fcfs
from src.algorithms.heft import HEFTScheduler
from src.algorithms.QIPSO import QIPSO_Scheduler
from src.algorithms.dqn_agent import DQNAgent
from src.environment.scheduler_env import TaskOffloadingEnv

NODE_POWERS = [1.0 + 0.2*i for i in range(8)]  # Customize if necessary

def calculate_energy(schedule, node_powers=NODE_POWERS):
    total_energy = 0.0
    for task, info in schedule.items():
        exec_time = info['end_time'] - info['start_time']
        node = info['assigned_node']
        total_energy += exec_time * node_powers[node]
    return total_energy

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def evaluate_fcfs(G, num_nodes=8):
    (makespan, energy), runtime = time_function(lambda G: (schedule_fcfs(G, num_edge_nodes=num_nodes, verbose=False, visualize=False)['makespan'],
                                                           schedule_fcfs(G, num_edge_nodes=num_nodes, verbose=False, visualize=False)['energy']), G)
    return makespan, energy, runtime

def evaluate_heft(G, num_nodes=8):
    def run_heft(G):
        scheduler = HEFTScheduler(num_edge_nodes=num_nodes)
        schedule = scheduler.schedule(G)
        makespan = scheduler.get_makespan()
        energy = 0
        for task, info in schedule.items():
            exec_time = info['end_time'] - info['start_time']
            node = info['assigned_node']
            energy += exec_time * NODE_POWERS[node]
        return makespan, energy

    (makespan, energy), runtime = time_function(run_heft, G)
    return makespan, energy, runtime

def evaluate_qipso(G, num_nodes=8):
    def run_qipso(G):
        scheduler = QIPSO_Scheduler(graph=G, num_edge_nodes=num_nodes, num_particles=30, max_iter=100)
        schedule, makespan, energy = scheduler.run_optimization()
        return makespan, energy

    (makespan, energy), runtime = time_function(run_qipso, G)
    return makespan, energy, runtime

def evaluate_dqn(G, model_path, num_nodes=8):
    def run_dqn(G, model_path):
        env = TaskOffloadingEnv(G, num_nodes=num_nodes)
        agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
        checkpoint = torch.load(model_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.update_target_model()

        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon=0)
            state, reward, done, info = env.step(action)

        schedule = env.get_schedule()
        makespan = max(info['end_time'] for info in schedule.values())
        energy = calculate_energy(schedule, NODE_POWERS)
        return makespan, energy

    (makespan, energy), runtime = time_function(run_dqn, G, model_path)
    return makespan, energy, runtime

def main():
    dag_folder = Path("C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/processed")
    model_folder = Path("C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/results/models")
    output_folder = Path("C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/results/aggregated")
    output_folder.mkdir(parents=True, exist_ok=True)
    num_nodes = 8

    dag_files = sorted(dag_folder.glob("*.gml"))
    algorithms = ['FCFS', 'HEFT', 'QIPSO', 'DQN']
    results = {algo: [] for algo in algorithms}

    for dag_file in dag_files:
        print(f"Evaluating {dag_file.name}...")
        G = nx.read_gml(str(dag_file))
        for node in G.nodes():
            if 'exec_time' not in G.nodes[node]:
                G.nodes[node]['exec_time'] = 10.0
            else:
                G.nodes[node]['exec_time'] = float(G.nodes[node]['exec_time'])

        # FCFS
        try:
            m, e, rt = evaluate_fcfs(G, num_nodes)
            results['FCFS'].append({'dag': dag_file.name, 'makespan': m, 'energy': e, 'runtime_sec': rt})
        except Exception as ex:
            print(f"FCFS failed on {dag_file.name}: {ex}")
            results['FCFS'].append({'dag': dag_file.name, 'makespan': None, 'energy': None, 'runtime_sec': None})

        # HEFT
        try:
            m, e, rt = evaluate_heft(G, num_nodes)
            results['HEFT'].append({'dag': dag_file.name, 'makespan': m, 'energy': e, 'runtime_sec': rt})
        except Exception as ex:
            print(f"HEFT failed on {dag_file.name}: {ex}")
            results['HEFT'].append({'dag': dag_file.name, 'makespan': None, 'energy': None, 'runtime_sec': None})

        # QIPSO
        try:
            m, e, rt = evaluate_qipso(G, num_nodes)
            results['QIPSO'].append({'dag': dag_file.name, 'makespan': m, 'energy': e, 'runtime_sec': rt})
        except Exception as ex:
            print(f"QIPSO failed on {dag_file.name}: {ex}")
            results['QIPSO'].append({'dag': dag_file.name, 'makespan': None, 'energy': None, 'runtime_sec': None})

        # DQN
        model_path = model_folder / f"{dag_file.stem}_final.pth"
        if model_path.exists():
            try:
                m, e, rt = evaluate_dqn(G, str(model_path), num_nodes)
                results['DQN'].append({'dag': dag_file.name, 'makespan': m, 'energy': e, 'runtime_sec': rt})
            except Exception as ex:
                print(f"DQN failed on {dag_file.name}: {ex}")
                results['DQN'].append({'dag': dag_file.name, 'makespan': None, 'energy': None, 'runtime_sec': None})
        else:
            print(f"No trained DQN model for {dag_file.stem}, skipping DQN evaluation")
            results['DQN'].append({'dag': dag_file.name, 'makespan': None, 'energy': None, 'runtime_sec': None})

    # Export results to CSV and JSON
    export_results(results, output_folder)

    # Generate plots
    plot_results(results, algorithms)

def export_results(results, output_folder):
    # Save CSV
    csv_path = output_folder / "scheduler_results.csv"
    fieldnames = ['dag', 'algorithm', 'makespan', 'energy', 'runtime_sec']
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for algo, data_list in results.items():
            for data in data_list:
                writer.writerow({
                    'dag': data['dag'],
                    'algorithm': algo,
                    'makespan': data['makespan'],
                    'energy': data['energy'],
                    'runtime_sec': data['runtime_sec']
                })
    print(f"Results exported to CSV: {csv_path}")

    # Save JSON
    json_path = output_folder / "scheduler_results.json"
    with open(json_path, 'w') as f_json:
        json.dump(results, f_json, indent=4)
    print(f"Results exported to JSON: {json_path}")

import numpy as np
import matplotlib.pyplot as plt

def plot_results(results, algorithms):
    dag_names = [r['dag'] for r in next(iter(results.values()))]
    x = np.arange(len(dag_names))
    width = 0.2

    plt.figure(figsize=(20, 8))

    # Makespan bar chart
    plt.subplot(1, 2, 1)
    for i, algo in enumerate(algorithms):
        makespans = [r['makespan'] if r['makespan'] is not None else 0 for r in results[algo]]
        plt.bar(x + i*width, makespans, width=width, label=algo)
    plt.xticks(
        ticks=range(0, len(dag_names), 5),  # Show every 5th DAG label (positional arguments)
        labels=[dag_names[i] for i in range(0, len(dag_names), 5)],
        rotation=90,
        fontsize=8
    )
    plt.ylabel("Makespan")
    plt.title("Makespan Comparison Across DAGs")
    plt.legend()
    plt.grid(axis='y')

    # Energy bar chart
    plt.subplot(1, 2, 2)
    for i, algo in enumerate(algorithms):
        energies = [r['energy'] if r['energy'] is not None else 0 for r in results[algo]]
        plt.bar(x + i*width, energies, width=width, label=algo)
    plt.xticks(
        ticks=range(0, len(dag_names), 5),  # Show every 5th DAG label (positional arguments)
        labels=[dag_names[i] for i in range(0, len(dag_names), 5)],
        rotation=90,
        fontsize=8
    )
    plt.ylabel("Energy")
    plt.title("Energy Consumption Comparison Across DAGs")
    plt.legend()
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
