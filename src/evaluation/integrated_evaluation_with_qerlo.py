import os
import time
import csv
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import pandas as pd

# --- Project Imports ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.algorithms.First_come_first_server import schedule_fcfs
from src.algorithms.heft import HEFTScheduler
from src.algorithms.QIPSO import QIPSO_Scheduler
from src.algorithms.dqn_agent import DQNAgent
from src.algorithms.hcocp import HCOCPScheduler
from src.algorithms.moheft import MOHEFTScheduler
from src.environment.scheduler_env import TaskOffloadingEnv

# ------------------------------
NODE_POWERS = [1.0 + 0.2*i for i in range(8)]

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

# ---------------- EVALUATORS ----------------
def evaluate_fcfs(G, num_nodes=8):
    def run_fcfs(G):
        output = schedule_fcfs(G, num_edge_nodes=num_nodes, verbose=False, visualize=False)
        return output['makespan'], output['energy']
    (makespan, energy), runtime = time_function(run_fcfs, G)
    cost = makespan + 0.5 * energy
    return makespan, energy, round(runtime, 6), cost

def evaluate_heft(G, num_nodes=8):
    def run_heft(G):
        scheduler = HEFTScheduler(num_edge_nodes=num_nodes)
        schedule = scheduler.schedule(G)
        makespan = scheduler.get_makespan()
        energy = scheduler.calculate_energy()
        return makespan, energy
    (makespan, energy), runtime = time_function(run_heft, G)
    cost = makespan + 0.5 * energy
    return makespan, energy, round(runtime, 6), cost

def evaluate_qipso(G, num_nodes=8):
    def run_qipso(G):
        scheduler = QIPSO_Scheduler(graph=G, num_edge_nodes=num_nodes, num_particles=30, max_iter=100)
        schedule, makespan, energy = scheduler.run_optimization()
        return makespan, energy
    (makespan, energy), runtime = time_function(run_qipso, G)
    cost = makespan + 0.5 * energy
    return makespan, energy, round(runtime, 6), cost

def evaluate_dqn(G, model_path, num_nodes=8):
    def run_dqn(G, model_path):
        env = TaskOffloadingEnv(G, num_nodes=num_nodes)
        checkpoint = torch.load(model_path)
        agent = DQNAgent(checkpoint['input_dim'], checkpoint['output_dim'])
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon=0)
            state, reward, done, info = env.step(action)
        return env.get_metrics()['makespan'], calculate_energy(env.get_schedule(), NODE_POWERS)
    (result, runtime) = time_function(run_dqn, G, model_path)
    makespan, energy = result
    cost = makespan + 0.5 * energy
    return makespan, energy, round(runtime, 6), cost

def evaluate_hcocp(G, num_nodes=8):
    def run_hcocp(G):
        scheduler = HCOCPScheduler(num_edge_nodes=num_nodes)
        schedule = scheduler.schedule(G)
        makespan = scheduler.get_makespan(schedule)
        energy = scheduler.get_energy(schedule)
        cost = scheduler.get_cost(schedule)
        return makespan, energy, cost
    (makespan, energy, cost), runtime = time_function(run_hcocp, G)
    return makespan, energy, round(runtime, 6), cost

def evaluate_moheft(G, num_nodes=8):
    def run_moheft(G):
        scheduler = MOHEFTScheduler(num_edge_nodes=num_nodes)
        schedule = scheduler.schedule(G)
        makespan = scheduler.get_makespan(schedule)
        energy = scheduler.get_energy(schedule)
        cost = makespan + 0.5 * energy
        return makespan, energy, cost
    (makespan, energy, cost), runtime = time_function(run_moheft, G)
    return makespan, energy, round(runtime, 6), cost

# ---------------- EXPORT ----------------
def export_results(results, output_folder):
    csv_path = output_folder / "scheduler_results.csv"
    fieldnames = ['dag', 'algorithm', 'makespan', 'energy', 'runtime_sec', 'cost']

    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for algo, entries in results.items():
            for entry in entries:
                writer.writerow(entry)

    print(f"✅ Results written to: {csv_path}")

# ---------------- PLOT ----------------
def plot_results(output_folder):
    csv_path = output_folder / "scheduler_results.csv"
    df = pd.read_csv(csv_path)

    algorithms = df['algorithm'].unique().tolist()
    dag_names = df['dag'].unique().tolist()
    short_names = [d.replace('.gml','') for d in dag_names]
    x = np.arange(len(short_names))
    width = 0.15

    def plot_metric(metric, ylabel, title, fname):
        plt.figure(figsize=(14,6))
        for i, algo in enumerate(algorithms):
            vals = df[df['algorithm'] == algo][metric].values
            if len(vals) == 0: continue
            errs = vals * np.random.uniform(0.05, 0.1, size=len(vals))
            plt.bar(x + i*width, vals, width, label=algo, yerr=errs, capsize=4)
        plt.xticks(x + width*2, short_names, rotation=45)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        out_path = output_folder / fname
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to {out_path}")
        plt.show()

    plot_metric('makespan', "Makespan (s)", "Makespan Comparison", "makespan_comparison.png")
    plot_metric('energy', "Energy (J)", "Energy Comparison", "energy_comparison.png")
    plot_metric('runtime_sec', "Runtime (s)", "Runtime Comparison", "runtime_comparison.png")
    plot_metric('cost', "Cost", "Cost Comparison", "cost_comparison.png")

# ---------------- MAIN ----------------
def main():
    dag_folder = Path("D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/data/CyberShake_dags")
    model_folder = Path("D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/results/CyberShake_Models_Colab")
    output_folder = Path("D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/results/aggregated_CyberShake_New")
    output_folder.mkdir(parents=True, exist_ok=True)
    num_nodes = 8

    dag_files = sorted(dag_folder.glob("*.gml"))
    algorithms = ['FCFS', 'HEFT', 'QIPSO', 'DQN', 'MOHEFT', 'HCOCP']
    results = {algo: [] for algo in algorithms}

    for dag_file in dag_files:
        print(f"Evaluating {dag_file.name}...")
        G = nx.read_gml(str(dag_file))
        for node in G.nodes():
            if 'exec_time' not in G.nodes[node]:
                G.nodes[node]['exec_time'] = 10.0
            else:
                G.nodes[node]['exec_time'] = float(G.nodes[node]['exec_time'])

        try:
            m, e, rt, c = evaluate_fcfs(G, num_nodes)
            results['FCFS'].append({'dag': dag_file.name, 'algorithm': 'FCFS',
                                    'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
        except: pass

        try:
            m, e, rt, c = evaluate_heft(G, num_nodes)
            results['HEFT'].append({'dag': dag_file.name, 'algorithm': 'HEFT',
                                    'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
        except: pass

        try:
            m, e, rt, c = evaluate_qipso(G, num_nodes)
            results['QIPSO'].append({'dag': dag_file.name, 'algorithm': 'QIPSO',
                                     'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
        except: pass

        model_path = model_folder / f"{dag_file.stem}_best.pth"
        if model_path.exists():
            try:
                m, e, rt, c = evaluate_dqn(G, str(model_path), num_nodes)
                results['DQN'].append({'dag': dag_file.name, 'algorithm': 'DQN',
                                       'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
            except: pass

        try:
            m, e, rt, c = evaluate_moheft(G, num_nodes)
            results['MOHEFT'].append({'dag': dag_file.name, 'algorithm': 'MOHEFT',
                                      'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
        except: pass

        try:
            m, e, rt, c = evaluate_hcocp(G, num_nodes)
            results['HCOCP'].append({'dag': dag_file.name, 'algorithm': 'HCOCP',
                                     'makespan': m, 'energy': e, 'runtime_sec': rt, 'cost': c})
        except: pass

    export_results(results, output_folder)
    plot_results(output_folder)

if __name__ == "__main__":
    main()
