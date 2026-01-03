"""
MOHEFT: Multi-Objective HEFT Scheduler
Extends HEFT to consider makespan + energy/cost simultaneously.
"""

import numpy as np

class MOHEFTScheduler:
    def __init__(self, num_edge_nodes=8, weight_makespan=0.5, weight_energy=0.5, node_powers=None):
        self.num_edge_nodes = num_edge_nodes
        self.weight_makespan = weight_makespan
        self.weight_energy = weight_energy
        if node_powers is None:
            self.node_powers = [1.0 + 0.2*i for i in range(num_edge_nodes)]
        else:
            self.node_powers = node_powers

    def schedule(self, G):
        """
        Returns: dict {task: {assigned_node, start_time, end_time}}
        """
        ready = [n for n in G.nodes() if G.in_degree(n) == 0]
        scheduled = {}
        available_time = [0.0] * self.num_edge_nodes

        while ready:
            task = ready.pop(0)
            exec_time = float(G.nodes[task].get('exec_time', 1.0))

            # Choose node based on weighted tradeoff
            best_score, best_node, best_start = float('inf'), None, None
            for node in range(self.num_edge_nodes):
                start = max([scheduled[p]['end_time'] for p in G.predecessors(task)] or [0.0])
                start = max(start, available_time[node])
                end = start + exec_time

                makespan_est = end
                energy_est = exec_time * self.node_powers[node]
                score = self.weight_makespan * makespan_est + self.weight_energy * energy_est

                if score < best_score:
                    best_score, best_node, best_start = score, node, start

            scheduled[task] = {
                'assigned_node': best_node,
                'start_time': best_start,
                'end_time': best_start + exec_time
            }
            available_time[best_node] = scheduled[task]['end_time']

            for succ in G.successors(task):
                if all(pred in scheduled for pred in G.predecessors(succ)):
                    ready.append(succ)

        return scheduled

    def get_makespan(self, schedule):
        return max(info['end_time'] for info in schedule.values())

    def get_energy(self, schedule):
        return sum((info['end_time'] - info['start_time']) *
                   self.node_powers[info['assigned_node']] for info in schedule.values())
