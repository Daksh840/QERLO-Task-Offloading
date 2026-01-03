import random
import numpy as np

class PSOScheduler:
    def __init__(self, num_edge_nodes=8, swarm_size=30, max_iter=100,
                 w=0.5, c1=1.5, c2=1.5, seed=42, verbose=False):
        self.num_nodes = num_edge_nodes
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.node_powers = [1.0 + 0.2*i for i in range(num_edge_nodes)]
        self.verbose = verbose

        # Set reproducibility
        random.seed(seed)
        np.random.seed(seed)

    def evaluate(self, G, assignment):
        schedule = {}
        time_per_node = [0.0]*self.num_nodes
        for idx, node in enumerate(G.nodes()):
            exec_time = float(G.nodes[node].get('exec_time', 10.0))
            assigned = int(assignment[idx]) % self.num_nodes
            start_time = time_per_node[assigned]
            end_time = start_time + exec_time
            time_per_node[assigned] = end_time
            schedule[node] = {
                'assigned_node': assigned,
                'start_time': start_time,
                'end_time': end_time
            }
        makespan = max(time_per_node)
        energy = sum((s['end_time']-s['start_time'])*self.node_powers[s['assigned_node']]
                     for s in schedule.values())
        return makespan, energy, schedule

    def schedule(self, G):
        num_tasks = len(G.nodes())
        # initialize swarm
        particles = [np.random.randint(0, self.num_nodes, num_tasks) for _ in range(self.swarm_size)]
        velocities = [np.random.uniform(-1, 1, num_tasks) for _ in range(self.swarm_size)]

        personal_best = particles.copy()
        personal_best_scores = [float('inf')]*self.swarm_size
        global_best, global_best_score, global_best_sched = None, float('inf'), None

        for it in range(self.max_iter):
            for i, particle in enumerate(particles):
                makespan, energy, schedule = self.evaluate(G, particle)
                score = makespan + 0.5*energy
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particle.copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best = particle.copy()
                    global_best_sched = schedule
            # update particles
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(num_tasks), np.random.rand(num_tasks)
                velocities[i] = (self.w*velocities[i]
                                 + self.c1*r1*(personal_best[i]-particles[i])
                                 + self.c2*r2*(global_best-particles[i]))
                particles[i] = np.round(particles[i] + velocities[i]).astype(int)
                particles[i] = np.clip(particles[i], 0, self.num_nodes-1)

            if self.verbose and it % 10 == 0:
                print(f"[PSO Iter {it}] Best cost={global_best_score:.2f}")

        return global_best_sched

    def get_makespan(self, schedule):
        return max(task['end_time'] for task in schedule.values())

    def get_energy(self, schedule):
        return sum((s['end_time']-s['start_time'])*self.node_powers[s['assigned_node']]
                   for s in schedule.values())
