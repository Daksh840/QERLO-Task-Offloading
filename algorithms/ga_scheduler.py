import random
import numpy as np

class GAScheduler:
    def __init__(self, num_edge_nodes=8, population_size=30, generations=100,
                 mutation_rate=0.1, seed=42, verbose=False):
        self.num_nodes = num_edge_nodes
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.node_powers = [1.0 + 0.2*i for i in range(num_edge_nodes)]
        self.verbose = verbose

        # Set reproducibility
        random.seed(seed)
        np.random.seed(seed)

    def initialize_population(self, G):
        return [[random.randint(0, self.num_nodes-1) for _ in range(len(G.nodes()))]
                for _ in range(self.population_size)]

    def evaluate(self, G, assignment):
        schedule = {}
        time_per_node = [0.0]*self.num_nodes
        for idx, node in enumerate(G.nodes()):
            exec_time = float(G.nodes[node].get('exec_time', 10.0))
            assigned = assignment[idx]
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

    def select_parents(self, population, fitness):
        idxs = np.argsort(fitness)
        return [population[i] for i in idxs[:2]]  # top-2

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1)-2)
        child = parent1[:point] + parent2[point:]
        return child

    def mutate(self, assignment):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(assignment)-1)
            assignment[idx] = random.randint(0, self.num_nodes-1)
        return assignment

    def schedule(self, G):
        population = self.initialize_population(G)
        best_schedule, best_score = None, float('inf')

        for gen in range(self.generations):
            fitness = []
            evals = []
            for assignment in population:
                makespan, energy, schedule = self.evaluate(G, assignment)
                cost = makespan + 0.5*energy
                fitness.append(cost)
                evals.append((makespan, energy, schedule))
            parents = self.select_parents(population, fitness)
            new_population = []
            for _ in range(self.population_size):
                child = self.crossover(random.choice(parents), random.choice(parents))
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
            # track best
            idx_best = np.argmin(fitness)
            if fitness[idx_best] < best_score:
                best_score = fitness[idx_best]
                best_makespan, best_energy, best_schedule = evals[idx_best]

            if self.verbose and gen % 10 == 0:
                print(f"[GA Gen {gen}] Best cost={best_score:.2f}")

        return best_schedule

    def get_makespan(self, schedule):
        return max(task['end_time'] for task in schedule.values())

    def get_energy(self, schedule):
        return sum((s['end_time']-s['start_time'])*self.node_powers[s['assigned_node']]
                   for s in schedule.values())
