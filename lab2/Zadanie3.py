import pygad
import math

S = ['𝑥', '𝑦', 'z', 'v', 'u', '𝑤']

def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)

def fitness_func(instance, solution, solution_idx):
    return endurance(
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
    )
gene_space = {
    'low': 0.00,
    "high": 0.99,
}

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=30,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=len(S),
                       parent_selection_type="sss",
                       keep_parents=2,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=30)
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance.plot_fitness()
