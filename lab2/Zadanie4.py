import pygad

S = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]

def checkStep(x, y, maze):
    if x == -1 or x > 9:
        return False
    if y == -1 or y > 9:
        return False
    if maze[y][x] == 0:
        return True  # droga
    if maze[y][x] == 1:
        return False  # ściana

def fitness_func(instance, solution, solution_idx):
    y = 0
    x = 0
    goodAnswers = 0
    for i in range(len(solution)):
        if solution[i] == 0:  # gora
            if checkStep(x, y-1, S) == False:
                continue
            y = y - 1
        if solution[i] == 1:  # prawo
            if checkStep(x+1, y, S) == False:
                continue
            x = x + 1
        if solution[i] == 2:  # dol
            if checkStep(x, y+1, S) == False:
                continue
            y = y + 1
        if solution[i] == 3:  # lewo
            if checkStep(x - 1, y, S) == False:
                continue
            x = x - 1

        if x == 9 and y == 9:  # wyjscie z labiryntu
            return 100

    return 18 - (9 - x + 9 - y)


ga_instance = pygad.GA(gene_space=[0, 1, 2, 3],
                       num_generations=800,
                       num_parents_mating=50,
                       fitness_func=fitness_func,
                       sol_per_pop=100,
                       num_genes=30,
                       parent_selection_type="sss",
                       keep_parents=3,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=2,
                       stop_criteria=["reach_100"])
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance.plot_fitness()
