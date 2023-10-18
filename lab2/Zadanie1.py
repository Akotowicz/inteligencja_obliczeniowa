import pygad
# najlepszy wynik - 1630

S1 = ["zegar", "obraz pejzaz", "obraz portret", "radio", "laptop", "lampka nocna", "sztućce",
      "porcelana", "figura z brązu", "skórzana torebka", "odkurzacz"]
S = [(100, 7), (300, 7), (200, 6), (40, 2), (500, 5), (70, 6), (100, 1), (250, 3), (300, 10), (280, 3), (300, 15)]

gene_space = [0, 1]  # geny to liczby: 0 lub 1

def fitness_func(instance, solution, solution_idx):
    sumaWaga = 0
    sumaWartosc = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            sumaWaga += S[i][1]
            sumaWartosc += S[i][0]
    # print(sumaWaga, sumaWartosc)
    if sumaWaga > 25:
        return 0
    return sumaWartosc

fitness_function = fitness_func

sol_per_pop = 10  # ile chromsomów w populacji
num_genes = len(S1)  # ile genow ma chromosom

num_parents_mating = 5  # ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
num_generations = 30  # ile pokolen
keep_parents = 2  # ilu rodzicow zachowac (kilka procent)

parent_selection_type = "sss"  # sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
crossover_type = "single_point"  # w il =u punktach robic krzyzowanie?

mutation_type = "random"
mutation_percent_genes = 15  # mutacja ma dzialac na ilu procent genow?

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

BestSumaWartosc = 0
for i in range(len(solution)):
    if solution[i] == 1:
        BestSumaWartosc += S[i][0]

# tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
print("Predicted output based on the best solution : {prediction}".format(prediction=BestSumaWartosc))

ga_instance.plot_fitness()  # wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
