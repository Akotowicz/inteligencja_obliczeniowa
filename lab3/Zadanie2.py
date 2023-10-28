import matplotlib.pyplot as plt
import random
from aco import AntColony

plt.style.use("dark_background")

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (22, 16),
    (1,5),
    (81,33),
    (2,44),
    (34,78),
    (65,21),
    (72,98),
    (92,44),
    (52,81),
)

def random_coord():
    r = random.randint(0, len(COORDS))
    return r

def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))

plot_nodes()
# colony = AntColony(COORDS, ant_count=20, alpha=0.5, beta=1.2,
#                     pheromone_evaporation_rate=0.4, pheromone_constant=1000.0,
#                     iterations=30)

colony = AntColony(COORDS, ant_count=20, alpha=0.8, beta=1.8,
                    pheromone_evaporation_rate=0.4, pheromone_constant=1000.0,
                    iterations=30)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.show()